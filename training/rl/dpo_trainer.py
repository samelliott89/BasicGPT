"""
Direct Preference Optimization (DPO) Trainer

DPO is a simpler alternative to RLHF that directly optimizes the policy
without needing a separate reward model.

Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from typing import Optional, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import copy

from training.rl.config import DPOConfig


class PreferenceDataset(Dataset):
    """
    Dataset for preference pairs (chosen, rejected).
    
    Each sample contains:
    - prompt: The input prompt
    - chosen: The preferred response
    - rejected: The less preferred response
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        config: DPOConfig,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.config = config
        
        # Load dataset
        print(f"Loading preference dataset: {dataset_name}")
        self.data = load_dataset(dataset_name, split=split)
        
        if config.max_samples:
            self.data = self.data.select(range(min(config.max_samples, len(self.data))))
        
        print(f"✓ Loaded {len(self.data)} preference pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # For Anthropic HH-RLHF format, chosen/rejected are full conversations
        # For other formats, we might need to concatenate prompt + response
        chosen = item.get(self.config.chosen_field, "")
        rejected = item.get(self.config.rejected_field, "")
        
        # Tokenize chosen
        chosen_enc = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize rejected
        rejected_enc = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization (DPO).
    
    DPO optimizes the policy directly using preference pairs:
    
    L_DPO = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x) - log_pi_ref(y_w|x) + log_pi_ref(y_l|x))))
    
    where:
    - y_w = chosen (winning) response
    - y_l = rejected (losing) response
    - pi = policy model
    - pi_ref = reference model (frozen)
    - beta = KL penalty coefficient
    
    Example:
        config = DPOConfig(model_name="path/to/sft_model")
        trainer = DPOTrainer(config)
        trainer.train()
    """
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self._load_models()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_models(self):
        """Load policy and reference models."""
        print(f"Loading policy model: {self.config.model_name}")
        
        if self.config.model_type == "huggingface":
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
            
            # Add padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            # Load reference model (frozen copy)
            print(f"Loading reference model: {self.config.ref_model_name}")
            self.ref_model = GPT2LMHeadModel.from_pretrained(self.config.ref_model_name)
            self.ref_model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise NotImplementedError("BasicGPT DPO not yet implemented")
        
        # Move to device
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        print(f"✓ Models loaded on {self.device}")
        print(f"  Policy parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_log_probs(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for each token.
        
        Args:
            model: Language model
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Per-token log probabilities [batch, seq_len-1]
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab]
        
        # Shift logits and labels for next-token prediction
        # logits: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        shift_labels = input_ids[:, 1:]    # [batch, seq_len-1]
        shift_mask = attention_mask[:, 1:] # [batch, seq_len-1]
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len-1]
        
        # Mask padding and sum
        token_log_probs = token_log_probs * shift_mask
        
        # Sum log probs per sequence (total log prob)
        sequence_log_probs = token_log_probs.sum(dim=-1)  # [batch]
        
        return sequence_log_probs
    
    def _compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the DPO loss.
        
        DPO loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        
        where log_ratio = log(pi(y|x)) - log(pi_ref(y|x))
        """
        # Compute log ratios (policy vs reference)
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO logits
        logits = self.config.beta * (chosen_log_ratios - rejected_log_ratios)
        
        # DPO loss (binary cross entropy with sigmoid)
        # We want chosen > rejected, so labels = 1
        loss = -F.logsigmoid(logits).mean()
        
        # Compute metrics
        with torch.no_grad():
            # Accuracy: how often does policy prefer chosen?
            accuracy = (logits > 0).float().mean().item()
            
            # Reward margin
            chosen_rewards = self.config.beta * chosen_log_ratios
            rejected_rewards = self.config.beta * rejected_log_ratios
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "reward_margin": reward_margin,
            "chosen_log_ratio": chosen_log_ratios.mean().item(),
            "rejected_log_ratio": rejected_log_ratios.mean().item(),
        }
        
        return loss, metrics
    
    def train(self):
        """Run the DPO training loop."""
        print("\n" + "=" * 60)
        print("Starting Direct Preference Optimization (DPO)")
        print("=" * 60)
        
        # Create dataloader
        train_loader = self._create_dataloader()
        total_steps = len(train_loader) * self.config.num_epochs
        total_steps = total_steps // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        print(f"\nTraining configuration:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Samples: {len(train_loader.dataset)}")
        print(f"  Beta (KL coefficient): {self.config.beta}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Total steps: {total_steps}")
        print()
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.use_fp16 else None
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)
            
            epoch_metrics = {"loss": 0, "accuracy": 0, "reward_margin": 0}
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                
                # Forward pass
                if self.config.use_fp16:
                    with torch.cuda.amp.autocast():
                        # Policy log probs
                        policy_chosen_logps = self._get_log_probs(
                            self.model, chosen_input_ids, chosen_attention_mask
                        )
                        policy_rejected_logps = self._get_log_probs(
                            self.model, rejected_input_ids, rejected_attention_mask
                        )
                        
                        # Reference log probs (no grad)
                        with torch.no_grad():
                            ref_chosen_logps = self._get_log_probs(
                                self.ref_model, chosen_input_ids, chosen_attention_mask
                            )
                            ref_rejected_logps = self._get_log_probs(
                                self.ref_model, rejected_input_ids, rejected_attention_mask
                            )
                        
                        # Compute DPO loss
                        loss, metrics = self._compute_dpo_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            ref_chosen_logps,
                            ref_rejected_logps,
                        )
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    # Same without autocast
                    policy_chosen_logps = self._get_log_probs(
                        self.model, chosen_input_ids, chosen_attention_mask
                    )
                    policy_rejected_logps = self._get_log_probs(
                        self.model, rejected_input_ids, rejected_attention_mask
                    )
                    
                    with torch.no_grad():
                        ref_chosen_logps = self._get_log_probs(
                            self.ref_model, chosen_input_ids, chosen_attention_mask
                        )
                        ref_rejected_logps = self._get_log_probs(
                            self.ref_model, rejected_input_ids, rejected_attention_mask
                        )
                    
                    loss, metrics = self._compute_dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                    )
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update metrics
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k] += v
                num_batches += 1
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_metrics["loss"] / num_batches
                        avg_acc = epoch_metrics["accuracy"] / num_batches
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "acc": f"{avg_acc:.2%}",
                        })
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step, epoch_metrics, num_batches)
            
            # End of epoch summary
            avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
            print(f"\nEpoch {epoch + 1} complete:")
            print(f"  Loss: {avg_metrics['loss']:.4f}")
            print(f"  Accuracy: {avg_metrics['accuracy']:.2%}")
            print(f"  Reward margin: {avg_metrics['reward_margin']:.4f}")
        
        print("\n" + "=" * 60)
        print("DPO Training complete!")
        print("=" * 60)
    
    def _create_dataloader(self) -> DataLoader:
        """Create dataloader for preference pairs."""
        dataset = PreferenceDataset(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer,
            config=self.config,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    
    def _save_checkpoint(self, step: int, metrics: Dict, num_batches: int):
        """Save a training checkpoint."""
        checkpoint_path = self.output_dir / f"step_{step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving checkpoint: step_{step}")
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        avg_metrics = {k: v / num_batches for k, v in metrics.items()}
        torch.save({
            "step": step,
            "metrics": avg_metrics,
            "config": self.config.__dict__,
        }, checkpoint_path / "training_state.pt")


if __name__ == "__main__":
    print("DPO Trainer Module")
    print("=" * 40)
    print()
    print("Direct Preference Optimization (DPO) trains a model to prefer")
    print("'chosen' responses over 'rejected' responses.")
    print()
    print("Example usage:")
    print("  from training.rl import DPOTrainer, DPOConfig")
    print()
    print("  config = DPOConfig(")
    print("      model_name='path/to/sft_checkpoint',  # Start from SFT model")
    print("      dataset_name='Anthropic/hh-rlhf',")
    print("      beta=0.1,  # KL penalty")
    print("  )")
    print()
    print("  trainer = DPOTrainer(config)")
    print("  trainer.train()")

