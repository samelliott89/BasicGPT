"""
Supervised Fine-Tuning (SFT) Trainer

Train a language model to follow instructions using supervised learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import math

from training.finetune.config import SFTConfig, InstructionTemplate


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following data.
    
    Tokenizes instruction-input-output triplets for causal language modeling.
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        config: SFTConfig,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.template = InstructionTemplate()
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        self.data = load_dataset(dataset_name, split=split)
        
        if config.max_samples:
            self.data = self.data.select(range(min(config.max_samples, len(self.data))))
        
        print(f"✓ Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Extract fields
        instruction = item.get(self.config.instruction_field, "")
        input_text = item.get(self.config.input_field, "")
        output = item.get(self.config.output_field, "")
        
        # Format the full text
        full_text = self.template.format_full(instruction, input_text, output)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # For causal LM, labels = input_ids (shifted internally by the model)
        # We mask the prompt portion to only train on the response
        labels = input_ids.clone()
        
        # Find where the response starts (after "### Response:\n")
        prompt_only = self.template.format_prompt(instruction, input_text)
        prompt_tokens = self.tokenizer(prompt_only, return_tensors="pt")["input_ids"].shape[1]
        
        # Mask the prompt portion (set to -100 to ignore in loss)
        labels[:prompt_tokens] = -100
        
        # Also mask padding
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTTrainer:
    """
    Trainer for supervised fine-tuning.
    
    Example:
        config = SFTConfig(model_name="gpt2", dataset_name="tatsu-lab/alpaca")
        trainer = SFTTrainer(config)
        trainer.train()
    """
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Load the model and tokenizer."""
        if self.config.model_type == "huggingface":
            print(f"Loading HuggingFace model: {self.config.model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
            
            # Add padding token (GPT-2 doesn't have one by default)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            # Load BasicGPT model
            from model import GPT
            print(f"Loading BasicGPT checkpoint: {self.config.model_name}")
            self.model = GPT.from_pretrained(self.config.model_name, device=self.device)
            
            # Use tiktoken tokenizer
            from data import Tokenizer
            self.tokenizer = Tokenizer()
        
        self.model.to(self.device)
        print(f"✓ Model loaded on {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_dataloader(self, split: str = "train") -> DataLoader:
        """Create a dataloader for the dataset."""
        dataset = InstructionDataset(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer,
            config=self.config,
            split=split,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )
    
    def _create_optimizer(self) -> tuple:
        """Create optimizer and scheduler."""
        # Separate weight decay for different parameter groups
        no_decay = ["bias", "LayerNorm.weight", "ln_"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        
        return optimizer
    
    def train(self):
        """Run the training loop."""
        print("\n" + "=" * 60)
        print("Starting Supervised Fine-Tuning")
        print("=" * 60)
        
        # Create dataloader
        train_loader = self._create_dataloader("train")
        total_steps = len(train_loader) * self.config.num_epochs
        total_steps = total_steps // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        print(f"\nTraining configuration:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Samples: {len(train_loader.dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print()
        
        # Create optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Mixed precision
        scaler = None
        if self.config.use_fp16:
            scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)
            
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                if self.config.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / self.config.gradient_accumulation_steps
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
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
                        avg_loss = epoch_loss / num_batches
                        lr = scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step, epoch_loss / num_batches)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
            
            # Save end-of-epoch checkpoint
            self._save_checkpoint(global_step, avg_epoch_loss, is_epoch_end=True, epoch=epoch + 1)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
    
    def _save_checkpoint(
        self, 
        step: int, 
        loss: float, 
        is_epoch_end: bool = False,
        epoch: int = None
    ):
        """Save a training checkpoint."""
        if is_epoch_end:
            checkpoint_name = f"epoch_{epoch}"
        else:
            checkpoint_name = f"step_{step}"
        
        checkpoint_path = self.output_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving checkpoint: {checkpoint_name}")
        
        # Save model
        if self.config.model_type == "huggingface":
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
        else:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "step": step,
                "loss": loss,
            }, checkpoint_path / "checkpoint.pt")
        
        # Save training state
        torch.save({
            "step": step,
            "loss": loss,
            "config": self.config.__dict__,
        }, checkpoint_path / "training_state.pt")
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate a response for a given prompt."""
        self.model.eval()
        
        # Format the prompt
        template = InstructionTemplate()
        
        # Check if prompt is already formatted
        if "### Instruction:" not in prompt:
            formatted_prompt = template.format_prompt(prompt, "")
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text[len(formatted_prompt):].strip()
        
        return response


if __name__ == "__main__":
    # Quick test
    print("Testing SFT Trainer...")
    
    config = SFTConfig(
        model_name="gpt2",
        dataset_name="tatsu-lab/alpaca",
        max_samples=100,  # Small for testing
        num_epochs=1,
        batch_size=2,
    )
    
    trainer = SFTTrainer(config)
    
    # Test generation before training
    print("\nTesting generation (before training):")
    response = trainer.generate("What is machine learning?")
    print(f"Response: {response[:200]}...")
    
    print("\n✓ SFT Trainer ready!")
    print("To train: trainer.train()")

