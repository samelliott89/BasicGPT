"""
Code Generation SFT Trainer

Train a model to generate working PyTorch nn.Module code
from natural language descriptions.
"""

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.codegen.config import CodeGenSFTConfig, EXAMPLE_CHALLENGES
from utils.device import get_best_device


class CodeChallengeDataset(Dataset):
    """
    Dataset of (prompt, code) pairs for code generation SFT.
    
    Handles prompt and response separately to ensure:
    1. Full prompts are preserved (truncated if too long, with warning)
    2. Full code responses are preserved (truncated if too long, with warning)
    3. Total sequence fits in model context
    
    JSONL format:
        {"prompt": "Create a residual block...", "code": "class ResBlock..."}
    """
    
    def __init__(
        self,
        data_path: str | None,
        tokenizer,
        max_prompt_length: int = 256,
        max_response_length: int = 768,
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_length = max_prompt_length + max_response_length
        self.challenges = []
        self.truncation_warnings = {"prompt": 0, "response": 0}
        
        # Load from JSONL if exists
        if data_path and Path(data_path).exists():
            print(f"Loading challenges from: {data_path}")
            with open(data_path) as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        self.challenges.append(json.loads(line))
        else:
            print("⚠️  No dataset file found, using example challenges")
            self.challenges = EXAMPLE_CHALLENGES
        
        if max_samples:
            self.challenges = self.challenges[:max_samples]
        
        # Validate sequence lengths
        self._validate_lengths()
        
        print(f"✓ Loaded {len(self.challenges)} code challenges")
        print(f"  Max prompt: {max_prompt_length} tokens")
        print(f"  Max response: {max_response_length} tokens")
        print(f"  Total max: {self.max_length} tokens")
    
    def _validate_lengths(self):
        """Check if any samples will be truncated and warn."""
        truncated_prompts = 0
        truncated_responses = 0
        
        for item in self.challenges:
            prompt_text = f"# Task: {item['prompt']}\n\nimport torch\nimport torch.nn as nn\n\n"
            prompt_tokens = len(self.tokenizer.encode(prompt_text))
            
            response_tokens = len(self.tokenizer.encode(item["code"]))
            
            if prompt_tokens > self.max_prompt_length:
                truncated_prompts += 1
            if response_tokens > self.max_response_length:
                truncated_responses += 1
        
        if truncated_prompts > 0:
            print(f"⚠️  {truncated_prompts}/{len(self.challenges)} prompts will be truncated")
        if truncated_responses > 0:
            print(f"⚠️  {truncated_responses}/{len(self.challenges)} responses will be truncated")
            print(f"   Consider increasing max_response_length")
    
    def __len__(self):
        return len(self.challenges)
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        item = self.challenges[idx]
        
        # Separate prompt and response
        prompt_text = f"# Task: {item['prompt']}\n\nimport torch\nimport torch.nn as nn\n\n"
        response_text = item["code"]
        
        # Tokenize separately to control lengths
        prompt_enc = self.tokenizer(
            prompt_text,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        response_enc = self.tokenizer(
            response_text,
            max_length=self.max_response_length,
            truncation=True,
            add_special_tokens=False,  # Don't add BOS again
        )
        
        # Combine: prompt + response
        input_ids = prompt_enc["input_ids"] + response_enc["input_ids"]
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        # Labels = input_ids, mask padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CodeGenSFTTrainer:
    """
    Trainer for code generation SFT.
    
    Trains a code model (e.g., DeepSeek-Coder) to generate working
    PyTorch code from natural language descriptions.
    
    Example:
        config = CodeGenSFTConfig(
            model_name="deepseek-ai/deepseek-coder-1.3b-base",
            dataset_path="./data/pycode/challenges.jsonl",
        )
        trainer = CodeGenSFTTrainer(config)
        trainer.train()
    """
    
    def __init__(self, config: CodeGenSFTConfig):
        self.config = config
        self.device = get_best_device()
        
        # Load model
        print(f"Loading model: {config.model_name}")
        # Always load in float32 for stable training, use AMP for speed
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,  # Float32 for stable training
            trust_remote_code=True,  # DeepSeek models need this
        )
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Model loaded: {self._count_params():.1f}M parameters")
    
    def _count_params(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6
    
    def _create_dataloader(self) -> DataLoader:
        dataset = CodeChallengeDataset(
            data_path=self.config.dataset_path,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
            max_response_length=self.config.max_response_length,
            max_samples=self.config.max_samples,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    
    def _create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "ln_"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def train(self):
        """Run the training loop."""
        print("\n" + "=" * 60)
        print("Starting Code Generation SFT")
        print("=" * 60)
        
        train_loader = self._create_dataloader()
        total_batches = len(train_loader) * self.config.num_epochs
        total_steps = max(1, total_batches // self.config.gradient_accumulation_steps)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        print(f"\nConfiguration:")
        print(f"  Model: {self.config.model_name}")
        print(f"  Samples: {len(train_loader.dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Grad accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Total steps: {total_steps}")
        print()
        
        optimizer = self._create_optimizer()
        
        # Learning rate scheduler (handles edge case of very small datasets)
        def lr_lambda(step):
            # Warmup phase
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            # Decay phase
            decay_steps = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / decay_steps
            return max(0.1, 1.0 - progress)  # Don't go below 0.1
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Mixed precision with AMP
        is_cuda = torch.cuda.is_available() and "cuda" in str(self.device)
        use_amp = self.config.use_fp16 and is_cuda
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        
        if use_amp:
            print("  Using AMP (FP16 mixed precision)")
        
        # Training
        self.model.train()
        global_step = 0
        best_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)
            
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                
                if loss is None:
                    print(f"  Warning: loss is None at batch {batch_idx}")
                    continue
                if torch.isnan(loss):
                    print(f"  Warning: NaN loss at batch {batch_idx}, skipping")
                    optimizer.zero_grad()  # Clear any gradients
                    continue
                
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward
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
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
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
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self._save_checkpoint(global_step, avg_epoch_loss, epoch=epoch + 1)
            
            # Save best
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self._save_checkpoint(global_step, avg_epoch_loss, name="best")
        
        print("\n" + "=" * 60)
        print("Code Generation SFT Complete!")
        print("=" * 60)
    
    def _save_checkpoint(self, step: int, loss: float, epoch: int = None, name: str = None):
        """Save checkpoint."""
        if name:
            checkpoint_name = name
        elif epoch:
            checkpoint_name = f"epoch_{epoch}"
        else:
            checkpoint_name = f"step_{step}"
        
        checkpoint_path = self.output_dir / checkpoint_name
        print(f"\n💾 Saving: {checkpoint_name}")
        
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        torch.save({
            "step": step,
            "loss": loss,
            "config": self.config.__dict__,
        }, checkpoint_path / "training_state.pt")
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate code from a prompt."""
        self.model.eval()
        
        # Format prompt
        formatted = f"# Task: {prompt}\n\nimport torch\nimport torch.nn as nn\n\n"
        
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                # Use greedy decoding to avoid NaN issues with sampling
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding - more stable
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            except RuntimeError as e:
                # Fallback to greedy if sampling fails
                print(f"  Warning: Generation error, using greedy: {e}")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the code (after the prompt)
        if "class " in generated:
            # Find the class definition
            code_start = generated.find("class ")
            if code_start != -1:
                return generated[code_start:]
        
        return generated[len(formatted):]


if __name__ == "__main__":
    print("Code Generation SFT Trainer")
    print("=" * 40)
    print()
    print("Usage:")
    print("  from training.codegen import CodeGenSFTTrainer, CodeGenSFTConfig")
    print()
    print("  config = CodeGenSFTConfig(")
    print('      model_name="deepseek-ai/deepseek-coder-1.3b-base",')
    print('      dataset_path="./data/pycode/challenges.jsonl",')
    print("  )")
    print("  trainer = CodeGenSFTTrainer(config)")
    print("  trainer.train()")

