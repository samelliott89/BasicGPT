"""
Supervised Fine-Tuning (SFT) Trainer

Train a language model to follow instructions using supervised learning.
Supports both HuggingFace and BasicGPT models through unified interface.
"""

from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Tokenizer, load_model
from training.finetune.config import InstructionTemplate, SFTConfig
from utils.device import get_best_device


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following data.

    Tokenizes instruction-input-output triplets for causal language modeling.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Tokenizer,
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

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # Extract fields
        instruction = item.get(self.config.instruction_field, "")
        input_text = item.get(self.config.input_field, "")
        output = item.get(self.config.output_field, "")

        # Format the full text
        full_text = self.template.format_full(instruction, input_text, output)

        # Tokenize using our unified tokenizer interface
        encoding = self.tokenizer(
            full_text,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels = input_ids (HuggingFace shifts internally)
        # Only mask padding, don't mask prompt (simpler, works better)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTTrainer:
    """
    Trainer for supervised fine-tuning.

    Works with both HuggingFace and BasicGPT models.

    Example:
        # With GPT-2
        config = SFTConfig(model_name="gpt2")
        trainer = SFTTrainer(config)
        trainer.train()

        # With your trained model
        config = SFTConfig(
            model_name="./checkpoints/best/checkpoint.pt",
            model_type="basicgpt"
        )
        trainer = SFTTrainer(config)
        trainer.train()
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.device = get_best_device()

        # Load model and tokenizer using unified loader
        self.model, self.tokenizer = load_model(
            config.model_name,
            model_type=config.model_type,
            device=self.device,
        )

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight", "ln_"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

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

        print("\nTraining configuration:")
        print(f"  Model: {self.config.model_name}")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Samples: {len(train_loader.dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(
            f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}"
        )
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print()

        # Optimizer and scheduler
        optimizer = self._create_optimizer()

        # Simple linear warmup + decay scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.use_fp16 else None

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)

            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc="Training")

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                if self.config.use_fp16:
                    with torch.cuda.amp.autocast():
                        output = self.model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = output.loss
                else:
                    output = self.model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = output.loss

                # Skip batch if loss is None or NaN
                if loss is None or torch.isnan(loss):
                    continue

                loss = loss / self.config.gradient_accumulation_steps

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
                        progress_bar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{lr:.2e}",
                            }
                        )

                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step, epoch_loss / num_batches)

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

            # Save epoch checkpoint
            self._save_checkpoint(global_step, avg_epoch_loss, epoch=epoch + 1)

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

    def _save_checkpoint(self, step: int, loss: float, epoch: int = None):
        """Save a training checkpoint."""
        if epoch:
            checkpoint_name = f"epoch_{epoch}"
        else:
            checkpoint_name = f"step_{step}"

        checkpoint_path = self.output_dir / checkpoint_name

        print(f"\n💾 Saving checkpoint: {checkpoint_name}")

        # Use our unified save interface
        self.model.save(str(checkpoint_path))

        # Save tokenizer if HuggingFace
        if hasattr(self.tokenizer, "save"):
            self.tokenizer.save(str(checkpoint_path))

        # Save training state
        torch.save(
            {
                "step": step,
                "loss": loss,
                "config": self.config.__dict__,
            },
            checkpoint_path / "training_state.pt",
        )

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate a response for a given prompt."""
        self.model.eval()

        # Format prompt
        template = InstructionTemplate()
        if "### Instruction:" not in prompt:
            formatted_prompt = template.format_prompt(prompt, "")
        else:
            formatted_prompt = prompt

        # Tokenize
        encoding = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())

        # Extract response
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text[len(formatted_prompt) :].strip()

        return response


if __name__ == "__main__":
    print("SFT Trainer")
    print("=" * 40)
    print()
    print("Usage:")
    print("  # With GPT-2")
    print("  config = SFTConfig(model_name='gpt2')")
    print("  trainer = SFTTrainer(config)")
    print("  trainer.train()")
    print()
    print("  # With your trained BasicGPT")
    print("  config = SFTConfig(")
    print("      model_name='./checkpoints/best/checkpoint.pt',")
    print("      model_type='basicgpt'")
    print("  )")
    print("  trainer = SFTTrainer(config)")
    print("  trainer.train()")
