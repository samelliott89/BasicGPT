"""
Reward Model for RLHF

A reward model learns to predict human preferences by scoring responses.
It takes (prompt + response) and outputs a scalar reward.

Training uses pairwise ranking loss (Bradley-Terry model):
    loss = -log(sigmoid(reward_chosen - reward_rejected))

This pushes the model to score "chosen" responses higher than "rejected" ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM

from model import Tokenizer, load_model, ModelType
from training.rl.config import RewardModelConfig
from utils.device import get_best_device


# =============================================================================
# Reward Model Architecture
# =============================================================================


class RewardModel(nn.Module):
    """
    Reward model that predicts a scalar score for (prompt + response).

    Architecture:
        Base LLM (frozen or trainable) → Last hidden state → Linear → Scalar

    The scalar represents "how good" the response is according to human preferences.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        freeze_base: bool = False,
    ):
        """
        Args:
            base_model: The underlying language model (e.g., GPT-2).
            hidden_size: Dimension of the model's hidden states.
            freeze_base: If True, freeze the base model weights.
        """
        super().__init__()
        self.base_model = base_model
        self.freeze_base = freeze_base

        # Remove the LM head if present (we only need hidden states)
        if hasattr(self.base_model, "lm_head"):
            # Keep the transformer, remove the LM head
            self.transformer = self.base_model.transformer
        else:
            self.transformer = self.base_model

        # Freeze base model if requested
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Scalar reward head: maps hidden_size → 1
        self.reward_head = nn.Linear(hidden_size, 1)

        # Initialize with small weights for stable training
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute reward scores for input sequences.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Mask for padding (batch_size, seq_len)

        Returns:
            Reward scores (batch_size,) - higher = better response
        """
        # Get hidden states from base model
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback for models that return tuple
            hidden_states = outputs[0]

        # Get the hidden state of the last non-padding token
        if attention_mask is not None:
            # Find last non-padding position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            # No padding, use last token
            last_hidden = hidden_states[:, -1, :]

        # Predict scalar reward
        reward = self.reward_head(last_hidden).squeeze(-1)

        return reward

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        freeze_base: bool = False,
        device: torch.device | None = None,
    ) -> "RewardModel":
        """
        Create a reward model from a pretrained model.

        Args:
            model_name: HuggingFace model name or path (e.g., "gpt2", "gpt2-medium")
            freeze_base: Whether to freeze the base model weights
            device: Device to load model on

        Returns:
            Initialized RewardModel
        """
        device = device or get_best_device()

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = base_model.config.hidden_size

        # Create reward model
        reward_model = cls(base_model, hidden_size, freeze_base)
        reward_model.to(device)

        return reward_model

    def save(self, path: str | Path):
        """Save the reward model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "hidden_size": self.reward_head.in_features,
                "freeze_base": self.freeze_base,
            },
            path / "reward_model.pt",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        base_model_name: str = "gpt2",
        device: torch.device | None = None,
    ) -> "RewardModel":
        """Load a saved reward model."""
        device = device or get_best_device()
        path = Path(path)

        checkpoint = torch.load(path / "reward_model.pt", map_location=device)

        # Recreate base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        hidden_size = checkpoint["hidden_size"]
        freeze_base = checkpoint.get("freeze_base", False)

        # Create and load reward model
        reward_model = cls(base_model, hidden_size, freeze_base)
        reward_model.load_state_dict(checkpoint["model_state_dict"])
        reward_model.to(device)

        return reward_model


# =============================================================================
# Preference Dataset
# =============================================================================


class PreferenceDataset(Dataset):
    """
    Dataset of (prompt, chosen, rejected) triplets for reward model training.

    Each sample contains:
        - chosen: The response humans preferred
        - rejected: The response humans rejected

    Common datasets:
        - "Anthropic/hh-rlhf": Human helpfulness/harmlessness
        - "stanfordnlp/SHP": Reddit preferences
        - "argilla/ultrafeedback-binarized-preferences": GPT-4 judged
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Tokenizer,
        max_length: int = 512,
        max_samples: int | None = None,
        split: str = "train",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field

        print(f"Loading preference dataset: {dataset_name}")
        self.data = load_dataset(dataset_name, split=split)

        if max_samples:
            self.data = self.data.select(range(min(max_samples, len(self.data))))

        print(f"✓ Loaded {len(self.data)} preference pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        chosen = item[self.chosen_field]
        rejected = item[self.rejected_field]

        # Tokenize both responses
        chosen_enc = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        rejected_enc = self.tokenizer(
            rejected,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# =============================================================================
# Reward Model Trainer
# =============================================================================


class RewardTrainer:
    """
    Trainer for reward model using pairwise ranking loss.

    The training objective is:
        loss = -log(sigmoid(r_chosen - r_rejected))

    This is the Bradley-Terry model, which learns to rank chosen > rejected.

    Example:
        config = RewardModelConfig(model_name="gpt2")
        trainer = RewardTrainer(config)
        trainer.train()
    """

    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.device = get_best_device()

        # Create reward model
        print(f"Loading base model: {config.model_name}")
        self.model = RewardModel.from_pretrained(
            config.model_name,
            freeze_base=False,  # Train the whole model
            device=self.device,
        )

        # Load tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Wrap in our unified interface for consistency
        self.tokenizer_wrapper = Tokenizer(self.tokenizer, ModelType.HUGGINGFACE)

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_dataloader(self, split: str = "train") -> DataLoader:
        """Create dataloader for preference data."""
        dataset = PreferenceDataset(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer_wrapper,
            max_length=self.config.max_length,
            max_samples=getattr(self.config, "max_samples", None),
            split=split,
            chosen_field=self.config.chosen_field,
            rejected_field=self.config.rejected_field,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )

    def train(self):
        """Run the reward model training loop."""
        print("\n" + "=" * 60)
        print("Starting Reward Model Training")
        print("=" * 60)

        train_loader = self._create_dataloader("train")

        print("\nTraining configuration:")
        print(f"  Base model: {self.config.model_name}")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Samples: {len(train_loader.dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Epochs: {self.config.num_epochs}")
        print()

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)

            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc="Training")

            for batch in progress_bar:
                # Move to device
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.device)

                # Forward pass for both
                reward_chosen = self.model(chosen_ids, chosen_mask)
                reward_rejected = self.model(rejected_ids, rejected_mask)

                # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
                # This pushes reward_chosen > reward_rejected
                loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                accuracy = (reward_chosen > reward_rejected).float().mean().item()
                epoch_accuracy += accuracy
                num_batches += 1
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy:.2%}",
                    }
                )

            # End of epoch summary
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches

            print(f"\nEpoch {epoch + 1} complete:")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Average accuracy: {avg_accuracy:.2%}")

            # Save checkpoint
            self._save_checkpoint(epoch + 1, avg_loss, avg_accuracy)

        print("\n" + "=" * 60)
        print("Reward Model Training Complete!")
        print("=" * 60)

    def _save_checkpoint(self, epoch: int, loss: float, accuracy: float):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"epoch_{epoch}"
        print(f"\n💾 Saving checkpoint: epoch_{epoch}")

        self.model.save(checkpoint_path)

        # Also save training metadata
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "config": self.config.__dict__,
            },
            checkpoint_path / "training_state.pt",
        )

    def evaluate(self, split: str = "test") -> dict:
        """Evaluate reward model on a dataset split."""
        self.model.eval()

        try:
            eval_loader = self._create_dataloader(split)
        except ValueError:
            print(f"No {split} split available, using train split")
            eval_loader = self._create_dataloader("train")

        total_correct = 0
        total_samples = 0
        total_margin = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.device)

                reward_chosen = self.model(chosen_ids, chosen_mask)
                reward_rejected = self.model(rejected_ids, rejected_mask)

                # Count correct predictions
                correct = (reward_chosen > reward_rejected).sum().item()
                total_correct += correct
                total_samples += len(reward_chosen)

                # Average margin between chosen and rejected
                margin = (reward_chosen - reward_rejected).mean().item()
                total_margin += margin * len(reward_chosen)

        accuracy = total_correct / total_samples
        avg_margin = total_margin / total_samples

        print(f"\nEvaluation Results ({split}):")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Avg margin: {avg_margin:.4f}")

        return {"accuracy": accuracy, "margin": avg_margin}

    def score(self, text: str) -> float:
        """Score a single text (prompt + response)."""
        self.model.eval()

        encoding = self.tokenizer_wrapper(
            text,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            reward = self.model(input_ids, attention_mask)

        return reward.item()


if __name__ == "__main__":
    print("Reward Model Training")
    print("=" * 40)
    print()
    print("Usage:")
    print("  from training.rl.reward_model import RewardTrainer, RewardModelConfig")
    print()
    print("  config = RewardModelConfig(")
    print('      model_name="gpt2",')
    print('      dataset_name="Anthropic/hh-rlhf",')
    print("  )")
    print("  trainer = RewardTrainer(config)")
    print("  trainer.train()")
    print()
    print("The reward model learns to score responses based on human preferences.")
    print("Higher scores = better responses according to the preference data.")

