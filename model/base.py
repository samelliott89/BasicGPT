"""
Base model interface for language models.

This abstraction allows swapping between:
- HuggingFace models (GPT-2, etc.)
- BasicGPT custom models

Usage:
    # Load GPT-2
    model = load_model("gpt2", model_type="huggingface")

    # Load your custom model
    model = load_model("./checkpoints/best/checkpoint.pt", model_type="basicgpt")

    # Both have the same interface:
    output = model.generate(input_ids, max_new_tokens=100)
    loss = model.forward(input_ids, labels=labels)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from config import GenerationConfig

# Default generation config instance for default parameter values
_default_gen_config = GenerationConfig()


@dataclass
class ModelOutput:
    """Standardized output from model forward pass."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class BaseLanguageModel(ABC):
    """
    Abstract base class for language models.

    Both HuggingFace and BasicGPT models implement this interface.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (1 = attend, 0 = ignore)
            labels: Target token IDs for loss computation [batch, seq_len]

        Returns:
            ModelOutput with logits and optional loss
        """
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = _default_gen_config.max_new_tokens,
        temperature: float = _default_gen_config.temperature,
        top_k: int = _default_gen_config.top_k,
        top_p: float = _default_gen_config.top_p,
        repetition_penalty: float = _default_gen_config.repetition_penalty,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate new tokens.

        Args:
            input_ids: Starting sequence [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (0.0 = disabled)
            repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)

        Returns:
            Generated sequence including input [batch, seq_len + new_tokens]
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to path."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: torch.device = None) -> "BaseLanguageModel":
        """Load model from path."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get model device."""
        pass

    @property
    @abstractmethod
    def config(self):
        """Get model configuration."""
        pass


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Both HuggingFace and tiktoken tokenizers implement this interface.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        pass

    def __call__(
        self,
        text: str | list[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = None,
    ) -> dict:
        """
        Tokenize text with padding and truncation.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            return_tensors: "pt" for PyTorch tensors, None for lists

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Handle single string
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        all_input_ids = []
        all_attention_masks = []

        for t in texts:
            # Encode
            ids = self.encode(t)

            # Truncate if needed
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]

            # Create attention mask (1 for real tokens)
            mask = [1] * len(ids)

            # Pad if needed
            if padding:
                pad_length = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_length
                mask = mask + [0] * pad_length

            all_input_ids.append(ids)
            all_attention_masks.append(mask)

        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"])
            result["attention_mask"] = torch.tensor(result["attention_mask"])

        # Squeeze if single input
        if isinstance(text, str) and return_tensors == "pt":
            result["input_ids"] = result["input_ids"]
            result["attention_mask"] = result["attention_mask"]

        return result
