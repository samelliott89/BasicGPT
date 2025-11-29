"""
GPT Model Configuration

Contains only the model architecture configuration.
Training, data, and generation configs are in their respective modules.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration class for the GPT model architecture.

    This holds all the hyperparameters for the model structure.
    """

    vocab_size: int = 100277  # Vocabulary size from tiktoken cl100k_base
    d_model: int = 256  # Dimension of the model (embedding size)
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 16  # Number of transformer layers
    max_length: int = 1024  # Maximum sequence length (context window)
    dropout: float = 0.1  # Dropout rate for regularization

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_length > 0, "max_length must be positive"
