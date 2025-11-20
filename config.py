"""
Configuration classes for GPT model training, generation, and evaluation.

This module provides centralized configuration management for all components
of the GPT training pipeline.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from learning_rate import LearningRateConfig

@dataclass
class GPTConfig:
    """
    Configuration class for the GPT model architecture.
    
    This holds all the hyperparameters (settings) for the model structure.
    Using a dataclass makes it easy to pass around configuration.
    """
    vocab_size: int = 100277  # Vocabulary size from tiktoken cl100k_base
    d_model: int = 256  # Dimension of the model (embedding size)
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 16  # Number of transformer layers
    max_length: int = 1024  # Maximum sequence length (context window)
    dropout: float = 0.1  # Dropout rate for regularization


@dataclass
class DataConfig:
    """
    Configuration class for data loading and preprocessing.
    
    Controls how data is loaded from the dataset and prepared for training.
    """
    max_samples: Optional[int] = 5000000  # Maximum number of samples to use (None = all)
    max_length: int = 1024  # Maximum sequence length for tokenization
    text_field: str = "synthetic_answer"  # Which field to use from dataset
    include_reasoning: bool = False  # Whether to include reasoning steps in training data
    filter_english_only: bool = True  # Filter to only English samples (uses dataset's 'language' field)
    streaming: bool = True  # Use streaming mode for large datasets
    timeout: int = 300  # Timeout in seconds for dataset download
    num_retries: int = 3  # Number of retry attempts on connection failure
    num_workers: int = 0  # Number of parallel workers for data loading (0 for IterableDataset)
    num_dataset_workers: int = 4  # Number of parallel workers for dataset loading


@dataclass
class TrainingConfig:
    """
    Configuration class for model training.
    
    Controls training hyperparameters and optimization settings.
    """
    # Model architecture (can override GPTConfig defaults)
    gpt_config: GPTConfig = field(default_factory=GPTConfig)
    
    # Training hyperparameters
    batch_size: int = 32  # Batch size for training (optimized for 92GB VRAM with max_length=1024)
    lr_config: LearningRateConfig = field(default_factory=LearningRateConfig)
    epochs: int = 1  # Number of training epochs
    weight_decay: float = 0.01  # L2 regularization (for AdamW)
    beta1: float = 0.9  # Adam beta1 parameter
    beta2: float = 0.95  # Adam beta2 parameter
    max_grad_norm: float = 1.0  # Gradient clipping max norm
    gradient_accumulation_steps: int = 4  # Number of steps to accumulate gradients (1 = no accumulation, effective_batch_size = batch_size * gradient_accumulation_steps = 32 * 4 = 128)
    
    # Data configuration
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Training settings
    save_dir: str = "./checkpoints"  # Directory to save model checkpoints
    save_every_n_batches: Optional[int] = None  # Save checkpoint every N batches (None = only at end of epoch)
    print_every_n_batches: int = 50  # Print progress every N batches
    use_mixed_precision: bool = True  # Use FP16/BF16 mixed precision training
    use_gradient_checkpointing: bool = False  # Use gradient checkpointing to save memory
    checkpoint_interval: int = 10000  # Save checkpoint every N batches


@dataclass
class GenerationConfig:
    """
    Configuration class for text generation.
    
    Controls sampling strategies and generation parameters.
    """
    max_new_tokens: int = 200  # Maximum number of new tokens to generate
    temperature: float = 0.8  # Temperature for sampling (lower = more focused, higher = more random)
    top_k: int = 50  # Top-k sampling: only consider top k tokens (0 = disabled)
    top_p: float = 0.9  # Top-p (nucleus) sampling: cumulative probability threshold (0.0 = disabled)
    repetition_penalty: float = 1.2  # Penalty for repeating tokens (>1.0 reduces repetition, higher = less repetition)


@dataclass
class EvaluationConfig:
    """
    Configuration class for model evaluation.
    
    Controls evaluation metrics and test settings.
    Uses GenerationConfig for generation parameters to ensure consistency.
    """
    # Evaluation data settings
    eval_samples: int = 1000  # Number of samples to use for evaluation
    batch_size: int = 8  # Batch size for evaluation
    max_length: int = 1024  # Maximum sequence length for evaluation
    
    # Generation config for quality tests (shared with generation script)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Accuracy evaluation settings
    top_k_accuracy: int = 5  # Calculate top-k accuracy (for accuracy metrics, not generation)
    
    # Test prompts for generation quality evaluation
    test_prompts: list[str] = field(default_factory=lambda: [
        "The meaning of life is",
        "In the future, artificial intelligence will",
        "Once upon a time, there was",
        "The key to success is",
        "Science has shown that"
    ])


@dataclass
class OptimizerConfig:
    """
    Configuration class for optimizer settings.
    
    Can be used to customize optimizer behavior.
    """
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8  # Epsilon for numerical stability


@dataclass
class TokenizerConfig:
    """
    Configuration class for tokenizer settings.
    """
    encoding_name: str = "cl100k_base"  # Tiktoken encoding name
    # vocab_size is determined by the encoding, not set here
