"""
Configuration for Reinforcement Learning training.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DPOConfig:
    """
    Configuration for Direct Preference Optimization (DPO).

    DPO directly optimizes the policy without needing a separate reward model.
    It uses pairs of (chosen, rejected) responses to learn preferences.

    Reference: https://arxiv.org/abs/2305.18290
    """

    # Model configuration
    model_name: str = "gpt2"  # Start with GPT-2 or path to SFT model
    model_type: Literal["huggingface", "basicgpt"] = "huggingface"

    # Reference model (for KL penalty)
    ref_model_name: str | None = None  # None = use same as model_name

    # Dataset configuration
    # Popular preference datasets:
    # - "Anthropic/hh-rlhf" (human helpfulness/harmlessness)
    # - "stanfordnlp/SHP" (reddit preferences)
    # - "argilla/ultrafeedback-binarized-preferences"
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train"
    max_samples: int | None = None

    # Field names in preference dataset
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient (higher = more conservative)

    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 5e-7  # Lower than SFT
    num_epochs: int = 1  # DPO often needs fewer epochs
    max_length: int = 512
    max_prompt_length: int = 256

    # Optimization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Precision
    use_fp16: bool = True
    use_bf16: bool = False

    # Checkpointing
    output_dir: str = "./checkpoints/dpo"
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100

    # Logging
    wandb_project: str = "basicgpt-dpo"
    wandb_run_name: str | None = None

    def __post_init__(self):
        if self.ref_model_name is None:
            self.ref_model_name = self.model_name


@dataclass
class PPOConfig:
    """
    Configuration for Proximal Policy Optimization (PPO).

    PPO is the classic RLHF algorithm that uses a reward model.
    More complex than DPO but can be more powerful.
    """

    # Model configuration
    model_name: str = "gpt2"
    model_type: Literal["huggingface", "basicgpt"] = "huggingface"

    # Reward model
    reward_model_name: str = ""  # Path to trained reward model

    # Dataset (prompts only)
    dataset_name: str = "Anthropic/hh-rlhf"
    prompt_field: str = "prompt"

    # PPO hyperparameters
    ppo_epochs: int = 4  # PPO epochs per batch
    kl_penalty: float = 0.1
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95

    # Training
    batch_size: int = 4
    mini_batch_size: int = 1
    learning_rate: float = 1e-6
    num_epochs: int = 1
    max_length: int = 512

    # Generation during training
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Checkpointing
    output_dir: str = "./checkpoints/ppo"
    save_steps: int = 100
    logging_steps: int = 10


@dataclass
class RewardModelConfig:
    """
    Configuration for training a reward model.

    The reward model learns to score responses based on human preferences.
    Used by PPO for computing rewards during training.
    """

    # Base model to finetune
    model_name: str = "gpt2"

    # Dataset with preference pairs
    # Popular options:
    # - "Anthropic/hh-rlhf" (human helpfulness/harmlessness preferences)
    # - "stanfordnlp/SHP" (reddit preferences)
    # - "argilla/ultrafeedback-binarized-preferences" (GPT-4 judged)
    dataset_name: str = "Anthropic/hh-rlhf"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    max_samples: int | None = None  # Limit samples (useful for testing)

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 1
    max_length: int = 512
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "./checkpoints/reward_model"
    logging_steps: int = 10
