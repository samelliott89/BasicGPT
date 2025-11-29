import math

from config import TrainingConfig
from learning_rate import LearningRateConfig


def calculate_lr_config(
    steps_per_epoch: int,
    total_epochs: int,
    gradient_accumulation: int,
    batch_size: int,
    base_lr: float = LearningRateConfig.max_lr,
) -> LearningRateConfig:
    """
    Helper to calculate correct LR config.
    """
    # Calculate optimizer steps
    optimizer_steps_per_epoch = steps_per_epoch // gradient_accumulation
    total_optimizer_steps = optimizer_steps_per_epoch * total_epochs

    # Adjust LR for effective batch size
    effective_batch_size = batch_size * gradient_accumulation
    max_lr = base_lr * math.sqrt(effective_batch_size / 32)
    min_lr = max_lr * 0.1

    return LearningRateConfig(
        warmup_steps=LearningRateConfig.warmup_steps,
        total_steps=total_optimizer_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )


# Usage:
lr_config = calculate_lr_config(
    steps_per_epoch=126519,
    total_epochs=TrainingConfig.epochs,
    gradient_accumulation=TrainingConfig.gradient_accumulation_steps,
    batch_size=TrainingConfig.batch_size,
    base_lr=LearningRateConfig.max_lr,
)

print("Calculated LR config:")
print(f"  total_steps: {lr_config.total_steps:,}")
print(f"  max_lr: {lr_config.max_lr:.2e}")
print(f"  min_lr: {lr_config.min_lr:.2e}")
