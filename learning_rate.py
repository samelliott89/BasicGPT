# Cosine decay learning rate schedule

import math
from dataclasses import dataclass

# Notes:
    # batch_size = 32
    # gradient_accumulation_steps = 4
    # steps_per_epoch = 126519  # batches per epoch
    # total_epochs = 3

    # Calculate optimizer steps (not batch steps):
        # optimizer_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
        # total_optimizer_steps = optimizer_steps_per_epoch * total_epochs

    # Result:
        # optimizer_steps_per_epoch = 126519 // 4 = 31,629
        # total_optimizer_steps = 31629 * 3 = 94,887

@dataclass
class LearningRateConfig:
    """
    Configuration class for learning rate settings.
    """
    warmup_steps: int = 2000
    total_steps: int = 94887      # (126519 // 4) * 3 [steps_per_epoch]
    max_lr: float = 6e-4          # Adjusted for effective_batch=128
    min_lr: float = 6e-5          # 10% of max_lr

def get_lr(step: int, config: LearningRateConfig) -> float:
    """
    Warmup + Cosine decay schedule.
    
    Args:
        step: Current optimizer step (accounts for gradient accumulation)
        config: LearningRateConfig object
        
    Returns:
        Learning rate for the given step
    """
    if step < config.warmup_steps:
        # Linear warmup: 0 → max_lr
        return config.max_lr * step / config.warmup_steps
    
    # Cosine decay after warmup: max_lr → min_lr
    progress = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
    # Clamp progress to [0, 1] to avoid issues if step > total_steps
    progress = max(0.0, min(1.0, progress))
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr + (config.max_lr - config.min_lr) * cosine_decay