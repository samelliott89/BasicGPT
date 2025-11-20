# Cosine decay learning rate schedule

import math

from dataclasses import dataclass

@dataclass
class LearningRateConfig:
    """
    Configuration class for learning rate settings.
    """
    warmup_steps: int = 2000
    total_steps: int = 150000
    max_lr: float = 3e-4
    min_lr: float = 3e-5

def get_lr(step: int, config: LearningRateConfig) -> float:
    """
    Warmup + Cosine decay schedule.
    
    Args:
        step: Current training step
        config: LearningRateConfig object
        
    Returns:
        Learning rate for the given step
    """
    if step < config.warmup_steps:
        # Linear warmup
        return config.max_lr * step / config.warmup_steps
    
    # Cosine decay after warmup
    progress = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
    # Clamp progress to [0, 1] to avoid issues if step > total_steps
    progress = max(0.0, min(1.0, progress))

    return config.min_lr + (config.max_lr - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))