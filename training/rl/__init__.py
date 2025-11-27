"""
Reinforcement Learning (RL) Module

Train language models using reinforcement learning techniques:
- DPO: Direct Preference Optimization (simpler, no reward model needed)
- PPO: Proximal Policy Optimization (classic RLHF)
- Reward Model: Train a reward model for RLHF
"""

from training.rl.dpo_trainer import DPOTrainer
from training.rl.config import DPOConfig

__all__ = ["DPOTrainer", "DPOConfig"]

