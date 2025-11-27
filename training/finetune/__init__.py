"""
Supervised Fine-Tuning (SFT) Module

Fine-tune a pretrained language model on instruction-following data.
Supports both:
- HuggingFace models (GPT-2, etc.)
- BasicGPT custom models
"""

from training.finetune.sft_trainer import SFTTrainer
from training.finetune.config import SFTConfig

__all__ = ["SFTTrainer", "SFTConfig"]

