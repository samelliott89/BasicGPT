"""
Utility functions for BasicGPT.
"""

from utils.checkpoints import correct_vocab_size_from_checkpoint
from utils.device import get_best_device, get_device_info

__all__ = [
    "get_best_device",
    "get_device_info",
    "correct_vocab_size_from_checkpoint",
]
