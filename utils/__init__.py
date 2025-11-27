"""
Utility functions for BasicGPT.
"""

from utils.device import get_best_device, get_device_info
from utils.checkpoints import correct_vocab_size_from_checkpoint

__all__ = [
    "get_best_device", 
    "get_device_info",
    "correct_vocab_size_from_checkpoint",
]
