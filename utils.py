"""
Utility functions for GPT model checkpoint handling and common operations.
"""

import torch
from config import GPTConfig


def correct_vocab_size_from_checkpoint(config: GPTConfig, checkpoint: dict) -> GPTConfig:
    """
    Correct vocab_size in config by checking the actual checkpoint weights.
    
    This is a temporary workaround for checkpoints that were saved with incorrect
    config values. The vocab_size is inferred from the token_embedding weight shape.
    
    Args:
        config: GPTConfig object (may have incorrect vocab_size)
        checkpoint: Dictionary loaded from checkpoint file
        
    Returns:
        GPTConfig with corrected vocab_size (or original if no correction needed)
    """
    if 'model_state_dict' in checkpoint:
        token_embedding_weight = checkpoint['model_state_dict'].get('token_embedding.weight')
        if token_embedding_weight is not None:
            vocab_size_from_checkpoint = token_embedding_weight.shape[0]
            if vocab_size_from_checkpoint != config.vocab_size:
                print(f"  ⚠️  Vocab size mismatch detected!")
                print(f"     Config says: {config.vocab_size}")
                print(f"     Checkpoint weights say: {vocab_size_from_checkpoint}")
                print(f"     Updating config to match checkpoint weights...")
                config.vocab_size = vocab_size_from_checkpoint
    
    return config

