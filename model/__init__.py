"""
BasicGPT Model Module

Usage:
    from model import load_model, ModelType

    # Load HuggingFace model
    model, tokenizer = load_model("gpt2", ModelType.HUGGINGFACE)

    # Load your trained model
    model, tokenizer = load_model("./checkpoints/best", ModelType.BASICGPT)

    # Auto-detect
    model, tokenizer = load_model("gpt2")
"""

from model.config import GPTConfig
from model.gpt import GPT
from model.loader import LanguageModel, ModelType, Tokenizer, load_model
from model.types import ModelOutput

__all__ = [
    # Core
    "GPT",
    "GPTConfig",
    # Unified loader
    "load_model",
    "ModelType",
    "LanguageModel",
    "Tokenizer",
    # Output type
    "ModelOutput",
]
