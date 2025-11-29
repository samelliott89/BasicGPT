"""
BasicGPT Model Module

Contains:
- GPT: Core GPT architecture
- GPTConfig: Model configuration
- load_model: Unified loader for any model type
- BaseLanguageModel, BaseTokenizer: Abstract interfaces
"""

from model.base import BaseLanguageModel, BaseTokenizer, ModelOutput
from model.config import GPTConfig
from model.gpt import GPT
from model.loader import load_model, load_tokenizer
from model.wrappers import (
    BasicGPTModel,
    BasicGPTTokenizer,
    HuggingFaceModel,
    HuggingFaceTokenizer,
)

__all__ = [
    # Core
    "GPT",
    "GPTConfig",
    # Unified loader
    "load_model",
    "load_tokenizer",
    # Base classes
    "BaseLanguageModel",
    "BaseTokenizer",
    "ModelOutput",
    # Wrappers
    "HuggingFaceModel",
    "HuggingFaceTokenizer",
    "BasicGPTModel",
    "BasicGPTTokenizer",
]
