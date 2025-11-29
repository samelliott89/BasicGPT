"""
Model loader - unified interface for loading any model type.

Usage:
    from model.loader import load_model, load_tokenizer

    # Load GPT-2 from HuggingFace
    model, tokenizer = load_model("gpt2")

    # Load your trained BasicGPT
    model, tokenizer = load_model("./checkpoints/best/checkpoint.pt", model_type="basicgpt")

    # Auto-detect based on path
    model, tokenizer = load_model("gpt2")  # HuggingFace (no .pt extension)
    model, tokenizer = load_model("./checkpoints/epoch_3")  # BasicGPT (local path)
"""

from pathlib import Path
from typing import Literal

import torch

from model.base import BaseLanguageModel, BaseTokenizer
from model.wrappers import (
    BasicGPTModel,
    BasicGPTTokenizer,
    HuggingFaceModel,
    HuggingFaceTokenizer,
)
from utils.device import get_best_device

ModelType = Literal["huggingface", "basicgpt", "auto"]


def load_model(
    path_or_name: str,
    model_type: ModelType = "auto",
    device: torch.device = None,
) -> tuple[BaseLanguageModel, BaseTokenizer]:
    """
    Load a model and tokenizer.

    Args:
        path_or_name: Model path or HuggingFace model name
            - "gpt2", "gpt2-medium", etc. for HuggingFace
            - "./checkpoints/..." for BasicGPT
        model_type: "huggingface", "basicgpt", or "auto" (default)
        device: Device to load on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)

    Examples:
        # HuggingFace GPT-2
        model, tokenizer = load_model("gpt2")
        model, tokenizer = load_model("gpt2-medium")

        # Your trained model
        model, tokenizer = load_model("./checkpoints/best/checkpoint.pt")

        # Explicit type
        model, tokenizer = load_model("gpt2", model_type="huggingface")
    """
    if device is None:
        device = get_best_device()

    # Auto-detect model type
    if model_type == "auto":
        model_type = _detect_model_type(path_or_name)

    print(f"Loading model: {path_or_name}")
    print(f"  Type: {model_type}")
    print(f"  Device: {device}")

    if model_type == "huggingface":
        model = HuggingFaceModel.load(path_or_name, device=device)
        tokenizer = HuggingFaceTokenizer.load(path_or_name)
        print(f"  Vocab size: {tokenizer.vocab_size:,}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    elif model_type == "basicgpt":
        model = BasicGPTModel.load(path_or_name, device=device)
        tokenizer = BasicGPTTokenizer.load()
        print(f"  Vocab size: {model.config.vocab_size:,}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("✓ Model loaded successfully")
    return model, tokenizer


def _detect_model_type(path_or_name: str) -> ModelType:
    """
    Auto-detect whether path is HuggingFace or BasicGPT.

    Rules:
    - If path exists locally and contains checkpoint.pt → BasicGPT
    - If path ends with .pt → BasicGPT
    - Otherwise → HuggingFace
    """
    path = Path(path_or_name)

    # Check if it's a local path with checkpoint
    if path.exists():
        if path.is_dir() and (path / "checkpoint.pt").exists():
            return "basicgpt"
        if path.suffix == ".pt":
            return "basicgpt"
        # Local folder might be a saved HF model
        if (path / "config.json").exists():
            return "huggingface"
        # Default to BasicGPT for local folders with checkpoint
        if path.is_dir():
            return "basicgpt"

    # Path doesn't exist locally - assume HuggingFace model name
    return "huggingface"


def load_tokenizer(
    path_or_name: str,
    tokenizer_type: ModelType = "auto",
) -> BaseTokenizer:
    """
    Load just a tokenizer.

    Args:
        path_or_name: Tokenizer path or HuggingFace name
        tokenizer_type: "huggingface", "basicgpt", or "auto"

    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "auto":
        tokenizer_type = _detect_model_type(path_or_name)

    if tokenizer_type == "huggingface":
        return HuggingFaceTokenizer.load(path_or_name)
    elif tokenizer_type == "basicgpt":
        return BasicGPTTokenizer.load()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
