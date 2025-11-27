"""
Device detection and management utilities.
"""

import torch


def get_best_device() -> torch.device:
    """
    Select the best available device for training/inference.
    Preference order: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_info(device: torch.device) -> str:
    """
    Return a short human-readable description of the selected device.
    """
    if device.type == "cuda":
        try:
            index = device.index if device.index is not None else 0
            name = torch.cuda.get_device_name(index)
            return f"CUDA ({name})"
        except Exception:
            return "CUDA"
    if device.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"

