"""
Device detection and management utilities.
"""

from enum import Enum

import torch


class Device(str, Enum):
    CUDA = torch.device("cuda")
    MPS = torch.device("mps")
    CPU = torch.device("cpu")

class DeviceName(str, Enum):
    CUDA = "CUDA"
    MPS = "Apple Metal (MPS)"
    CPU = "CPU"

def get_best_device() -> torch.device:
    """
    Select the best available device for training/inference.
    Preference order: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return Device.CUDA

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return Device.MPS

    return Device.CPU


def get_device_info(device: Device) -> str:
    """
    Return a short human-readable description of the selected device.
    """
    if device == Device.CUDA:
        try:
            index = device.index if device.index is not None else 0
            name = torch.cuda.get_device_name(index)
            return f"{DeviceName.CUDA} ({name})"
        except Exception:
            return DeviceName.CUDA
    if device == Device.MPS:
        return DeviceName.MPS
    if device == Device.CPU:
        return DeviceName.CPU
    else:
        raise ValueError(f"Invalid device: {device}")

