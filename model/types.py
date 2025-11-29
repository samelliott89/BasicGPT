"""
Model types and dataclasses.
"""

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """Standardized output from model forward pass."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
