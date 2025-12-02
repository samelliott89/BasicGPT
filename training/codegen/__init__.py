"""
Code Generation Training Module

Train models to generate working PyTorch code from natural language.

Pipeline:
1. SFT: Learn to generate code from descriptions
2. RL: Refine with execution feedback (CodeGenEnv)
"""

from training.codegen.config import (
    CodeChallenge,
    CodeGenRLConfig,
    CodeGenSFTConfig,
    EXAMPLE_CHALLENGES,
)
from training.codegen.sft_trainer import CodeGenSFTTrainer
from training.codegen.rl_trainer import CodeGenRLTrainer

__all__ = [
    "CodeGenSFTConfig",
    "CodeGenRLConfig",
    "CodeChallenge",
    "CodeGenSFTTrainer",
    "CodeGenRLTrainer",
    "EXAMPLE_CHALLENGES",
]

