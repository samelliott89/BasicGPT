"""
Configuration for Code Generation Training Pipeline

Stages:
1. SFT on code challenges (prompt → working code)
2. RL with execution feedback (CodeGenEnv)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class CodeGenSFTConfig:
    """
    Configuration for SFT on code generation challenges.
    
    The model learns to generate working PyTorch nn.Module code
    from natural language descriptions.
    """
    
    # Model
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # Dataset - JSONL with {prompt, code, test_code} entries
    dataset_path: str = "./data/pycode/challenges.jsonl"
    max_samples: int | None = None
    
    # Sequence lengths (separate to ensure full sequences)
    # Total max = max_prompt_length + max_response_length
    max_prompt_length: int = 256   # Task descriptions are short
    max_response_length: int = 768  # Code can be long
    
    @property
    def max_length(self) -> int:
        """Total sequence length for training."""
        return self.max_prompt_length + self.max_response_length
    
    # Training
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    
    # Optimization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Precision
    use_fp16: bool = True
    use_bf16: bool = False
    
    # Output
    output_dir: str = "./checkpoints/codegen_sft"
    logging_steps: int = 10
    save_steps: int = 100


@dataclass
class CodeGenRLConfig:
    """
    Configuration for RL training with execution feedback.
    
    Uses CodeGenEnv to get rewards from actual code execution.
    
    Sequence handling:
        - Prompt is tokenized separately (max_prompt_length)
        - Response is generated up to max_new_tokens
        - Total context = prompt + generated (for KL computation)
    """
    
    # Policy model (start from SFT checkpoint)
    model_name: str = "./checkpoints/codegen_sft/best"
    
    # Reference model (frozen, for KL penalty)
    ref_model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # Challenges dataset
    challenges_path: str = "./data/pycode/challenges.jsonl"
    
    # RL hyperparameters
    num_episodes: int = 10000
    kl_coef: float = 0.1  # KL penalty coefficient
    
    # Sequence lengths
    max_prompt_length: int = 256   # Task description (will truncate if longer)
    max_new_tokens: int = 768      # Generated code length (model can stop early)
    
    # Generation settings
    temperature: float = 0.8
    top_p: float = 0.95
    
    # PPO specifics
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training
    batch_size: int = 4
    learning_rate: float = 5e-8  # Extremely small for RL stability
    max_grad_norm: float = 0.1   # Very aggressive clipping
    eval_only: bool = False      # Skip gradient updates (for testing)
    
    # Execution environment
    execution_timeout: float = 10.0
    torch_threads: int = 1
    seed: int = 1337
    
    # Output
    output_dir: str = "./checkpoints/codegen_rl"
    logging_steps: int = 10
    save_steps: int = 100


@dataclass
class CodeChallenge:
    """
    A single code generation challenge.
    
    Example:
        {
            "prompt": "Create a residual block with two 3x3 convolutions",
            "code": "class ResBlock(nn.Module): ...",
            "test_code": "model = ResBlock(64); x = torch.randn(2, 64, 32, 32); assert model(x).shape == x.shape"
        }
    """
    prompt: str
    code: str
    test_code: str = ""
    
    # Metadata
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    category: Literal["cnn", "attention", "transformer", "rnn", "other"] = "cnn"


# Example challenges for testing
EXAMPLE_CHALLENGES = [
    {
        "prompt": "Create a simple 2-layer CNN that takes 64 input channels and outputs 128 channels",
        "code": '''class SimpleCNN(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x''',
        "test_code": "",
        "difficulty": "easy",
        "category": "cnn"
    },
    {
        "prompt": "Create a residual block with skip connection, batch norm, and ReLU",
        "code": '''class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)''',
        "test_code": "",
        "difficulty": "medium",
        "category": "cnn"
    },
    {
        "prompt": "Create a multi-head self-attention layer with 8 heads and 256 dimensions",
        "code": '''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, L, _ = x.shape
        
        q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)''',
        "test_code": "",
        "difficulty": "medium",
        "category": "attention"
    },
]

