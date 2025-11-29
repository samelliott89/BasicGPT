"""
Configuration for Supervised Fine-Tuning (SFT).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SFTConfig:
    """
    Configuration for supervised fine-tuning.

    Attributes:
        model_name: HuggingFace model name or path to BasicGPT checkpoint
        model_type: "huggingface" or "basicgpt"
        dataset_name: HuggingFace dataset name for instruction data

        # Training hyperparameters
        batch_size: Training batch size
        learning_rate: Peak learning rate
        num_epochs: Number of training epochs
        max_length: Maximum sequence length

        # Optimization
        weight_decay: L2 regularization
        warmup_ratio: Fraction of steps for warmup
        gradient_accumulation_steps: Steps to accumulate before update

        # Checkpointing
        output_dir: Directory to save checkpoints
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
    """

    # Model configuration
    model_name: str = "gpt2"  # Default to GPT-2
    model_type: Literal["huggingface", "basicgpt"] = "huggingface"

    # Dataset configuration
    dataset_name: str = "tatsu-lab/alpaca"  # Classic instruction dataset
    dataset_split: str = "train"
    max_samples: int | None = None  # None = use all

    # Input formatting
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"

    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 512

    # Optimization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Precision
    use_fp16: bool = True
    use_bf16: bool = False  # Prefer BF16 on Ampere+ GPUs

    # Checkpointing
    output_dir: str = "./checkpoints/sft"
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 500

    # Logging
    wandb_project: str = "basicgpt-sft"
    wandb_run_name: str | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.use_fp16 and self.use_bf16:
            raise ValueError("Cannot use both FP16 and BF16")


@dataclass
class InstructionTemplate:
    """
    Template for formatting instruction data.

    Example:
        template = InstructionTemplate()
        formatted = template.format(
            instruction="Write a poem about AI",
            input="",
            output="In silicon dreams..."
        )
    """

    # Prompt template (no response)
    prompt_template: str = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

    # Full template (with response for training)
    full_template: str = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    # Special tokens
    eos_token: str = "<|endoftext|>"

    def format_prompt(self, instruction: str, input: str = "") -> str:
        """Format just the prompt (for inference)."""
        return self.prompt_template.format(instruction=instruction, input=input)

    def format_full(self, instruction: str, input: str, output: str) -> str:
        """Format the full example (for training)."""
        return (
            self.full_template.format(instruction=instruction, input=input, output=output)
            + self.eos_token
        )
