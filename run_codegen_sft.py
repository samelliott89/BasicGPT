#!/usr/bin/env python3
"""
Train a code generation model with SFT.

Stage 1 of the code generation pipeline:
    DeepSeek-Coder → SFT → (RL)

Usage:
    # Quick test
    python run_codegen_sft.py --test

    # Full training
    python run_codegen_sft.py --epochs 3

    # With custom dataset
    python run_codegen_sft.py --dataset ./data/pycode/challenges.jsonl
"""

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from training.codegen.config import CodeGenSFTConfig
from training.codegen.sft_trainer import CodeGenSFTTrainer


def main():
    parser = argparse.ArgumentParser(description="Code Generation SFT")
    
    parser.add_argument(
        "--model",
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="Base model (default: deepseek-coder-1.3b-base)",
    )
    parser.add_argument(
        "--dataset",
        default="./data/pycode/challenges.jsonl",
        help="Path to challenges JSONL file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test with example challenges only",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/codegen_sft",
        help="Output directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 training",
    )
    
    args = parser.parse_args()
    
    if args.test:
        args.max_samples = 3
        args.epochs = 1
        args.batch_size = 1  # Small batch for test mode
        print("🧪 Test mode: training on 3 example challenges")
    
    # Adjust gradient accumulation based on batch size
    grad_accum = 1 if args.batch_size <= 2 else 4
    
    config = CodeGenSFTConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        use_fp16=args.fp16,
        gradient_accumulation_steps=grad_accum,
    )
    
    trainer = CodeGenSFTTrainer(config)
    trainer.train()
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing generation")
    print("=" * 60)
    
    test_prompts = [
        "Create a simple 2-layer CNN",
        "Create a residual block with batch normalization",
        "Create a multi-head self-attention layer",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 40)
        code = trainer.generate(prompt, max_new_tokens=300)
        print(code[:500] + "..." if len(code) > 500 else code)


if __name__ == "__main__":
    main()

