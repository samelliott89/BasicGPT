#!/usr/bin/env python3
"""
Run Supervised Fine-Tuning with GPT-2.

Usage:
    python run_sft.py                           # Use defaults (GPT-2, Alpaca)
    python run_sft.py --model gpt2-medium       # Larger model
    python run_sft.py --max-samples 1000        # Quick test run
"""

import argparse
import os

# Suppress HuggingFace warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from training.finetune.config import SFTConfig  # noqa: E402
from training.finetune.sft_trainer import SFTTrainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run SFT with GPT-2")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca", help="Dataset name")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output-dir", default="./checkpoints/sft", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    args = parser.parse_args()

    # Create config
    config = SFTConfig(
        model_name=args.model,
        model_type="huggingface",
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        output_dir=args.output_dir,
        use_fp16=args.fp16,
    )

    print("\n" + "=" * 60)
    print("SFT Configuration")
    print("=" * 60)
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Max samples: {config.max_samples or 'all'}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max length: {config.max_length}")
    print(f"  Output: {config.output_dir}")
    print(f"  FP16: {config.use_fp16}")
    print()

    # Create trainer and run
    trainer = SFTTrainer(config)
    trainer.train()

    # Test generation
    print("\n" + "=" * 60)
    print("Testing generation...")
    print("=" * 60)

    test_prompts = [
        "Write a short poem about coding",
        "Explain what machine learning is in simple terms",
        "Give me 3 tips for learning Python",
    ]

    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        response = trainer.generate(prompt, max_new_tokens=100)
        print(f"🤖 Response: {response}")


if __name__ == "__main__":
    main()
