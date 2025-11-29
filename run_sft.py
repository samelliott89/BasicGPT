#!/usr/bin/env python3
"""
Run Supervised Fine-Tuning (SFT) on GPT-2.

This script fine-tunes GPT-2 to follow instructions using the Alpaca dataset.

Usage:
    python run_sft.py                    # Run with defaults
    python run_sft.py --epochs 3         # Custom epochs
    python run_sft.py --model gpt2-medium  # Larger model
"""

import argparse

from training.finetune import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model name (gpt2, gpt2-medium, gpt2-large)",
    )

    # Dataset args
    parser.add_argument(
        "--dataset", type=str, default="tatsu-lab/alpaca", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to use (None = all)"
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")

    # Output args
    parser.add_argument("--output-dir", type=str, default="./checkpoints/sft")

    args = parser.parse_args()

    # Create config
    config = SFTConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
    )

    # Print config
    print("\n" + "=" * 60)
    print("Supervised Fine-Tuning Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Max samples: {config.max_samples or 'all'}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max length: {config.max_length}")
    print(f"Output dir: {config.output_dir}")
    print()

    # Create trainer and train
    trainer = SFTTrainer(config)
    trainer.train()

    # Test generation
    print("\n" + "=" * 60)
    print("Testing generation after training")
    print("=" * 60)

    prompts = [
        "Explain machine learning in simple terms.",
        "Write a short poem about the ocean.",
        "What are three tips for being productive?",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = trainer.generate(prompt, max_new_tokens=100)
        print(f"Response: {response}")
        print("-" * 40)


if __name__ == "__main__":
    main()
