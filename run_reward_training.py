#!/usr/bin/env python3
"""
Train a Reward Model

The reward model learns to predict human preferences.
Given (prompt + response), it outputs a scalar score.

Usage:
    # Quick test (10 samples)
    python run_reward_training.py --test

    # Full training
    python run_reward_training.py --epochs 3

    # Custom model
    python run_reward_training.py --model gpt2-medium --epochs 2
"""

import argparse
import os

# Suppress warnings for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from training.rl.config import RewardModelConfig
from training.rl.reward_model import RewardTrainer


def main():
    parser = argparse.ArgumentParser(description="Train a reward model")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Base model name (default: gpt2)",
    )
    parser.add_argument(
        "--dataset",
        default="Anthropic/hh-rlhf",
        help="Preference dataset (default: Anthropic/hh-rlhf)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (for testing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run with 10 samples",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/reward_model",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Quick test mode
    if args.test:
        args.max_samples = 10
        args.epochs = 1
        print("🧪 Test mode: training on 10 samples only")

    # Create config
    config = RewardModelConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )

    # Train
    trainer = RewardTrainer(config)
    trainer.train()

    # Quick evaluation
    print("\n" + "=" * 60)
    print("Testing the trained reward model")
    print("=" * 60)

    # Test on some examples
    test_cases = [
        # Good response
        "Human: What is 2+2?\nAssistant: 2+2 equals 4.",
        # Bad response
        "Human: What is 2+2?\nAssistant: I don't know, maybe try Google.",
        # Helpful response
        "Human: How do I learn Python?\nAssistant: Start with the official Python tutorial at python.org. It covers basics like variables, loops, and functions. Then try small projects like a calculator or to-do list.",
        # Unhelpful response
        "Human: How do I learn Python?\nAssistant: Just figure it out yourself.",
    ]

    print("\nScoring test responses:")
    print("-" * 40)

    for text in test_cases:
        score = trainer.score(text)
        # Truncate for display
        display = text[:60] + "..." if len(text) > 60 else text
        display = display.replace("\n", " ")
        print(f"  Score: {score:+.4f} | {display}")

    print("\n✓ Higher scores = better responses")
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

