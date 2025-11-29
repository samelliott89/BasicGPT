#!/usr/bin/env python3
"""
Run Direct Preference Optimization (DPO) on a fine-tuned model.

DPO trains the model to prefer "chosen" responses over "rejected" responses.
This is typically run AFTER supervised fine-tuning (SFT).

Usage:
    # After running SFT:
    python run_dpo.py --model ./checkpoints/sft/epoch_3

    # Or start from base GPT-2:
    python run_dpo.py --model gpt2
"""

import argparse

from training.rl import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization")

    # Model args
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Path to SFT model or HuggingFace model name"
    )
    parser.add_argument(
        "--ref-model", type=str, default=None, help="Reference model (default: same as --model)"
    )

    # Dataset args
    parser.add_argument(
        "--dataset", type=str, default="Anthropic/hh-rlhf", help="Preference dataset name"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use")

    # DPO hyperparameters
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty coefficient (higher = more conservative)",
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)

    # Output args
    parser.add_argument("--output-dir", type=str, default="./checkpoints/dpo")

    args = parser.parse_args()

    # Create config
    config = DPOConfig(
        model_name=args.model,
        ref_model_name=args.ref_model or args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        beta=args.beta,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
    )

    # Print config
    print("\n" + "=" * 60)
    print("Direct Preference Optimization (DPO) Configuration")
    print("=" * 60)
    print(f"Policy model: {config.model_name}")
    print(f"Reference model: {config.ref_model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Beta (KL coefficient): {config.beta}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Output dir: {config.output_dir}")
    print()

    # Create trainer and train
    trainer = DPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
