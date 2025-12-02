#!/usr/bin/env python3
"""
Train code generation model with RL (execution feedback).

Stage 2 of the pipeline:
    DeepSeek-Coder → SFT → RL (this script)

Rewards come from actual code execution:
    - Parse: +0.1
    - Instantiate: +0.2
    - Forward pass works: +0.3
    - Backward pass works: +0.2
    - Trains stably (no NaN): +0.2
    = Max reward: 1.0

Usage:
    # After running SFT
    python run_codegen_rl.py --model ./checkpoints/codegen_sft/best

    # Quick test
    python run_codegen_rl.py --test --episodes 100

    # Full training
    python run_codegen_rl.py --episodes 10000
"""

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from training.codegen.config import CodeGenRLConfig
from training.codegen.rl_trainer import CodeGenRLTrainer


def main():
    parser = argparse.ArgumentParser(description="Code Generation RL Training")
    
    parser.add_argument(
        "--model",
        default="./checkpoints/codegen_sft/best",
        help="Policy model (SFT checkpoint)",
    )
    parser.add_argument(
        "--ref-model",
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="Reference model for KL penalty",
    )
    parser.add_argument(
        "--challenges",
        default="./data/pycode/challenges.jsonl",
        help="Path to challenges JSONL",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of RL episodes",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.1,
        help="KL penalty coefficient",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Execution timeout in seconds",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip gradient updates (just evaluate)",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/codegen_rl",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.test:
        args.episodes = 100
        print("🧪 Test mode: 100 episodes only")
    
    config = CodeGenRLConfig(
        model_name=args.model,
        ref_model_name=args.ref_model,
        challenges_path=args.challenges,
        num_episodes=args.episodes,
        kl_coef=args.kl_coef,
        learning_rate=args.lr,
        execution_timeout=args.timeout,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
    )
    
    trainer = CodeGenRLTrainer(config)
    trainer.train()
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing final model")
    print("=" * 60)
    
    test_prompts = [
        "Create a simple convolutional block",
        "Create a residual block with skip connection",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 40)
        code = trainer.generate(prompt)
        print(code[:400] + "..." if len(code) > 400 else code)


if __name__ == "__main__":
    main()

