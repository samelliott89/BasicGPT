#!/usr/bin/env python3
"""
Run Code Generation Evaluations

Evaluates a code generation model on:
1. Execution-based tests (does it run?)
2. HumanEval benchmark (standard)
3. DS-1000 PyTorch subset (standard)
4. Held-out test set (custom)

Usage:
    # Run all evals
    python run_codegen_eval.py ./checkpoints/codegen_sft/best

    # Quick test (limited samples)
    python run_codegen_eval.py ./checkpoints/codegen_sft/best --quick

    # Execution-based only
    python run_codegen_eval.py ./checkpoints/codegen_sft/best --execution-only

    # Skip external benchmarks (no dataset download needed)
    python run_codegen_eval.py ./checkpoints/codegen_sft/best --skip-benchmarks
"""

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from evals.codegen.runner import CodeGenEvalRunner
from evals.codegen.held_out import split_challenges


def main():
    parser = argparse.ArgumentParser(description="Code Generation Evaluation")
    
    parser.add_argument(
        "model_path",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--challenges",
        default="./data/pycode/challenges.jsonl",
        help="Path to challenges JSONL",
    )
    parser.add_argument(
        "--test-set",
        default="./data/pycode/test_challenges.jsonl",
        help="Path to held-out test set",
    )
    parser.add_argument(
        "--output-dir",
        default="./evals/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 10 samples per eval",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per evaluation",
    )
    parser.add_argument(
        "--execution-only",
        action="store_true",
        help="Only run execution-based eval",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip HumanEval and DS-1000 (no downloads needed)",
    )
    parser.add_argument(
        "--skip-held-out",
        action="store_true",
        help="Skip held-out test set",
    )
    parser.add_argument(
        "--create-test-split",
        action="store_true",
        help="Create train/test split from challenges",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)",
    )
    
    args = parser.parse_args()
    
    # Create test split if requested
    if args.create_test_split:
        split_challenges(
            input_path=args.challenges,
            train_path="./data/pycode/train_challenges.jsonl",
            test_path=args.test_set,
            test_ratio=args.test_ratio,
        )
        print()
    
    # Set max samples
    max_samples = args.max_samples
    if args.quick:
        max_samples = 10
        print("🚀 Quick mode: 10 samples per evaluation\n")
    
    # Create runner
    runner = CodeGenEvalRunner(
        model_path=args.model_path,
        challenges_path=args.challenges,
        test_path=args.test_set,
        output_dir=args.output_dir,
    )
    
    # Run evaluations
    if args.execution_only:
        results = runner.run_execution(max_samples)
        print("\n" + str(results))
    else:
        runner.run_all(
            skip_humaneval=args.skip_benchmarks,
            skip_ds1000=args.skip_benchmarks,
            skip_held_out=args.skip_held_out,
            max_samples=max_samples,
        )


if __name__ == "__main__":
    main()

