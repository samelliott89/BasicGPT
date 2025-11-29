#!/usr/bin/env python3
"""
Run inference with any model.

Usage:
    python run_inference.py "Hello, world"
    python run_inference.py "Write a poem" --model gpt2-medium
    python run_inference.py "Once upon a time" --max-tokens 200
    python run_inference.py  # Interactive mode
"""

import argparse
import os

# Suppress all HuggingFace warnings before importing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt text")
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument("--type", default="auto", choices=["auto", "huggingface", "basicgpt"])
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, args.type)
    model.eval()
    print("✓ Ready!\n")

    def generate(prompt: str) -> str:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        return tokenizer.decode(output_ids[0].tolist())

    # Single prompt or interactive mode
    if args.prompt:
        print(f"Prompt: {args.prompt}\n")
        print(generate(args.prompt))
    else:
        print("Interactive mode (Ctrl+C to exit)")
        print("-" * 40)
        while True:
            try:
                prompt = input("\n📝 You: ").strip()
                if not prompt:
                    continue
                print(f"\n🤖 GPT-2: {generate(prompt)}")
            except KeyboardInterrupt:
                print("\n\nBye!")
                break


if __name__ == "__main__":
    main()
