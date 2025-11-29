#!/usr/bin/env python3
"""
Compare base model vs fine-tuned model.

Usage:
    # Compare base GPT-2 vs SFT checkpoint
    python compare_models.py --sft ./checkpoints/sft/epoch_1

    # Custom prompts
    python compare_models.py --sft ./checkpoints/sft/epoch_1 --prompt "Write a poem"

    # Interactive mode
    python compare_models.py --sft ./checkpoints/sft/epoch_1 --interactive
"""

import argparse
import os

# Suppress all HuggingFace warnings before importing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402

from model import load_model  # noqa: E402


def format_instruction(prompt: str) -> str:
    """Format a prompt in instruction template."""
    return f"""### Instruction:
{prompt}

### Response:"""


def generate(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )

    full_text = tokenizer.decode(output_ids[0].tolist())
    # Return only the generated part
    return full_text[len(prompt) :].strip()


def compare(base_model, base_tok, sft_model, sft_tok, prompt: str, max_tokens: int = 100):
    """Compare outputs from both models."""
    print(f"\n{'=' * 60}")
    print(f"📝 PROMPT: {prompt}")
    print("=" * 60)

    # Base model gets raw prompt
    print("\n🔵 BASE GPT-2:")
    print("-" * 40)
    base_output = generate(base_model, base_tok, prompt, max_tokens)
    print(base_output)

    # SFT model gets instruction-formatted prompt
    formatted_prompt = format_instruction(prompt)
    print("\n🟢 FINE-TUNED:")
    print("-" * 40)
    sft_output = generate(sft_model, sft_tok, formatted_prompt, max_tokens)
    # Extract just the response (after "### Response:")
    if "### Response:" in sft_output:
        sft_output = sft_output.split("### Response:")[-1].strip()
    print(sft_output)

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument("--base", default="gpt2", help="Base model (default: gpt2)")
    parser.add_argument("--sft", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--prompt", default=None, help="Single prompt to test")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Load base model
    print("Loading base model...")
    base_model, base_tok = load_model(args.base, "huggingface")
    base_model.eval()
    print("✓ Base model loaded")

    # Load SFT model
    print(f"\nLoading fine-tuned model from {args.sft}...")
    sft_model, sft_tok = load_model(args.sft, "huggingface")
    sft_model.eval()
    print("✓ Fine-tuned model loaded")

    # Default test prompts (instruction-style since that's what Alpaca trains on)
    default_prompts = [
        "### Instruction:\nWrite a short poem about coding\n\n### Response:",
        "### Instruction:\nExplain what machine learning is\n\n### Response:",
        "### Instruction:\nGive me 3 tips for learning Python\n\n### Response:",
        "The meaning of life is",
        "Once upon a time",
    ]

    if args.prompt:
        compare(base_model, base_tok, sft_model, sft_tok, args.prompt, args.max_tokens)
    elif args.interactive:
        print("\n" + "=" * 60)
        print("Interactive Comparison Mode (Ctrl+C to exit)")
        print("=" * 60)
        while True:
            try:
                prompt = input("\n📝 Enter prompt: ").strip()
                if not prompt:
                    continue
                compare(base_model, base_tok, sft_model, sft_tok, prompt, args.max_tokens)
            except KeyboardInterrupt:
                print("\n\nBye!")
                break
    else:
        print("\n" + "=" * 60)
        print("COMPARISON: Base GPT-2 vs Fine-tuned")
        print("=" * 60)

        for prompt in default_prompts:
            compare(base_model, base_tok, sft_model, sft_tok, prompt, args.max_tokens)


if __name__ == "__main__":
    main()
