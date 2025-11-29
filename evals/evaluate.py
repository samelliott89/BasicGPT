"""
Evaluation script for GPT model.

This script provides various evaluation metrics and tests to assess model performance:
- Perplexity (lower is better)
- Loss on validation/test data
- Generation quality tests
- Token prediction accuracy
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import EvaluationConfig
from data.datasets import create_data_loaders, load_validation_data
from data.tokenizer import Tokenizer
from model.config import GPTConfig
from model.gpt import GPT
from utils.device import get_best_device


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load model from checkpoint.

    Supports both:
    - Folder path: /path/to/checkpoint-folder/ (will look for checkpoint.pt inside)
    - File path: /path/to/checkpoint.pt (legacy support)
    """
    checkpoint_path_obj = Path(checkpoint_path)

    # If it's a directory, look for checkpoint.pt inside
    if checkpoint_path_obj.is_dir():
        checkpoint_file = checkpoint_path_obj / "checkpoint.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found in folder: {checkpoint_file}")
        checkpoint_path = str(checkpoint_file)
        checkpoint_folder = checkpoint_path_obj
    else:
        # It's a file path (legacy support)
        checkpoint_file = checkpoint_path_obj
        checkpoint_folder = checkpoint_path_obj.parent

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try new format first (gpt_config object)
    config = checkpoint.get("gpt_config")
    if config is None:
        # Try legacy format (config key)
        config = checkpoint.get("config")
        if config is None:
            config = GPTConfig()
        elif isinstance(config, dict):
            config = GPTConfig(**config)
    elif isinstance(config, dict):
        # If it's a dict (from state_dict), reconstruct
        config = GPTConfig(**config)
    # If it's already a GPTConfig object, use it directly

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    loss = checkpoint.get("train_loss", "unknown")

    print(f"✓ Loaded checkpoint from epoch {epoch} (train loss: {loss})")
    return model, config, epoch, checkpoint_folder


def calculate_perplexity(model: GPT, data_loader, device: torch.device) -> float:
    """
    Calculate perplexity on a dataset.

    Perplexity = exp(cross_entropy_loss)
    Lower perplexity = better model (model is less "perplexed" by the data)

    Args:
        model: The GPT model
        data_loader: DataLoader with validation/test data
        device: Device to run evaluation on

    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    print("Calculating perplexity...")
    vocab_size = model.vocab_size

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            # Debug: Check for invalid token IDs before clamping
            if batch_idx == 0:
                max_input = input_ids.max().item()
                min_input = input_ids.min().item()
                max_target = target_ids.max().item()
                min_target = target_ids.min().item()
                print(
                    f"  First batch - Input IDs range: [{min_input}, {max_input}], Target IDs range: [{min_target}, {max_target}]"
                )
                print(f"  Model vocab_size: {vocab_size}")
                if max_input >= vocab_size or max_target >= vocab_size:
                    print("  ⚠️  Found invalid token IDs! Clamping...")

            # Clamp token IDs to valid vocabulary range to prevent index errors
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            target_ids = torch.clamp(target_ids, 0, vocab_size - 1)

            # Ensure target_ids matches input_ids sequence length
            # (in case model truncated input_ids in forward pass)
            seq_len = input_ids.size(1)
            if target_ids.size(1) != seq_len:
                target_ids = target_ids[:, :seq_len]

            # Forward pass
            logits = model(input_ids)

            # Ensure logits and target_ids have matching sequence lengths
            logits_seq_len = logits.size(1)
            target_seq_len = target_ids.size(1)

            if logits_seq_len != target_seq_len:
                min_len = min(logits_seq_len, target_seq_len)
                logits = logits[:, :min_len, :]
                target_ids = target_ids[:, :min_len]

            # Calculate loss
            # Use reshape instead of view to handle non-contiguous tensors
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = target_ids.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="sum")

            # Count non-padding tokens (assuming 0 is padding)
            num_tokens = (targets_flat != 0).sum().item()

            total_loss += loss.item()
            total_tokens += num_tokens
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Average loss per token
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss, total_tokens


def calculate_accuracy(
    model: GPT, data_loader, device: torch.device, top_k: int = 1
) -> dict[str, float]:
    """
    Calculate token prediction accuracy.

    Args:
        model: The GPT model
        data_loader: DataLoader with validation/test data
        device: Device to run evaluation on
        top_k: Calculate top-k accuracy (1 = exact match, 5 = in top 5 predictions)

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    correct = 0
    total = 0
    top_k_correct = 0

    print(f"Calculating accuracy (top-{top_k})...")
    vocab_size = model.vocab_size

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            # Clamp token IDs to valid vocabulary range to prevent index errors
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            target_ids = torch.clamp(target_ids, 0, vocab_size - 1)

            # Ensure target_ids matches input_ids sequence length
            seq_len = input_ids.size(1)
            if target_ids.size(1) != seq_len:
                target_ids = target_ids[:, :seq_len]

            # Forward pass
            logits = model(input_ids)

            # Ensure logits and target_ids have matching sequence lengths
            logits_seq_len = logits.size(1)
            target_seq_len = target_ids.size(1)

            if logits_seq_len != target_seq_len:
                min_len = min(logits_seq_len, target_seq_len)
                logits = logits[:, :min_len, :]
                target_ids = target_ids[:, :min_len]

            # FIX: Evaluate ALL positions, not just the last one
            # Flatten to (batch_size * seq_length, vocab_size)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = target_ids.reshape(-1)

            # Create mask to ignore padding tokens (assuming 0 is padding)
            mask = targets_flat != 0

            # Get predictions for all positions
            predictions = torch.argmax(logits_flat, dim=-1)

            # Top-1 accuracy (only count non-padding tokens)
            correct += ((predictions == targets_flat) & mask).sum().item()
            total += mask.sum().item()

            # Top-k accuracy
            if top_k > 1:
                top_k_preds = torch.topk(logits_flat, k=min(top_k, logits_flat.size(-1)), dim=-1)[1]
                # Check if target is in top-k for each position
                targets_expanded = targets_flat.unsqueeze(1).expand_as(top_k_preds)
                top_k_matches = (top_k_preds == targets_expanded).any(dim=1)
                top_k_correct += (top_k_matches & mask).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    accuracy = (correct / total * 100) if total > 0 else 0.0
    top_k_acc = (top_k_correct / total * 100) if total > 0 and top_k > 1 else None

    results = {"accuracy": accuracy, "correct": correct, "total": total}
    if top_k_acc is not None:
        results[f"top_{top_k}_accuracy"] = top_k_acc

    return results


def evaluate_generation_quality(
    model: GPT,
    tokenizer: Tokenizer,
    test_prompts: list[str],
    device: torch.device,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> dict[str, str]:
    """
    Evaluate generation quality on test prompts.

    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        test_prompts: List of test prompts
        device: Device to run generation on
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary mapping prompts to generated completions
    """
    model.eval()
    results = {}

    print(f"Evaluating generation on {len(test_prompts)} prompts...")
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            # Encode prompt
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Generate with improved sampling
            generated_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())

            results[prompt] = generated_text
            print(f"  [{i + 1}/{len(test_prompts)}] Prompt: {prompt[:50]}...")

    return results


def evaluate_model(
    checkpoint_path: str,
    tokenizer: Tokenizer,
    device: torch.device,
    eval_config: EvaluationConfig = None,
    checkpoint_folder: Path = None,
):
    """
    Comprehensive model evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer: Tokenizer instance
        device: Device to run evaluation on
        eval_config: EvaluationConfig instance (uses defaults if None)
    """
    if eval_config is None:
        eval_config = EvaluationConfig()

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print()

    # Load model (will return checkpoint_folder if not provided)
    model, config, epoch, loaded_checkpoint_folder = load_model_from_checkpoint(
        checkpoint_path, device
    )
    # Use provided checkpoint_folder or the one from loading
    if checkpoint_folder is None:
        checkpoint_folder = loaded_checkpoint_folder

    # Verify vocab_size matches tokenizer
    if model.vocab_size != tokenizer.vocab_size:
        print(
            f"⚠️  WARNING: Model vocab_size ({model.vocab_size}) != Tokenizer vocab_size ({tokenizer.vocab_size})"
        )
        print(f"   This could cause index errors. Using model's vocab_size: {model.vocab_size}")
    else:
        print(f"✓ Vocab sizes match: {model.vocab_size}")
    print()

    # Load evaluation dataset
    print(f"Loading evaluation dataset ({eval_config.eval_samples} samples)...")
    eval_dataset = load_validation_data(
        tokenizer=tokenizer,
        max_length=eval_config.max_length,
        streaming=True,
        max_samples=eval_config.eval_samples,
        text_field="synthetic_answer",
    )

    eval_loader, _ = create_data_loaders(
        train_dataset=eval_dataset,
        batch_size=eval_config.batch_size,
        num_workers=0,  # Set to 0 for IterableDataset
    )
    print()

    # 1. Calculate Perplexity
    print("1. PERPLEXITY EVALUATION")
    print("-" * 60)
    perplexity, avg_loss, num_tokens = calculate_perplexity(model, eval_loader, device)
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Tokens evaluated: {num_tokens:,}")
    print()

    # 2. Calculate Accuracy
    print("2. ACCURACY EVALUATION")
    print("-" * 60)
    # Need to recreate loader since we exhausted it
    eval_dataset = load_validation_data(
        tokenizer=tokenizer,
        max_length=eval_config.max_length,
        streaming=True,
        max_samples=eval_config.eval_samples,
        text_field="synthetic_answer",
    )
    eval_loader, _ = create_data_loaders(
        train_dataset=eval_dataset, batch_size=eval_config.batch_size, num_workers=0
    )

    accuracy_results = calculate_accuracy(
        model, eval_loader, device, top_k=eval_config.top_k_accuracy
    )
    print(f"  Top-1 Accuracy: {accuracy_results['accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {accuracy_results.get('top_5_accuracy', 0):.2f}%")
    print(f"  Correct: {accuracy_results['correct']:,} / {accuracy_results['total']:,}")
    print()

    # 3. Generation Quality Tests
    print("3. GENERATION QUALITY EVALUATION")
    print("-" * 60)
    # Use test prompts from eval_config
    test_prompts = eval_config.test_prompts

    # Use generation config from eval_config (ensures consistency)
    gen_config = eval_config.generation_config
    generation_results = evaluate_generation_quality(
        model,
        tokenizer,
        test_prompts,
        device,
        max_tokens=gen_config.max_new_tokens,
        temperature=gen_config.temperature,
        top_k=gen_config.top_k,
        top_p=gen_config.top_p,
        repetition_penalty=gen_config.repetition_penalty,
    )

    for prompt, generated in generation_results.items():
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {generated[:200]}...")
    print()

    # Summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {epoch}")
    print(f"Perplexity: {perplexity:.2f} (lower is better)")
    print(f"Top-1 Accuracy: {accuracy_results['accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {accuracy_results.get('top_5_accuracy', 0):.2f}%")
    print("=" * 60)

    # Get total_batches from checkpoint if available
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    total_batches = checkpoint.get("total_batches", "unknown")

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "accuracy": accuracy_results["accuracy"],
        "top_5_accuracy": accuracy_results.get("top_5_accuracy", 0),
        "generations": generation_results,
        "epoch": epoch,
        "total_batches": total_batches,
    }


def main():
    """Main evaluation function."""
    # Load default evaluation config
    eval_config = EvaluationConfig()

    parser = argparse.ArgumentParser(description="Evaluate GPT model checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=eval_config.eval_samples,
        help=f"Number of samples to use for evaluation (default: {eval_config.eval_samples})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=eval_config.batch_size,
        help=f"Batch size for evaluation (default: {eval_config.batch_size})",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=eval_config.max_length,
        help=f"Maximum sequence length (default: {eval_config.max_length})",
    )

    # Generation parameters (from GenerationConfig)
    gen_config = eval_config.generation_config
    parser.add_argument(
        "--temperature",
        type=float,
        default=gen_config.temperature,
        help=f"Temperature for generation (default: {gen_config.temperature})",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=gen_config.top_k,
        help=f"Top-k for generation (default: {gen_config.top_k})",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=gen_config.top_p,
        help=f"Top-p for generation (default: {gen_config.top_p})",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=gen_config.repetition_penalty,
        help=f"Repetition penalty for generation (default: {gen_config.repetition_penalty})",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu), auto-detected if not specified",
    )

    args = parser.parse_args()

    # Update eval_config with command line arguments
    eval_config.eval_samples = args.eval_samples
    eval_config.batch_size = args.batch_size
    eval_config.max_length = args.max_length
    eval_config.generation_config.temperature = args.temperature
    eval_config.generation_config.top_k = args.top_k
    eval_config.generation_config.top_p = args.top_p
    eval_config.generation_config.repetition_penalty = args.repetition_penalty

    # Set device
    if args.device is None:
        device = get_best_device()
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print()

    # Initialize tokenizer
    tokenizer = Tokenizer(encoding_name="cl100k_base")

    # Determine checkpoint folder (handle both folder and file paths)
    checkpoint_path_obj = Path(args.checkpoint)
    if checkpoint_path_obj.is_dir():
        checkpoint_folder = checkpoint_path_obj
        checkpoint_file_path = str(checkpoint_folder / "checkpoint.pt")
    else:
        # Legacy: it's a file, use parent as folder
        checkpoint_folder = checkpoint_path_obj.parent
        checkpoint_file_path = args.checkpoint

    # Run evaluation
    results = evaluate_model(
        checkpoint_path=checkpoint_file_path,
        tokenizer=tokenizer,
        device=device,
        eval_config=eval_config,
        checkpoint_folder=checkpoint_folder,
    )

    # Save results in the checkpoint folder
    results_file = checkpoint_folder / "eval_results.txt"

    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {results.get('epoch', 'unknown')}\n")
        f.write(f"Total Batches: {results.get('total_batches', 'unknown')}\n")
        f.write(f"Perplexity: {results['perplexity']:.2f}\n")
        f.write(f"Loss: {results['loss']:.4f}\n")
        f.write(f"Top-1 Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {results['top_5_accuracy']:.2f}%\n")
        f.write("\nGenerations:\n")
        for prompt, generated in results["generations"].items():
            f.write(f"\nPrompt: {prompt}\n")
            f.write(f"Generated: {generated}\n")

    print(f"\n✓ Saved evaluation results to {results_file}")


if __name__ == "__main__":
    main()
