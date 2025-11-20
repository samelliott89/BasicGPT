"""
Script to generate text from a trained GPT model checkpoint.

This script loads a saved checkpoint and uses it to generate text.
"""

import torch
import argparse
from pathlib import Path
from tokenizer import Tokenizer
from gpt import GPT
from config import GPTConfig, GenerationConfig



def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config, epoch, loss)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint with weights_only=False to allow custom classes like GPTConfig
    # This is safe since we're loading our own checkpoint files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint (try both 'gpt_config' and 'config' for compatibility)
    config = checkpoint.get('gpt_config') or checkpoint.get('config')
    if config is None:
        print("Warning: No config found in checkpoint, using defaults")
        config = GPTConfig()
    elif isinstance(config, dict):
        # Convert dict to GPTConfig if needed
        config = GPTConfig(**config)
    # If it's already a GPTConfig object, use it directly
    
    # Create model with the config
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('train_loss', 'unknown')
    
    print(f"✓ Loaded checkpoint from epoch {epoch}")
    print(f"  Training loss: {loss}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    return model, config, epoch, loss


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: torch.device = None
) -> str:
    """
    Generate text from a prompt using the model.
    
    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Controls randomness (higher = more random, lower = more deterministic)
        device: Device to run generation on
        
    Returns:
        Generated text (including the original prompt)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    
    # Clamp token IDs to valid vocabulary range
    vocab_size = tokenizer.vocab_size
    input_ids = [min(max(token, 0), vocab_size - 1) for token in input_ids]
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print(f"Prompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...")
    print(f"  Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}, Repetition penalty: {repetition_penalty}")
    print("-" * 60)
    
    # Generate
    with torch.no_grad():  # Don't compute gradients during generation
        generated_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    
    # Decode the generated tokens
    generated_ids_list = generated_ids[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids_list)
    
    return generated_text


def main():
    """Main function for text generation."""
    # Load default generation config
    gen_config = GenerationConfig()
    
    parser = argparse.ArgumentParser(description="Generate text from GPT checkpoint")
    parser.add_argument("checkpoint", type=str,
                       help="Path to checkpoint file (e.g., ./checkpoints/checkpoint_epoch_1.pt)")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence",
                       help="Starting prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=gen_config.max_new_tokens,
                       help=f"Maximum number of new tokens to generate (default: {gen_config.max_new_tokens})")
    parser.add_argument("--temperature", type=float, default=gen_config.temperature,
                       help=f"Temperature for sampling (default: {gen_config.temperature}, higher = more random)")
    parser.add_argument("--top_k", type=int, default=gen_config.top_k,
                       help=f"Top-k sampling: only consider top k tokens (default: {gen_config.top_k}, 0 = disabled)")
    parser.add_argument("--top_p", type=float, default=gen_config.top_p,
                       help=f"Top-p (nucleus) sampling: cumulative probability threshold (default: {gen_config.top_p}, 0.0 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=gen_config.repetition_penalty,
                       help=f"Penalty for repeating tokens (default: {gen_config.repetition_penalty}, 1.0 = no penalty)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu), auto-detected if not specified")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    model, config, epoch, loss = load_checkpoint(str(checkpoint_path), device)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    print(f"✓ Tokenizer initialized (vocab_size={tokenizer.vocab_size})")
    print()
    
    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device
    )
    
    # Print results
    print()
    print("=" * 60)
    print("Generated Text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    print()
    
    # Optionally save to file
    output_file = checkpoint_path.parent / f"generated_epoch_{epoch}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Max tokens: {args.max_tokens}\n")
        f.write("-" * 60 + "\n")
        f.write(generated_text)
    
    print(f"✓ Saved generated text to {output_file}")


if __name__ == "__main__":
    main()

