"""
Main training script for GPT model on SYNTH dataset.

This script ties everything together:
1. Loads and prepares the SYNTH dataset
2. Creates the GPT model
3. Trains the model
4. Saves checkpoints
"""

import torch
import argparse
from pathlib import Path

from tokenizer import Tokenizer
from gpt import GPT, GPTConfig, train_epoch, evaluate
from prepare_data import load_synth_dataset, create_data_loaders
from device import get_best_device, get_device_info


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GPT on SYNTH dataset")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to use (None = all)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--d_model", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save model checkpoints")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming mode for large datasets")
    parser.add_argument("--text_field", type=str, default="synthetic_answer",
                       help="Which field to use from dataset")
    
    args = parser.parse_args()
    
    # Set device (use GPU if available)
    device = get_best_device()
    print(f"Using device: {device} - {get_device_info(device)}")
    print()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    print(f"✓ Tokenizer initialized (vocab_size={tokenizer.vocab_size})")
    print()
    
    # Load training dataset
    print("Loading SYNTH training dataset...")
    train_dataset = load_synth_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="train",
        streaming=args.streaming,
        max_samples=args.max_samples,
        text_field=args.text_field
    )
    print(f"✓ Loaded {len(train_dataset)} training samples")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, _ = create_data_loaders(
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4 if device.type == "cuda" else 0
    )
    print(f"✓ Created train loader with {len(train_loader)} batches")
    print()
    
    # Create model configuration
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    
    # Create model
    print("Creating GPT model...")
    model = GPT(config)
    model = model.to(device)  # Move model to device
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Number of layers: {config.n_layers}")
    print(f"  Number of heads: {config.n_heads}")
    print()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 60)
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        print(f"\nEpoch {epoch + 1} completed!")
        print(f"  Average training loss: {train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_loss': train_loss,
        }, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()

