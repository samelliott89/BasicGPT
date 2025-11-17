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
from gpt import GPT, train_epoch, evaluate
from prepare_data import load_synth_dataset, create_data_loaders
from device import get_best_device, get_device_info
from config import GPTConfig, TrainingConfig, DataConfig

def main():
    """Main training function."""
    # Load default configurations
    training_config = TrainingConfig()
    gpt_config = training_config.gpt_config
    data_config = training_config.data_config
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GPT on SYNTH dataset")
    
    # Model architecture arguments
    parser.add_argument("--d_model", type=int, default=gpt_config.d_model,
                       help=f"Model dimension (default: {gpt_config.d_model})")
    parser.add_argument("--n_heads", type=int, default=gpt_config.n_heads,
                       help=f"Number of attention heads (default: {gpt_config.n_heads})")
    parser.add_argument("--n_layers", type=int, default=gpt_config.n_layers,
                       help=f"Number of transformer layers (default: {gpt_config.n_layers})")
    parser.add_argument("--max_length", type=int, default=gpt_config.max_length,
                       help=f"Maximum sequence length (default: {gpt_config.max_length})")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=training_config.batch_size,
                       help=f"Batch size for training (default: {training_config.batch_size})")
    parser.add_argument("--epochs", type=int, default=training_config.epochs,
                       help=f"Number of training epochs (default: {training_config.epochs})")
    parser.add_argument("--learning_rate", type=float, default=training_config.learning_rate,
                       help=f"Learning rate (default: {training_config.learning_rate})")
    parser.add_argument("--save_dir", type=str, default=training_config.save_dir,
                       help=f"Directory to save model checkpoints (default: {training_config.save_dir})")
    
    # Data arguments
    parser.add_argument("--max_samples", type=int, default=data_config.max_samples,
                       help=f"Maximum number of samples to use (default: {data_config.max_samples}, None = all)")
    parser.add_argument("--streaming", dest="streaming", action="store_const", const=True, default=data_config.streaming,
                       help=f"Use streaming mode for large datasets (default: {data_config.streaming})")
    parser.add_argument("--no-streaming", dest="streaming", action="store_const", const=False,
                       help="Disable streaming mode (downloads entire dataset)")
    parser.add_argument("--timeout", type=int, default=data_config.timeout,
                       help=f"Timeout in seconds for dataset download (default: {data_config.timeout})")
    parser.add_argument("--num-retries", type=int, default=data_config.num_retries,
                       help=f"Number of retry attempts on connection failure (default: {data_config.num_retries})")
    parser.add_argument("--text_field", type=str, default=data_config.text_field,
                       help=f"Which field to use from dataset (default: {data_config.text_field})")
    
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
        text_field=args.text_field,
        num_retries=args.num_retries,
        timeout=args.timeout
    )
    # Note: IterableDataset doesn't support len(), so we skip this for streaming mode
    try:
        print(f"✓ Loaded {len(train_dataset)} training samples")
    except TypeError:
        print(f"✓ Loaded training dataset (streaming mode - length unknown)")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    # Determine number of workers (0 for IterableDataset/streaming, otherwise use config)
    num_workers = 0 if args.streaming else (data_config.num_workers if device.type == "cuda" else 0)
    
    train_loader, _ = create_data_loaders(
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    # Note: DataLoaders with IterableDataset don't support len()
    try:
        print(f"✓ Created train loader with {len(train_loader)} batches")
    except TypeError:
        print(f"✓ Created train loader (streaming mode - batch count unknown)")
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
    model = GPT(gpt_config)
    model = model.to(device)  # Move model to device
    
    # Enable mixed precision training (FP16/BF16) to save memory
    # This can reduce memory usage by ~50% with minimal accuracy loss
    use_amp = training_config.use_mixed_precision and device.type == "cuda"  # Only use AMP on CUDA
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("  Using Automatic Mixed Precision (AMP) for memory efficiency")
    
    # Clear CUDA cache after model creation
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"  GPU Memory after model load: {torch.cuda.memory_allocated(device) / (1024**3):.2f}GB")
    
    # Count parameters and estimate model size
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimate model size in GB (assuming float32 = 4 bytes)
    model_size_gb = num_params * 4 / (1024 ** 3)
    
    print(f"✓ Model created!")
    print(f"  Total parameters: {num_params:,} ({num_params/1e9:.2f}B)")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Estimated model size: {model_size_gb:.2f} GB (float32)")
    print(f"  Vocab size: {gpt_config.vocab_size:,}")
    print(f"  Model dimension: {gpt_config.d_model}")
    print(f"  Number of layers: {gpt_config.n_layers}")
    print(f"  Number of heads: {gpt_config.n_heads}")
    print(f"  Max sequence length: {gpt_config.max_length}")
    print()
    
    # Create optimizer with weight decay for regularization
    # Using AdamW (Adam with weight decay) is better for large models
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=(training_config.beta1, training_config.beta2)
    )
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler
        )
        
        print(f"\nEpoch {epoch + 1} completed!")
        print(f"  Average training loss: {train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': gpt_config,  # Save GPTConfig (model architecture)
            'train_loss': train_loss,
        }, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()

