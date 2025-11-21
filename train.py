"""
Main training script for GPT model on SYNTH dataset.

This script ties everything together:
1. Loads and prepares the SYNTH dataset
2. Creates the GPT model
3. Trains the model
4. Saves checkpoints
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from tokenizer import Tokenizer
from gpt import GPT, train_epoch, evaluate
from prepare_data import load_training_data, load_validation_data, create_data_loaders
from device import get_best_device, get_device_info
from config import GPTConfig, TrainingConfig, DataConfig
from learning_rate import get_lr

def save_checkpoint(checkpoint_path, epoch, total_batches, model, optimizer, scheduler, 
                   gpt_config, training_config, data_config, train_loss, val_loss, lr):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'total_batches': total_batches,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'gpt_config': gpt_config,
        'training_config': training_config,
        'data_config': data_config,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': lr,
    }, checkpoint_path)

def evaluate_validation(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    print("  Running validation...")
    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    reduction='sum'
                )
            
            # Count non-padding tokens
            num_tokens = (target_ids != 0).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
            
            if batch_idx >= 100:  # Evaluate on ~100 batches for speed
                break
    
    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')

    return avg_loss

def main():
    """Main training function."""
    # Load default configurations
    training_config = TrainingConfig()
    gpt_config = GPTConfig()
    data_config = DataConfig()
    
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
    train_dataset = load_training_data(
        tokenizer=tokenizer,
        max_length=data_config.max_length,
        streaming=data_config.streaming,
        max_samples=data_config.max_samples,
        text_field=data_config.text_field,
        num_retries=data_config.num_retries,
        timeout=data_config.timeout
    )
    # Note: IterableDataset doesn't support len(), so we skip this for streaming mode
    try:
        print(f"✓ Loaded {len(train_dataset)} training samples")
    except TypeError:
        print(f"✓ Loaded training dataset (streaming mode - length unknown)")
    print()
    
    # Load validation dataset
    print("Loading SYNTH validation dataset...")
    val_dataset = load_validation_data(
        tokenizer=tokenizer,
        max_length=data_config.max_length,
        streaming=data_config.streaming,
        max_samples=data_config.max_samples,
        text_field=data_config.text_field,
        num_retries=data_config.num_retries,
        timeout=data_config.timeout
    )
    try:
        print(f"✓ Loaded {len(val_dataset)} validation samples")
    except TypeError:
        print(f"✓ Loaded validation dataset (streaming mode - length unknown)")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    # Determine number of workers (0 for IterableDataset/streaming, otherwise use config)
    num_workers = 0 if data_config.streaming else (data_config.num_workers if device.type == "cuda" else 0)
    
    train_loader, val_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config.batch_size,
        num_workers=num_workers
    )

    # Note: DataLoaders with IterableDataset don't support len()
    try:
        print(f"✓ Created train loader with {len(train_loader)} batches")
    except TypeError:
        print(f"✓ Created train loader (streaming mode - batch count unknown)")
    try:
        print(f"✓ Created val loader with {len(val_loader)} batches")
    except TypeError:
        print(f"✓ Created val loader (streaming mode - batch count unknown)")
    print()
    
    # Create model configuration with tokenizer's vocab_size
    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size
    )
    
    # Calculate effective batch size
    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps

    
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
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Gradient accumulation steps: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print()
    
    # Create optimizer with weight decay for regularization
    # Using AdamW (Adam with weight decay) is better for large models
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1.0, # Set to 1.0 so LambdaLR can scale it properly
        weight_decay=training_config.weight_decay,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.eps
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, training_config.lr_config)
    )
    
    # Create save directory
    save_dir = Path(training_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    # Track total batch count across epochs for checkpoint naming
    total_batches = 0
    checkpoint_interval = 50000  # Save checkpoint every 50k batches
    best_val_loss = float('inf')  # Track best validation loss
    
    for epoch in range(training_config.epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.epochs}")
        print("-" * 60)
        
        # Train for one epoch
        train_loss, batches_processed = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            scaler=scaler,
            save_dir=save_dir,
            total_batches=total_batches,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            val_loader=val_loader,
            eval_fn=evaluate_validation,
            training_config=training_config
        )
        
        total_batches += batches_processed

        # Evaluate on validation set if available
        val_loss = None
        if val_loader is not None:
            print("\nEvaluating on validation set...")
            val_loss = evaluate_validation(model, val_loader, device)
        
        print(f"\nEpoch {epoch + 1} completed!")
        print(f"  Average training loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Average validation loss: {val_loss:.4f}")
        print(f"  Total batches processed: {total_batches:,}")
        
        # Save checkpoint at end of epoch
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
        # Create folder for this checkpoint
        max_samples_str = f"{data_config.max_samples // 1_000_000}m" if data_config.max_samples else "all"
        checkpoint_folder_name = f"data-{max_samples_str}-batch-{total_batches}-{timestamp}"
        checkpoint_folder = save_dir / checkpoint_folder_name
        checkpoint_folder.mkdir(parents=True, exist_ok=True)

        # Save regular epoch checkpoint (always)
        checkpoint_file = checkpoint_folder / "checkpoint.pt"
        save_checkpoint(checkpoint_file, 
                epoch + 1, 
                total_batches, 
                model, 
                optimizer, 
                scheduler,
                gpt_config, 
                training_config, 
                data_config, 
                train_loss, 
                val_loss, 
                optimizer.param_groups[0]['lr']
            )
        print(f"  Saved epoch checkpoint to {checkpoint_file}")

        # Save best checkpoint (only if improved and validation loss is available)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save in main save_dir, NOT inside epoch folder
            best_checkpoint_file = save_dir / "checkpoint_best.pt"
            save_checkpoint(best_checkpoint_file, 
                epoch + 1, 
                total_batches, 
                model, 
                optimizer, 
                scheduler,
                gpt_config, 
                training_config, 
                data_config, 
                train_loss, 
                val_loss, 
                optimizer.param_groups[0]['lr']
            )
            print(f"  ✓ New best validation loss: {val_loss:.4f}!")
            print(f"  ✓ Saved best checkpoint to {best_checkpoint_file}")
        elif val_loss is not None:
            print(f"  Val loss did not improve (best: {best_val_loss:.4f})")
    
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Checkpoints saved in: {save_dir}")

if __name__ == "__main__":
    main()

