"""
Main training script for GPT model with multi-GPU support via Accelerate.

This script supports:
- Single GPU: python train.py
- Multi-GPU:  accelerate launch train.py

The training pipeline:
1. Loads and prepares datasets (with probabilistic multi-dataset sampling)
2. Creates the GPT model
3. Trains with distributed data parallel (if multi-GPU)
4. Saves checkpoints
5. Logs metrics to Weights & Biases (optional)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from functools import partial
import os

from accelerate import Accelerator
from accelerate.utils import set_seed

from tokenizer import Tokenizer
from gpt import GPT, train_epoch, evaluate
from prepare_data import load_datasets, create_data_loaders
from config import GPTConfig, TrainingConfig, DataConfig
from learning_rate import get_lr

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run 'pip install wandb' for logging.")


def save_checkpoint(accelerator, checkpoint_path, epoch, total_batches, model, optimizer, scheduler, 
                   gpt_config, training_config, data_config, train_loss, val_loss, lr):
    """Save training checkpoint (only on main process)."""
    # unwrap model if using DDP
    unwrapped_model = accelerator.unwrap_model(model)
    
    accelerator.save({
        'epoch': epoch,
        'total_batches': total_batches,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'gpt_config': gpt_config,
        'training_config': training_config,
        'data_config': data_config,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': lr,
    }, checkpoint_path)


def evaluate_validation(model, val_loader, accelerator, num_batches=100):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    if accelerator.is_main_process:
        print("  Running validation...")
    
    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
            # Data is already on correct device via accelerator.prepare()
            
            # Forward pass (mixed precision handled by accelerator)
            with accelerator.autocast():
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
            
            if batch_idx >= num_batches:
                break
    
    model.train()
    
    # Gather losses across all processes
    total_loss_tensor = torch.tensor([total_loss], device=accelerator.device)
    total_tokens_tensor = torch.tensor([total_tokens], device=accelerator.device)
    
    gathered_loss = accelerator.gather(total_loss_tensor).sum().item()
    gathered_tokens = accelerator.gather(total_tokens_tensor).sum().item()
    
    avg_loss = gathered_loss / gathered_tokens if gathered_tokens > 0 else float('inf')
    return avg_loss


def main():
    """Main training function with multi-GPU support."""
    
    # Initialize Accelerator
    # - Handles device placement automatically
    # - Handles mixed precision (set via accelerate config or here)
    # - Handles distributed training
    accelerator = Accelerator(
        mixed_precision="bf16",  # Use bfloat16 for modern GPUs (Blackwell, A100)
        gradient_accumulation_steps=4,  # Will be overridden by config
        log_with="wandb" if WANDB_AVAILABLE else None,
    )
    
    # Set seed for reproducibility across all processes
    set_seed(42)
    
    # Load default configurations
    training_config = TrainingConfig()
    gpt_config = GPTConfig()
    data_config = DataConfig()
    
    # Initialize W&B (only on main process)
    if WANDB_AVAILABLE and accelerator.is_main_process:
        # Create run name with timestamp
        run_name = f"gpt-{gpt_config.n_layers}L-{gpt_config.d_model}D-{datetime.now().strftime('%m%d-%H%M')}"
        
        # Initialize W&B through accelerator
        accelerator.init_trackers(
            project_name="BasicGPT",
            config={
                # Model config
                "model/vocab_size": gpt_config.vocab_size,
                "model/d_model": gpt_config.d_model,
                "model/n_heads": gpt_config.n_heads,
                "model/n_layers": gpt_config.n_layers,
                "model/max_length": gpt_config.max_length,
                "model/dropout": gpt_config.dropout,
                # Training config
                "training/batch_size": training_config.batch_size,
                "training/gradient_accumulation": training_config.gradient_accumulation_steps,
                "training/epochs": training_config.epochs,
                "training/weight_decay": training_config.weight_decay,
                # Data config
                "data/datasets": [ds.value for ds in data_config.current_datasets],
                "data/probabilities": data_config.dataset_probabilities,
                "data/max_samples": data_config.max_samples,
                "data/max_length": data_config.max_length,
                # Hardware
                "hardware/num_gpus": accelerator.num_processes,
                "hardware/mixed_precision": accelerator.mixed_precision,
            },
            init_kwargs={"wandb": {"name": run_name}},
        )
        print(f"✓ W&B initialized: {run_name}")
    
    # Update accelerator with config values
    accelerator.gradient_accumulation_steps = training_config.gradient_accumulation_steps
    
    # Print info only on main process
    if accelerator.is_main_process:
        print("=" * 60)
        print("BasicGPT Training with Accelerate")
        print("=" * 60)
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")
        print(f"Device: {accelerator.device}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
        print()
    
    # Initialize tokenizer (same on all processes)
    if accelerator.is_main_process:
        print("Initializing tokenizer...")
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    if accelerator.is_main_process:
        print(f"✓ Tokenizer initialized (vocab_size={tokenizer.vocab_size})")
        print()
    
    # Wait for all processes before loading data
    accelerator.wait_for_everyone()
    
    # Load training dataset(s)
    # Each GPU streams its own copy of the data (simpler, slight overlap OK for pretraining)
    if accelerator.is_main_process:
        print("=" * 60)
        print(f"Loading training dataset(s): {[ds.value for ds in data_config.current_datasets]}")
        print(f"Dataset probabilities: {data_config.dataset_probabilities}")
        print("=" * 60)
    
    # Use different random seed per process for dataset shuffling
    process_seed = 42 + accelerator.process_index
    
    train_dataset = load_datasets(
        dataset_names=data_config.current_datasets,
        tokenizer=tokenizer,
        is_training=True,
        probabilities=data_config.dataset_probabilities,
        max_length=data_config.max_length,
        streaming=data_config.streaming,
        max_samples=data_config.max_samples,
        num_retries=data_config.num_retries,
        timeout=data_config.timeout
    )
    
    if accelerator.is_main_process:
        print()
        print("=" * 60)
        print(f"Loading validation dataset(s): {[ds.value for ds in data_config.current_datasets]}")
        print("=" * 60)
    
    val_dataset = load_datasets(
        dataset_names=data_config.current_datasets,
        tokenizer=tokenizer,
        is_training=False,
        probabilities=data_config.dataset_probabilities,
        max_length=data_config.max_length,
        streaming=data_config.streaming,
        max_samples=data_config.max_samples,
        num_retries=data_config.num_retries,
        timeout=data_config.timeout
    )
    
    # Create data loaders
    if accelerator.is_main_process:
        print()
        print("Creating data loaders...")
    
    # For streaming/IterableDataset, use 0 workers
    num_workers = 0 if data_config.streaming else data_config.num_workers
    
    train_loader, val_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config.batch_size,
        num_workers=num_workers
    )
    
    if accelerator.is_main_process:
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
    
    # Calculate effective batch size (across all GPUs)
    per_gpu_batch = training_config.batch_size
    grad_accum = training_config.gradient_accumulation_steps
    num_gpus = accelerator.num_processes
    effective_batch_size = per_gpu_batch * grad_accum * num_gpus
    
    # Create model
    if accelerator.is_main_process:
        print("Creating GPT model...")
    model = GPT(gpt_config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_gb = num_params * 4 / (1024 ** 3)
    
    if accelerator.is_main_process:
        print(f"✓ Model created!")
        print(f"  Total parameters: {num_params:,} ({num_params/1e9:.2f}B)")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Estimated model size: {model_size_gb:.2f} GB (float32)")
        print(f"  Vocab size: {gpt_config.vocab_size:,}")
        print(f"  Model dimension: {gpt_config.d_model}")
        print(f"  Number of layers: {gpt_config.n_layers}")
        print(f"  Number of heads: {gpt_config.n_heads}")
        print(f"  Max sequence length: {gpt_config.max_length}")
        print(f"  Per-GPU batch size: {per_gpu_batch}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Effective batch size: {effective_batch_size}")
        print()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1.0,  # Scaled by scheduler
        weight_decay=training_config.weight_decay,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.eps
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, training_config.lr_config)
    )
    
    # Prepare everything with accelerator
    # This handles:
    # - Moving model to correct device
    # - Wrapping model with DDP (if multi-GPU)
    # - Preparing dataloaders for distributed sampling
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    if accelerator.is_main_process:
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"  GPU Memory after model load: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")
        print()
    
    # Create save directory (only main process creates it)
    save_dir = Path(training_config.save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait for directory creation
    accelerator.wait_for_everyone()
    
    # Training loop
    if accelerator.is_main_process:
        print("Starting training...")
        print("=" * 60)
    
    total_batches = 0
    best_val_loss = float('inf')
    
    for epoch in range(training_config.epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{training_config.epochs}")
            print("-" * 60)
        
        # Train for one epoch
        train_loss, batches_processed = train_epoch_accelerate(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            epoch=epoch,
            save_dir=save_dir,
            total_batches=total_batches,
            training_config=training_config,
            gpt_config=gpt_config,
            data_config=data_config,
            val_loader=val_loader,
        )
        
        total_batches += batches_processed
        
        # Evaluate on validation set
        val_loss = None
        if val_loader is not None:
            if accelerator.is_main_process:
                print("\nEvaluating on validation set...")
            val_loss = evaluate_validation(model, val_loader, accelerator)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1} completed!")
            print(f"  Average training loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Average validation loss: {val_loss:.4f}")
            print(f"  Total batches processed: {total_batches:,}")
        
        # Save checkpoint (only main process)
        if accelerator.is_main_process:
            timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
            max_samples_str = f"{data_config.max_samples // 1_000_000}m" if data_config.max_samples else "all"
            checkpoint_folder_name = f"data-{max_samples_str}-batch-{total_batches}-{timestamp}"
            checkpoint_folder = save_dir / checkpoint_folder_name
            checkpoint_folder.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = checkpoint_folder / "checkpoint.pt"
            save_checkpoint(
                accelerator, checkpoint_file,
                epoch + 1, total_batches, model, optimizer, scheduler,
                gpt_config, training_config, data_config,
                train_loss, val_loss, optimizer.param_groups[0]['lr']
            )
            print(f"  Saved epoch checkpoint to {checkpoint_file}")
            
            # Save best checkpoint
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_file = save_dir / "checkpoint_best.pt"
                save_checkpoint(
                    accelerator, best_checkpoint_file,
                    epoch + 1, total_batches, model, optimizer, scheduler,
                    gpt_config, training_config, data_config,
                    train_loss, val_loss, optimizer.param_groups[0]['lr']
                )
                print(f"  ✓ New best validation loss: {val_loss:.4f}!")
                print(f"  ✓ Saved best checkpoint to {best_checkpoint_file}")
            elif val_loss is not None:
                print(f"  Val loss did not improve (best: {best_val_loss:.4f})")
        
        # Log epoch summary to W&B
        accelerator.log({
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss if val_loss else 0,
            "epoch/number": epoch + 1,
        }, step=total_batches)
        
        # Sync all processes before next epoch
        accelerator.wait_for_everyone()
    
    # End W&B run
    accelerator.end_training()
    
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Checkpoints saved in: {save_dir}")
        if WANDB_AVAILABLE:
            print(f"View training logs at: https://wandb.ai")


def train_epoch_accelerate(
    model,
    train_loader,
    optimizer,
    scheduler,
    accelerator,
    epoch,
    save_dir,
    total_batches,
    training_config,
    gpt_config,
    data_config,
    val_loader=None,
):
    """
    Train for one epoch using Accelerate.
    
    This replaces the gpt.train_epoch function when using Accelerate.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer_steps = 0
    
    log_every_n_steps = training_config.log_every_n_steps
    val_check_interval = training_config.val_check_interval
    checkpoint_interval = training_config.checkpoint_interval
    
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        # Data is already on correct device via accelerator.prepare()
        
        # Clamp token IDs to valid vocabulary range
        vocab_size = accelerator.unwrap_model(model).vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
        
        # Use accelerator's gradient accumulation context
        with accelerator.accumulate(model):
            # Forward pass with automatic mixed precision
            with accelerator.autocast():
                logits = model(input_ids)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = target_ids.view(-1)
                loss = F.cross_entropy(logits_flat, targets_flat)
            
            # Backward pass (accelerator handles scaling)
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step (only when gradients are synced after accumulation)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track if we did an optimizer step
        if accelerator.sync_gradients:
            optimizer_steps += 1
            current_lr = scheduler.get_last_lr()[0]
            
            # Log training loss
            if optimizer_steps % log_every_n_steps == 0:
                if accelerator.device.type == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                else:
                    memory_allocated = 0
                    memory_reserved = 0
                
                # Log to W&B
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch,
                    "train/step": optimizer_steps,
                    "system/gpu_memory_allocated_gb": memory_allocated,
                    "system/gpu_memory_reserved_gb": memory_reserved,
                }, step=optimizer_steps)
                
                # Print to console
                if accelerator.is_main_process:
                    if accelerator.device.type == "cuda":
                        print(f"  Step {optimizer_steps} (Batch {batch_idx + 1}), "
                              f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                              f"GPU: {memory_allocated:.2f}GB")
                    else:
                        print(f"  Step {optimizer_steps} (Batch {batch_idx + 1}), Loss: {loss.item():.4f}")
            
            # Run validation
            if val_loader is not None and optimizer_steps % val_check_interval == 0:
                if accelerator.is_main_process:
                    print(f"\n  Running validation at step {optimizer_steps}...")
                val_loss = evaluate_validation(model, val_loader, accelerator)
                
                # Log validation loss to W&B
                accelerator.log({
                    "val/loss": val_loss,
                    "val/perplexity": torch.exp(torch.tensor(val_loss)).item(),
                }, step=optimizer_steps)
                
                if accelerator.is_main_process:
                    print(f"  Validation Loss: {val_loss:.4f}")
                model.train()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        current_total_batches = total_batches + num_batches
        
        # Save checkpoint at intervals (only main process)
        if (current_total_batches % checkpoint_interval == 0 and 
            accelerator.is_main_process and save_dir is not None):
            timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
            max_samples_str = f"{data_config.max_samples // 1_000_000}m" if data_config.max_samples else "all"
            checkpoint_folder_name = f"data-{max_samples_str}-batch-{current_total_batches}-{timestamp}"
            checkpoint_folder = save_dir / checkpoint_folder_name
            checkpoint_folder.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = checkpoint_folder / "checkpoint.pt"
            print(f"\n  💾 Saving checkpoint at batch {current_total_batches:,}...")
            save_checkpoint(
                accelerator, checkpoint_file,
                epoch + 1, current_total_batches, model, optimizer, scheduler,
                gpt_config, training_config, data_config,
                total_loss / num_batches, None, optimizer.param_groups[0]['lr']
            )
            print(f"  ✓ Saved to {checkpoint_file}")
        
        # Clear CUDA cache periodically
        if accelerator.device.type == "cuda" and (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, num_batches


if __name__ == "__main__":
    main()
