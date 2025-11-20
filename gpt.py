"""
GPT (Generative Pre-trained Transformer) model implementation.

This module implements a GPT-style language model using PyTorch.
GPT is a decoder-only transformer model that predicts the next token
in a sequence given the previous tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from tokenizer import Tokenizer
from config import GPTConfig, DataConfig, TrainingConfig

gpt_config = GPTConfig()
data_config = DataConfig()
training_config = TrainingConfig()

class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model.
    
    This is a decoder-only transformer model that:
    1. Takes token IDs as input
    2. Embeds them into vectors
    3. Adds positional information
    4. Processes through transformer layers
    5. Outputs logits (scores) for each possible next token
    """
    
    def __init__(self, config: GPTConfig):
        """
        Initialize the GPT model.
        
        Args:
            config: GPTConfig object with model hyperparameters
        """
        super(GPT, self).__init__()
        
        # Store configuration
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        
        # Token embedding: converts token IDs to vectors
        # Each token gets a d_model-dimensional vector
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position embedding: encodes position information
        # Each position (0, 1, 2, ...) gets a d_model-dimensional vector
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        
        # Dropout for regularization (helps prevent overfitting)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer encoder layers
        # GPT uses decoder-style layers with masked self-attention
        # We use TransformerEncoderLayer but with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,  # Feedforward dimension (4x d_model)
            dropout=config.dropout,
            activation='gelu',  # GELU activation (common in GPT)
            batch_first=True,  # Input shape: (batch, seq, features)
            norm_first=True  # Pre-norm architecture (more stable for deep models)
        )
        
        # Stack multiple transformer layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
        # Enable gradient checkpointing to save memory (can be disabled if causing issues)
        # This trades compute for memory by recomputing activations during backward pass
        # Set to False if you encounter memory issues (checkpointing can sometimes cause problems)
        self.use_checkpointing = False  # Disabled by default - enable if needed
        
        # Layer normalization (helps with training stability)
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Output projection: converts hidden states to vocabulary logits
        # This gives us scores for each possible next token
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            mask: Optional attention mask (for padding)
            
        Returns:
            Logits of shape (batch_size, seq_length, vocab_size)
            These are scores for each possible next token at each position
        """
        batch_size, seq_length = input_ids.shape
        
        # Clamp input_ids to valid vocabulary range to prevent index errors
        # This is a safety check in case clamping wasn't done earlier
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Ensure sequence length doesn't exceed max_length
        # Note: We truncate here, but caller should ensure data is already correct length
        # to avoid shape mismatches with target_ids
        if seq_length > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            seq_length = self.max_length
        
        # Create position indices: [0, 1, 2, ..., seq_length-1]
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_length)
        
        # Clamp position indices to valid range [0, max_length-1]
        positions = torch.clamp(positions, 0, self.max_length - 1)
        
        # Get embeddings
        # Token embeddings: convert token IDs to vectors
        token_embeds = self.token_embedding(input_ids)  # (batch_size, seq_length, d_model)
        
        # Position embeddings: add position information
        pos_embeds = self.position_embedding(positions)  # (batch_size, seq_length, d_model)
        
        # Combine token and position embeddings
        x = token_embeds + pos_embeds  # (batch_size, seq_length, d_model)
        x = self.dropout(x)
        
        # Create causal mask for GPT (prevents looking at future tokens)
        # This ensures the model only uses previous tokens to predict the next one
        causal_mask = self._generate_causal_mask(seq_length, device=input_ids.device)
        
        # Pass through transformer layers
        # The mask ensures causal attention (can't see future tokens)
        # Use gradient checkpointing if enabled to save memory
        if self.use_checkpointing and self.training:
            # Gradient checkpointing: recompute activations during backward pass
            # This saves memory at the cost of extra computation
            # We need to wrap it in a lambda to pass the mask as a keyword argument
            def transformer_forward(x):
                return self.transformer(x, mask=causal_mask)
            x = checkpoint(transformer_forward, x, use_reentrant=False)
        else:
            x = self.transformer(x, mask=causal_mask)  # (batch_size, seq_length, d_model)
        
        # Final layer normalization
        x = self.ln_f(x)  # (batch_size, seq_length, d_model)
        
        # Project to vocabulary size (get logits for each token)
        logits = self.head(x)  # (batch_size, seq_length, vocab_size)
        
        return logits
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal (lower triangular) mask for attention.
        
        This mask ensures that each position can only attend to previous positions,
        which is essential for autoregressive (next-token prediction) models.
        
        Args:
            size: Sequence length
            device: Device to create the mask on
            
        Returns:
            A mask tensor where True means "mask out" (can't attend)
        """
        # Create a lower triangular matrix
        # True = mask out (can't attend), False = can attend
        # Use torch.triu with diagonal=1 to create upper triangular mask
        # This is more memory efficient than creating full matrix first
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
        return mask
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 100, 
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> torch.Tensor:
        """
        Generate new tokens given an input sequence.
        
        This is used for inference (making predictions), not training.
        Uses improved sampling strategies for better text quality.
        
        Args:
            input_ids: Starting sequence of token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Controls randomness (lower = more focused, higher = more random)
            top_k: Only sample from top k most likely tokens (0 = disabled)
            top_p: Nucleus sampling - only sample from tokens with cumulative prob <= top_p (0.0 = disabled)
            repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
            
        Returns:
            Generated sequence including the original input
        """
        self.eval()  # Set to evaluation mode
        
        generated = input_ids.clone()
        
        with torch.no_grad():  # Don't compute gradients during generation
            for _ in range(max_new_tokens):
                # Get logits for the current sequence
                logits = self.forward(generated)  # (batch_size, seq_length, vocab_size)
                
                # Get logits for the last position (next token prediction)
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    # Look at recent tokens (last 50 tokens) to prevent immediate repetition
                    # This is more effective than looking at all unique tokens
                    recent_window = 50
                    recent_tokens = generated[0, -recent_window:] if generated.shape[1] > recent_window else generated[0, :]
                    # Get unique tokens in recent window
                    unique_recent_tokens = torch.unique(recent_tokens)
                    # Apply penalty to recently seen tokens
                    next_token_logits[0, unique_recent_tokens] /= repetition_penalty
                    
                    # Also apply stronger penalty to the very last token to prevent immediate repetition
                    if generated.shape[1] > 0:
                        last_token = generated[0, -1].item()
                        next_token_logits[0, last_token] /= (repetition_penalty * 1.5)  # Extra penalty for immediate repetition
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k sampling: only consider top k tokens
                if top_k > 0:
                    # Get top k values and indices
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)), dim=-1)
                    # Create a new tensor with -inf for non-top-k tokens
                    filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
                    next_token_logits = filtered_logits
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
                
                # Top-p (nucleus) sampling: only sample from tokens that make up top_p probability mass
                if top_p > 0.0 and top_p < 1.0:
                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Find where cumulative probability exceeds top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token (the most likely one)
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    # Create mask for tokens to remove
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    # Set removed tokens' probabilities to 0
                    probs[indices_to_remove] = 0
                    # Renormalize
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample the next token
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for repetitive patterns (same token repeated many times)
                # This helps prevent issues like "!!!!!!!!!!!!!!!!"
                if generated.shape[1] >= 10:  # Only check if we have enough tokens
                    last_10_tokens = generated[0, -10:].tolist()
                    # Check if last 8 tokens are all the same
                    if len(set(last_10_tokens[-8:])) == 1:
                        # Detected repetitive pattern, stop generation
                        break
                
                # Stop if we've reached max_length
                if generated.shape[1] >= self.max_length:
                    break
        
        return generated


def train_epoch(
    model: GPT,
    train_loader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    scaler=None,
    save_dir=None,
    total_batches=0,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, int]:
    """
    Train the model for one epoch with gradient accumulation support.
    
    Args:
        model: The GPT model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        scheduler: Learning rate scheduler
        device: Device to run training on (CPU or GPU)
        epoch: Current epoch number
        scaler: Optional gradient scaler for mixed precision
        save_dir: Optional directory to save checkpoints
        total_batches: Total batch count from previous epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
        
    Returns:
        Tuple of (average training loss, number of batches processed)
    """
    from datetime import datetime
    import torch
    
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        # Move data to device (GPU if available)
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        
        # Clamp token IDs to valid vocabulary range to prevent index errors
        vocab_size = model.vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
        
        # Forward pass: get logits (predictions)
        # Use mixed precision if scaler is provided
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)  # (batch_size, seq_length, vocab_size)
                
                # Reshape for loss calculation
                logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, vocab)
                targets_flat = target_ids.view(-1)  # (batch*seq,)
                
                # Calculate loss (cross-entropy for classification)
                # Scale loss by accumulation steps for correct gradient magnitude
                loss = F.cross_entropy(logits_flat, targets_flat) / gradient_accumulation_steps
        else:
            logits = model(input_ids)  # (batch_size, seq_length, vocab_size)
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, vocab)
            targets_flat = target_ids.view(-1)  # (batch*seq,)
            
            # Calculate loss (cross-entropy for classification)
            # Scale loss by accumulation steps for correct gradient magnitude
            loss = F.cross_entropy(logits_flat, targets_flat) / gradient_accumulation_steps
        
        # Backward pass: accumulate gradients
        if scaler is not None:
            # Mixed precision backward pass
            scaler.scale(loss).backward()
        else:
            loss.backward()  # Accumulate gradients
        
        # Only update weights every gradient_accumulation_steps batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # Update weights
            
            # Step the learning rate scheduler after optimizer step
            scheduler.step()
            
            # Clear gradients after optimizer step
            optimizer.zero_grad(set_to_none=True)
        
        # Track loss (unscaled for reporting)
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Calculate current total batch count (across all epochs)
        current_total_batches = total_batches + num_batches
          
        # Save checkpoint every checkpoint_interval batches
        if save_dir is not None and current_total_batches % training_config.checkpoint_interval == 0:
            timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
            # Create folder for this checkpoint
            checkpoint_folder_name = f"data-{data_config.max_samples}-batch-{current_total_batches}-{timestamp}"
            checkpoint_folder = save_dir / checkpoint_folder_name
            checkpoint_folder.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint file inside the folder
            checkpoint_file = checkpoint_folder / "checkpoint.pt"
            print(f"Saving checkpoint to {checkpoint_folder_name}/checkpoint.pt")
            torch.save({
                'epoch': epoch + 1,
                'total_batches': current_total_batches,
                'batch_in_epoch': num_batches,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'gpt_config': gpt_config,
                'training_config': training_config,
                'data_config': data_config,
                'train_loss': total_loss / num_batches,
            }, checkpoint_file)
            print(f"\n  💾 Saved checkpoint at batch {current_total_batches:,}: {checkpoint_file}")
        
        # Clear CUDA cache periodically to prevent fragmentation
        if device.type == "cuda" and (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()
        
        # Print progress (less frequent for large batches)
        if (batch_idx + 1) % 50 == 0:
            # Get memory stats if on CUDA
            if device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
                try:
                    loader_length = len(train_loader)
                    print(f"  Batch {batch_idx + 1}/{loader_length}, Loss: {loss.item():.4f}, "
                          f"GPU Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
                except TypeError:
                    # IterableDataset doesn't support len()
                    print(f"  Batch {batch_idx + 1}, Loss: {loss.item():.4f}, "
                          f"GPU Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            else:
                try:
                    loader_length = len(train_loader)
                    print(f"  Batch {batch_idx + 1}/{loader_length}, Loss: {loss.item():.4f}")
                except TypeError:
                    # IterableDataset doesn't support len()
                    print(f"  Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, num_batches


def evaluate(model: GPT, val_loader, device: torch.device) -> float:
    """
    Evaluate the model on validation data.
    
    Args:
        model: The GPT model
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        Average validation loss
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # Don't compute gradients during evaluation
        for input_ids, target_ids in val_loader:
            # Move data to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, num_batches


if __name__ == "__main__":
    # Example: Create and test the model
    print("Creating GPT model...")
    
    # Initialize tokenizer to get correct vocab_size
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    
    # Create config with correct vocab_size
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,  # Use tokenizer's vocab_size
    )
    
    # Create model
    model = GPT(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Number of layers: {config.n_layers}")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    logits = model(input_ids)
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")  # Should be (batch_size, seq_length, vocab_size)
    print()
    
    print("Model is ready for training!")
