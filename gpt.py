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
from dataclasses import dataclass
from tokenizer import Tokenizer


@dataclass
class GPTConfig:
    """
    Configuration class for the GPT model.
    
    This holds all the hyperparameters (settings) for the model.
    Using a dataclass makes it easy to pass around configuration.
    """
    vocab_size: int = 100256  # Vocabulary size from tiktoken cl100k_base
    d_model: int = 512  # Dimension of the model (embedding size)
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 6  # Number of transformer layers
    max_length: int = 1024  # Maximum sequence length (context window)
    dropout: float = 0.1  # Dropout rate for regularization
    learning_rate: float = 3e-4  # Learning rate for optimizer
    epochs: int = 1  # Number of training epochs


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
            dim_feedforward=config.d_model * 4,  # Feedforward dimension
            dropout=config.dropout,
            activation='gelu',  # GELU activation (common in GPT)
            batch_first=True  # Input shape: (batch, seq, features)
        )
        
        # Stack multiple transformer layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
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
        
        # Create position indices: [0, 1, 2, ..., seq_length-1]
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_length)
        
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
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate new tokens given an input sequence.
        
        This is used for inference (making predictions), not training.
        
        Args:
            input_ids: Starting sequence of token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Controls randomness (higher = more random)
            
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
                next_token_logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
                
                # Sample the next token
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we've reached max_length
                if generated.shape[1] >= self.max_length:
                    break
        
        return generated


def train_epoch(
    model: GPT,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The GPT model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        device: Device to run training on (CPU or GPU)
        epoch: Current epoch number
        
    Returns:
        Average training loss for this epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        # Move data to device (GPU if available)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass: get logits (predictions)
        logits = model(input_ids)  # (batch_size, seq_length, vocab_size)
        
        # Reshape for loss calculation
        # CrossEntropyLoss expects (batch*seq, vocab) and (batch*seq,)
        logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, vocab)
        targets_flat = target_ids.view(-1)  # (batch*seq,)
        
        # Calculate loss (cross-entropy for classification)
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


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
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


if __name__ == "__main__":
    # Example: Create and test the model
    print("Creating GPT model...")
    
    # Initialize tokenizer to get correct vocab_size
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    
    # Create config with correct vocab_size
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,  # Use tokenizer's vocab_size
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_length=1024
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
