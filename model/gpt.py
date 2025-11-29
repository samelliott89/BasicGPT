"""
GPT (Generative Pre-trained Transformer) model implementation.

This module implements a GPT-style language model using PyTorch.
GPT is a decoder-only transformer model that predicts the next token
in a sequence given the previous tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.config import GPTConfig
from utils.device import get_best_device


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
        super().__init__()

        # Store configuration
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size

        # Token embedding: converts token IDs to vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Position embedding: encodes position information
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

        # Transformer encoder layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Enable gradient checkpointing to save memory
        self.use_checkpointing = False

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection to vocabulary
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
        """
        batch_size, seq_length = input_ids.shape

        # Clamp input_ids to valid vocabulary range
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)

        # Truncate if needed
        if seq_length > self.max_length:
            input_ids = input_ids[:, : self.max_length]
            seq_length = self.max_length

        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        positions = torch.clamp(positions, 0, self.max_length - 1)

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)

        x = token_embeds + pos_embeds
        x = self.dropout(x)

        # Create causal mask
        causal_mask = self._generate_causal_mask(seq_length, device=input_ids.device)

        # Pass through transformer
        if self.use_checkpointing and self.training:

            def transformer_forward(x):
                return self.transformer(x, mask=causal_mask)

            x = checkpoint(transformer_forward, x, use_reentrant=False)
        else:
            x = self.transformer(x, mask=causal_mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate a causal (lower triangular) mask for attention."""
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """
        Generate new tokens given an input sequence.

        Args:
            input_ids: Starting sequence of token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Controls randomness (lower = more focused)
            top_k: Only sample from top k most likely tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Generated sequence including the original input
        """
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :]

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    recent_window = 50
                    recent_tokens = (
                        generated[0, -recent_window:]
                        if generated.shape[1] > recent_window
                        else generated[0, :]
                    )
                    unique_recent_tokens = torch.unique(recent_tokens)
                    next_token_logits[0, unique_recent_tokens] /= repetition_penalty

                    if generated.shape[1] > 0:
                        last_token = generated[0, -1].item()
                        next_token_logits[0, last_token] /= repetition_penalty * 1.5

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-k sampling
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(
                        next_token_logits, min(top_k, next_token_logits.size(-1)), dim=-1
                    )
                    filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
                    next_token_logits = filtered_logits

                probs = F.softmax(next_token_logits, dim=-1)

                # Top-p sampling
                if top_p > 0.0 and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                # Check for repetitive patterns
                if generated.shape[1] >= 10:
                    last_10_tokens = generated[0, -10:].tolist()
                    if len(set(last_10_tokens[-8:])) == 1:
                        break

                if generated.shape[1] >= self.max_length:
                    break

        return generated

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: torch.device = None):
        """
        Load a pretrained model from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded GPT model
        """
        if device is None:
            device = get_best_device()

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get config from checkpoint or use default
        if "gpt_config" in checkpoint:
            config = checkpoint["gpt_config"]
        else:
            config = GPTConfig()

        # Check vocab size from weights
        if "model_state_dict" in checkpoint:
            token_emb_weight = checkpoint["model_state_dict"].get("token_embedding.weight")
            if token_emb_weight is not None:
                config.vocab_size = token_emb_weight.shape[0]

        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model


if __name__ == "__main__":
    # Test the model
    print("Creating GPT model...")
    config = GPTConfig()
    model = GPT(config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")

    # Test forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    logits = model(input_ids)
    print(f"✓ Forward pass: {input_ids.shape} -> {logits.shape}")
