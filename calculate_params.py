"""
Quick script to calculate model parameters for different configurations.
"""

def estimate_parameters(vocab_size, d_model, n_layers, n_heads, max_length):
    """Estimate total model parameters."""
    # Token embedding
    token_emb = vocab_size * d_model
    
    # Position embedding
    pos_emb = max_length * d_model
    
    # Transformer layers
    # Each layer has:
    # - Self-attention: 4 * d_model^2 (Q, K, V, output projections)
    # - Feedforward: 2 * (d_model * 4*d_model) = 8 * d_model^2
    # - Layer norms: 2 * d_model (negligible)
    params_per_layer = 4 * d_model * d_model + 8 * d_model * d_model  # Simplified
    transformer_params = n_layers * params_per_layer
    
    # Output head
    output_head = d_model * vocab_size
    
    total = token_emb + pos_emb + transformer_params + output_head
    
    return {
        'token_embedding': token_emb,
        'position_embedding': pos_emb,
        'transformer_layers': transformer_params,
        'output_head': output_head,
        'total': total
    }


# Target: ~60M parameters
vocab_size = 100256
max_length = 1024

print("Finding configuration for ~60M parameters:")
print("=" * 60)

# Try different configurations
configs = [
    (512, 8, 8),   # d_model, n_layers, n_heads
    (512, 10, 8),
    (384, 10, 6),
    (384, 12, 6),
    (256, 16, 8),
    (256, 20, 8),
]

for d_model, n_layers, n_heads in configs:
    params = estimate_parameters(vocab_size, d_model, n_layers, n_heads, max_length)
    total_m = params['total'] / 1e6
    print(f"d_model={d_model:3d}, n_layers={n_layers:2d}, n_heads={n_heads:2d} → {total_m:6.1f}M params")

print("\n" + "=" * 60)
print("For 1.3B tokens:")
print("=" * 60)
print("If avg sequence length = 512 tokens: 1.3B / 512 = ~2.5M samples")
print("If avg sequence length = 1024 tokens: 1.3B / 1024 = ~1.27M samples")
print("Recommended: --max_samples 2000000 (2M samples, ~1B tokens)")

