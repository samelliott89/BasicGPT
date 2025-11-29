"""
Quick script to calculate model parameters for different configurations.
"""

from config import DataConfig, GPTConfig


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
        "token_embedding": token_emb,
        "position_embedding": pos_emb,
        "transformer_layers": transformer_params,
        "output_head": output_head,
        "total": total,
    }


# Target: ~60M parameters
vocab_size = 100256
max_length = 1024

print("Finding configuration for ~60M parameters:")
print("=" * 60)

# Try different configurations
configs = [
    (512, 8, 8),  # d_model, n_layers, n_heads
    (512, 10, 8),
    (384, 10, 6),
    (384, 12, 6),
    (256, 16, 8),
    (256, 20, 8),
]

for d_model, n_layers, n_heads in configs:
    params = estimate_parameters(vocab_size, d_model, n_layers, n_heads, max_length)
    total_m = params["total"] / 1e6
    print(
        f"d_model={d_model:3d}, n_layers={n_layers:2d}, n_heads={n_heads:2d} → {total_m:6.1f}M params"
    )

print("\n" + "=" * 60)
print("For 1.3B tokens:")
print("=" * 60)
print("If avg sequence length = 512 tokens: 1.3B / 512 = ~2.5M samples")
print("If avg sequence length = 1024 tokens: 1.3B / 1024 = ~1.27M samples")
print("Recommended: --max_samples 2000000 (2M samples, ~1B tokens)")

print("\n" + "=" * 60)
print("Tokens per Parameter Analysis:")
print("=" * 60)

# Calculate tokens per parameter for different configurations
# Using config defaults

gpt_config = GPTConfig()
data_config = DataConfig()

# Estimate total tokens in dataset
# Assuming average sequence length (can be adjusted)
avg_seq_length = 512  # Conservative estimate
total_samples = data_config.max_samples or 2_000_000
total_tokens = total_samples * avg_seq_length

print("\nTraining data estimate:")
print(f"  Max samples: {total_samples:,}")
print(f"  Avg sequence length: {avg_seq_length} tokens")
print(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")

print("\nCurrent model config:")
print(f"  d_model: {gpt_config.d_model}")
print(f"  n_layers: {gpt_config.n_layers}")
print(f"  n_heads: {gpt_config.n_heads}")
print(f"  max_length: {gpt_config.max_length}")

# Calculate parameters for current config
current_params = estimate_parameters(
    gpt_config.vocab_size,
    gpt_config.d_model,
    gpt_config.n_layers,
    gpt_config.n_heads,
    gpt_config.max_length,
)
total_params = current_params["total"]
tokens_per_param = total_tokens / total_params

print(f"\n  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

print(f"\nTokens per parameter: {tokens_per_param:.1f}")
print("\nInterpretation:")
if tokens_per_param >= 20:
    print(f"  ✓ Excellent ({tokens_per_param:.1f} tokens/param)")
    print("    Model has plenty of training data for good generalization")
elif tokens_per_param >= 10:
    print(f"  ✓ Good ({tokens_per_param:.1f} tokens/param)")
    print("    Adequate training data for reasonable performance")
elif tokens_per_param >= 5:
    print(f"  ⚠ Moderate ({tokens_per_param:.1f} tokens/param)")
    print("    Consider more data or smaller model for better results")
else:
    print(f"  ⚠ Low ({tokens_per_param:.1f} tokens/param)")
    print("    Risk of underfitting - consider more data or smaller model")

# Calculate for all configs
print("\n" + "=" * 60)
print("Tokens per parameter for all configurations:")
print("=" * 60)
print(f"Using {total_tokens / 1e9:.2f}B tokens, avg seq length {avg_seq_length}")
print()

for d_model, n_layers, n_heads in configs:
    params = estimate_parameters(vocab_size, d_model, n_layers, n_heads, max_length)
    total_p = params["total"]
    tpp = total_tokens / total_p
    total_m = total_p / 1e6
    print(
        f"d_model={d_model:3d}, n_layers={n_layers:2d}, n_heads={n_heads:2d} → "
        f"{total_m:6.1f}M params → {tpp:5.1f} tokens/param"
    )
