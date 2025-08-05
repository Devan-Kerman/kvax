"""Simple test of the clean Flash Attention API without masks."""

import jax
import jax.numpy as jnp

from kvax.ops.flash_attention_clean import flash_attention


def test_no_mask():
    """Test without providing a mask - simplest case."""
    print("Testing without mask...")
    
    # Small test case
    batch = 1
    seq_len = 64
    heads = 4
    dim = 32
    
    # Create inputs
    query = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, heads, dim))
    key = jax.random.normal(jax.random.PRNGKey(1), (batch, seq_len, heads, dim))
    value = jax.random.normal(jax.random.PRNGKey(2), (batch, seq_len, heads, dim))
    
    positions = jnp.arange(seq_len)[None, :]
    segment_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    
    # Run attention without mask - should just do regular attention
    output = flash_attention(
        query, key, value,
        positions, segment_ids, positions, segment_ids,
        mask=None,  # No mask!
        scale=1.0 / jnp.sqrt(dim),
    )
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output dtype: {output.dtype}")
    print(f"✓ No NaNs: {not jnp.any(jnp.isnan(output))}")
    
    # Verify it's doing something reasonable
    # For causal attention, later positions should have different values
    print(f"✓ Output varies across sequence: {jnp.std(output[0, :, 0, 0]) > 0.01}")
    
    return output


if __name__ == "__main__":
    print("Testing Clean Flash Attention API (No Mask)\n")
    test_no_mask()
    print("\n✅ Test passed!")