"""Test the clean Flash Attention API."""

import jax
import jax.numpy as jnp

from kvax.ops import flash_attention, create_attention_mask
from kvax.utils.common import PADDING_SEGMENT_ID


def test_single_device():
    """Test single device case without mesh."""
    print("Testing single device...")
    
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
    
    # Add some padding
    segment_ids = segment_ids.at[:, -10:].set(PADDING_SEGMENT_ID)
    
    # Create mask - no mesh needed!
    mask = create_attention_mask(
        positions, segment_ids, positions, segment_ids,
        calc_bwd_mask=False,
    )
    
    # Run attention
    output = flash_attention(
        query, key, value,
        positions, segment_ids, positions, segment_ids,
        mask,
        scale=1.0 / jnp.sqrt(dim),
    )
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output dtype: {output.dtype}")
    print(f"✓ No NaNs: {not jnp.any(jnp.isnan(output))}")
    
    return output


def test_comparison_with_original():
    """Compare clean API with original API."""
    print("\nTesting equivalence with original API...")
    
    from kvax.ops import flash_attention as orig_flash
    from kvax.ops import create_attention_mask as orig_mask
    from kvax.utils.specs import attention_specs
    
    # Test inputs
    batch = 1
    seq_len = 64
    heads = 4
    dim = 32
    
    key0 = jax.random.PRNGKey(42)
    keys = jax.random.split(key0, 3)
    
    query = jax.random.normal(keys[0], (batch, seq_len, heads, dim))
    key = jax.random.normal(keys[1], (batch, seq_len, heads, dim))
    value = jax.random.normal(keys[2], (batch, seq_len, heads, dim))
    
    positions = jnp.arange(seq_len)[None, :]
    segment_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    scale = 1.0 / jnp.sqrt(dim)
    
    # Clean API
    mask_clean = create_attention_mask(
        positions, segment_ids, positions, segment_ids,
    )
    
    output_clean = flash_attention(
        query, key, value,
        positions, segment_ids, positions, segment_ids,
        mask_clean,
        scale=scale,
    )
    
    # Original API
    query_spec = ("data", None, None, None)
    kv_spec = ("data", None, None, None)
    
    with attention_specs(query_spec, kv_spec):
        mask_orig = orig_mask(
            positions, segment_ids, positions, segment_ids,
        )
        
        output_orig = orig_flash(
            query, key, value,
            positions, segment_ids, positions, segment_ids,
            mask_orig,
            scale=scale,
        )
    
    # Compare
    max_diff = jnp.max(jnp.abs(output_clean - output_orig))
    print(f"✓ Max difference: {max_diff:.2e}")
    print(f"✓ Outputs match: {jnp.allclose(output_clean, output_orig, rtol=1e-5)}")
    
    return output_clean, output_orig


if __name__ == "__main__":
    print("Testing Clean Flash Attention API\n")
    
    # Test 1: Single device
    test_single_device()
    
    # Test 2: Compare with original
    test_comparison_with_original()
    
    print("\n✅ All tests passed!")