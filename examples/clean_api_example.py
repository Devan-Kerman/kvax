"""
Example of using the clean Flash Attention API without context managers.

This shows how much simpler the API is compared to the original version.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from kvax.ops.flash_attention_clean import flash_attention, create_attention_mask
from kvax.utils.common import PADDING_SEGMENT_ID

# Example 1: Single device usage (no mesh needed)
def single_device_example():
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    
    # Create dummy inputs
    query = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Mark last 10 tokens as padding
    segment_ids = segment_ids.at[:, -10:].set(PADDING_SEGMENT_ID)
    
    # Create mask - no mesh or specs needed!
    mask = create_attention_mask(
        positions, segment_ids, positions, segment_ids
    )
    
    # Run attention - no context managers!
    output = flash_attention(
        query, key, value,
        positions, segment_ids, positions, segment_ids,
        mask,
        scale=1.0 / jnp.sqrt(head_dim),
    )
    
    print(f"Single device output shape: {output.shape}")
    return output


# Example 2: Multi-device with sharding
def multi_device_example():
    # Assume we have 2 devices
    devices = jax.devices()[:2]
    mesh = Mesh(devices, axis_names=('data',))
    
    batch_size = 4  # Will be split across devices
    seq_len = 256
    num_heads = 16
    head_dim = 64
    
    # Create inputs
    query = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Define sharding - just like shard_map!
    query_spec = ('data', None, None, None)  # Shard on batch dimension
    kv_spec = ('data', None, None, None)
    
    # Create mask
    mask = create_attention_mask(
        positions, segment_ids, positions, segment_ids,
        mesh=mesh,
        query_spec=query_spec,
        kv_spec=kv_spec,
    )
    
    # Run attention - clean and simple!
    output = flash_attention(
        query, key, value,
        positions, segment_ids, positions, segment_ids,
        mask,
        mesh=mesh,
        query_spec=query_spec,
        kv_spec=kv_spec,
        scale=1.0 / jnp.sqrt(head_dim),
    )
    
    print(f"Multi-device output shape: {output.shape}")
    return output


# Example 3: Context parallelism
def context_parallel_example():
    # Assume we have 4 devices in a 2x2 grid
    devices = jax.devices()[:4]
    mesh = Mesh(devices.reshape(2, 2), axis_names=('data', 'sequence'))
    
    batch_size = 2
    seq_len = 1024  # Will be split across sequence dimension
    num_heads = 16
    head_dim = 64
    
    # Create inputs
    query = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Sharding for context parallelism
    query_spec = ('data', 'sequence', None, None)  # Shard on batch AND sequence
    kv_spec = ('data', None, None, None)  # KV only sharded on batch
    
    # Create mask
    mask = create_attention_mask(
        positions, segment_ids, positions, segment_ids,
        mesh=mesh,
        query_spec=query_spec,
        kv_spec=kv_spec,
    )
    
    # Run attention with context parallelism
    output = flash_attention(
        query, key, value,
        positions, segment_ids, positions, segment_ids,
        mask,
        mesh=mesh,
        query_spec=query_spec,
        kv_spec=kv_spec,
        scale=1.0 / jnp.sqrt(head_dim),
    )
    
    print(f"Context parallel output shape: {output.shape}")
    return output


# Compare with the old API (for reference)
def old_api_example():
    """This is how you had to do it before - much more verbose!"""
    from kvax.ops import flash_attention as old_flash_attention
    from kvax.ops import create_attention_mask as old_create_mask
    from kvax.utils.specs import attention_specs
    
    # Setup
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    
    query = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Old way - needs context managers and mesh context
    devices = jax.devices()[:1]
    mesh = Mesh(devices, axis_names=('data',))
    
    query_spec = ('data', None, None, None)
    kv_spec = ('data', None, None, None)
    
    # Look at all this nesting!
    with mesh:
        with attention_specs(query_spec, kv_spec):
            mask = old_create_mask(
                positions, segment_ids, positions, segment_ids,
            )
            
            output = old_flash_attention(
                query, key, value,
                positions, segment_ids, positions, segment_ids,
                mask,
                scale=1.0 / jnp.sqrt(head_dim),
            )
    
    print(f"Old API output shape: {output.shape}")
    return output


if __name__ == "__main__":
    print("Clean Flash Attention API Examples\n")
    
    print("1. Single device (no mesh):")
    single_device_example()
    
    print("\n2. Multi-device with sharding:")
    if len(jax.devices()) >= 2:
        multi_device_example()
    else:
        print("   (Skipped - need at least 2 devices)")
    
    print("\n3. Context parallelism:")
    if len(jax.devices()) >= 4:
        context_parallel_example()
    else:
        print("   (Skipped - need at least 4 devices)")
    
    print("\n4. Old API for comparison:")
    old_api_example()