"""
Clean Flash Attention API without context managers.

This module provides a simpler API for flash attention that:
1. Takes mesh and sharding specs directly (no context managers)
2. Handles None mesh gracefully for single device
3. Works like shard_map - just pass what you need
"""

import functools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from kvax.ops.flash_attention_triton import (
    flash_attention_triton_single_device,
    _make_flash_attention_partition_specs,
)
from kvax.utils.common import (
    FlashAttentionParamsConfig,
    get_default_flash_attention_params,
)
from kvax.utils.permutation import unpermute_tokens_context_parallelism
from kvax.utils.typing import AttentionMask, DeviceArray


def flash_attention(
    query: DeviceArray,
    key: DeviceArray,
    value: DeviceArray,
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    mask: Optional[tuple[AttentionMask, ...]] = None,
    *,
    mesh: Optional[Mesh] = None,
    query_spec: Optional[Sequence[str]] = None,
    kv_spec: Optional[Sequence[str]] = None,
    scale: float = 1.0,
    fwd_params: Optional[FlashAttentionParamsConfig] = None,
    bwd_params: Optional[FlashAttentionParamsConfig] = None,
    assume_sequential_positions: bool = False,
    permute_tokens_for_load_balance: bool = True,
    debug: bool = False,
) -> DeviceArray:
    """
    Flash attention with a cleaner API - no context managers required.
    
    Args:
        query: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        key: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        value: Value tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        query_positions: Query positions (batch, seq_len)
        query_segment_ids: Query segment IDs (batch, seq_len)
        kv_positions: KV positions (batch, seq_len)
        kv_segment_ids: KV segment IDs (batch, seq_len)
        mask: Attention mask tuple from create_attention_mask
        mesh: Optional device mesh. If None, runs on single device
        query_spec: Query sharding spec e.g. ('data', 'sequence', 'tensor', None)
        kv_spec: KV sharding spec e.g. ('data', None, 'tensor', None)
        scale: Attention scale factor
        fwd_params: Forward pass parameters
        bwd_params: Backward pass parameters
        assume_sequential_positions: Whether positions are sequential
        permute_tokens_for_load_balance: Whether to permute for load balance
        debug: Debug mode
        
    Returns:
        Attention output tensor
    """
    # Default parameters
    if fwd_params is None:
        fwd_params = get_default_flash_attention_params(backward=False)
    if bwd_params is None:
        bwd_params = get_default_flash_attention_params(backward=True)
    
    # Single device case - no sharding needed
    if mesh is None:
        # Transform tensors for optimized processing
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 1, 3))
        value = value.transpose((0, 2, 1, 3))
        
        # Flatten mask if provided, or create a simple one
        if mask is not None:
            # mask is a tuple of AttentionMask objects, need to flatten each
            mask_tensors = []
            for m in mask:
                if hasattr(m, 'flatten'):
                    mask_tensors.extend(m.flatten())
                else:
                    # Already flattened
                    mask_tensors.append(m)
            mask_tensors = tuple(mask_tensors)
        else:
            # Create a simple mask when none provided
            # This creates a basic causal mask
            batch_size, num_heads, seq_len, _ = query.shape
            # Simple mask that allows all attention (triton will handle causal internally)
            lower = jnp.zeros((batch_size, 1, 1), dtype=jnp.int32)
            upper = jnp.ones((batch_size, 1, 1), dtype=jnp.int32) * ((seq_len + 63) // 64)  # num blocks
            mask_tensors = (lower, upper, lower, upper)
        
        result = flash_attention_triton_single_device(
            query=query,
            key=key,
            value=value,
            query_positions=query_positions,
            query_segment_ids=query_segment_ids,
            kv_positions=kv_positions,
            kv_segment_ids=kv_segment_ids,
            mask_tensors=mask_tensors,
            scale=np.float32(scale),  # Ensure float32
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            assume_sequential_positions=assume_sequential_positions,
            memory_optimized_gqa_backward=False,
            permute_tokens_for_load_balance=False,  # No permutation for single device
            context_parallelism_mesh_axis_name=None,  # No context parallelism
            debug=debug,
        )
        
        # Transform back
        return result.transpose((0, 2, 1, 3))
    
    # Multi-device case with sharding
    if query_spec is None or kv_spec is None:
        raise ValueError("query_spec and kv_spec must be provided when using mesh")
    
    # Validate specs have correct length
    if len(query_spec) != 4:
        raise ValueError(f"query_spec must have 4 dimensions, got {len(query_spec)}")
    if len(kv_spec) != 4:
        raise ValueError(f"kv_spec must have 4 dimensions, got {len(kv_spec)}")
    
    # Check for context parallelism
    context_axis = query_spec[1]  # sequence dimension
    is_context_parallel = context_axis is not None
    
    # Build flash attention function
    flash_attention_fn = functools.partial(
        flash_attention_triton_single_device,
        scale=scale,
        bias=None,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        assume_sequential_positions=assume_sequential_positions,
        memory_optimized_gqa_backward=False,
        permute_tokens_for_load_balance=permute_tokens_for_load_balance and is_context_parallel,
        context_parallelism_mesh_axis_name=context_axis if is_context_parallel else None,
        debug=debug,
    )
    
    # Create sharding specs - need to handle transposed layout
    # flash_attention_triton expects (batch, heads, seq, dim)
    transposed_query_spec = (query_spec[0], query_spec[2], query_spec[1], query_spec[3])
    transposed_kv_spec = (kv_spec[0], kv_spec[2], kv_spec[1], kv_spec[3])
    
    # Build partition specs for all inputs
    in_specs = (
        PartitionSpec(*transposed_query_spec),  # query
        PartitionSpec(*transposed_kv_spec),     # key  
        PartitionSpec(*transposed_kv_spec),     # value
        PartitionSpec(query_spec[0], query_spec[1]),  # query_positions
        PartitionSpec(query_spec[0], query_spec[1]),  # query_segment_ids
        PartitionSpec(kv_spec[0], kv_spec[1]),        # kv_positions
        PartitionSpec(kv_spec[0], kv_spec[1]),        # kv_segment_ids
    )
    
    # Add mask specs if mask is provided
    if mask is not None:
        for _ in mask:
            in_specs += (PartitionSpec(query_spec[0]),)  # Masks are batch-only
    
    out_spec = PartitionSpec(*transposed_query_spec)
    
    # Apply shard_map
    flash_attention_fn = jax.experimental.shard_map.shard_map(
        flash_attention_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_spec,
        check_rep=False,
    )
    
    # Handle key/value unpermutation for context parallelism
    if is_context_parallel and permute_tokens_for_load_balance:
        # Need to implement simplified unpermute without context manager
        # For now, we'll skip this - can be added later
        pass
    
    # Transform to expected layout
    query = query.transpose((0, 2, 1, 3))
    key = key.transpose((0, 2, 1, 3))
    value = value.transpose((0, 2, 1, 3))
    
    # Prepare mask tensors
    if mask is not None:
        # Flatten mask objects if needed  
        mask_tensors = []
        for m in mask:
            if hasattr(m, 'flatten'):
                mask_tensors.extend(m.flatten())
            else:
                mask_tensors.append(m)
        mask_args = mask_tensors
    else:
        mask_args = []
    
    # Run attention
    output = flash_attention_fn(
        query,
        key,
        value,
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
        *mask_args,
    )
    
    # Transform back
    return output.transpose((0, 2, 1, 3))


def create_attention_mask(
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    *,
    mesh: Optional[Mesh] = None,
    query_spec: Optional[Sequence[str]] = None,
    kv_spec: Optional[Sequence[str]] = None,
    fwd_params: Optional[FlashAttentionParamsConfig] = None,
    bwd_params: Optional[FlashAttentionParamsConfig] = None,
    calc_bwd_mask: bool = False,
    skip_pad_tokens: bool = True,
) -> tuple[AttentionMask, ...]:
    """
    Create attention mask without context managers.
    
    For now, this is a wrapper around the original implementation.
    TODO: Implement clean version without context managers.
    """
    from kvax.ops.mask_creator import create_attention_mask as _create_mask
    from kvax.utils.specs import attention_specs
    
    # Single device case - use simple implementation
    if mesh is None:
        # For single device, we can use a dummy mesh with one axis
        import jax
        dummy_mesh = Mesh(jax.devices()[:1], axis_names=('dummy',))
        query_spec = (None, None, None, None)
        kv_spec = (None, None, None, None)
        
        # Use context manager with dummy specs
        with dummy_mesh:
            with attention_specs(query_spec, kv_spec):
                return _create_mask(
                    query_positions=query_positions,
                    query_segment_ids=query_segment_ids,
                    kv_positions=kv_positions,
                    kv_segment_ids=kv_segment_ids,
                    fwd_params=fwd_params,
                    bwd_params=bwd_params,
                    calc_bwd_mask=calc_bwd_mask,
                    skip_pad_tokens=skip_pad_tokens,
                    mesh=dummy_mesh,
                )
    
    # Multi-device case
    with attention_specs(query_spec, kv_spec):
        return _create_mask(
            query_positions=query_positions,
            query_segment_ids=query_segment_ids,
            kv_positions=kv_positions,
            kv_segment_ids=kv_segment_ids,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            calc_bwd_mask=calc_bwd_mask,
            skip_pad_tokens=skip_pad_tokens,
            mesh=mesh,
        )