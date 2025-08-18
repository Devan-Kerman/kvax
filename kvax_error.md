# KVax Flash Attention Mask Error

## Error
```
ValueError: Length of mask_tensors should be equal to two
```

## Context
This error occurs when using the new kvax API (kvax-0.1.0) during the backward pass of flash attention. The error happens in `flash_attention_triton.py` at line 1677 in the `flash_attention_backward` function.

## Root Cause
The attention mask created by `create_attention_mask` doesn't have the correct format for gradient computation. The backward pass expects the mask to have exactly 2 mask tensors, but the created mask likely has a different structure.

## Potential Fixes

### Option 1: Disable Backward Mask Calculation (Quick Fix)
In `model.py`, change `calc_bwd_mask=True` to `calc_bwd_mask=False`:

```python
# In model.py, around line 220
attention_mask = create_attention_mask(
    query_positions=query_positions,
    query_segment_ids=kwargs['query_segment_ids'],
    kv_positions=kv_positions,
    kv_segment_ids=kwargs['kv_segment_ids'],
    calc_bwd_mask=False,  # Changed from True
    fwd_params=fwd_params,
    bwd_params=bwd_params,
)
```

This may disable some gradient computations through the attention mask but could allow training to proceed.

### Option 2: Pass Attention Mask Differently
Instead of passing the full attention mask object, try passing just the mask tensors:

```python
# In soft_attn.py, around line 189
out = flash_attention(
    query=q_reshaped,
    key=k,
    value=v,
    query_positions=query_positions,
    query_segment_ids=query_segment_ids,
    kv_positions=kv_positions,
    kv_segment_ids=kv_segment_ids,
    mask=attention_mask.mask_tensors if hasattr(attention_mask, 'mask_tensors') else attention_mask,
    scale=scale,
)
```

### Option 3: Update KVax Library
This might be a bug in the kvax 0.1.0 release. Check if there's a newer version or report the issue to the kvax maintainers.

## Status
No workaround applied - the issue needs to be fixed in the kvax library itself. The error occurs because the kvax backward pass expects exactly 2 mask tensors but the current API structure doesn't match this expectation.