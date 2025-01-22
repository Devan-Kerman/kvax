import functools

import chex
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from kvax.utils.common import PADDING_SEGMENT_ID
from kvax.utils.sharding import shard_map
from kvax.utils.specs import Axes, get_attention_specs
from kvax.utils.typing import DeviceArray


def make_segment_mask(
    query_segment_ids: DeviceArray,
    kv_segment_ids: DeviceArray,
) -> DeviceArray:
    chex.assert_equal_rank([query_segment_ids, kv_segment_ids])
    chex.assert_equal_shape_prefix(
        [query_segment_ids, kv_segment_ids],
        query_segment_ids.ndim - 1,
    )
    chex.assert_type([query_segment_ids, kv_segment_ids], jnp.integer)
    return query_segment_ids[..., :, None] == kv_segment_ids[..., None, :]


def _make_flash_attention_cudnn_partition_specs(
    use_mask: bool = False,
) -> tuple[tuple[Axes, ...], tuple[Axes, ...]]:
    specs = get_attention_specs()
    kv_specs = specs.kv_specs
    query_specs = specs.query_specs

    mask_axes = (
        (
            query_specs[Axes.batch],
            None,
            query_specs[Axes.query_sequence],
            kv_specs[Axes.kv_sequence],
        )
        if use_mask
        else None
    )

    in_specs = (
        query_specs,  # query
        kv_specs,  # key
        kv_specs,  # value
        mask_axes,  # mask
    )
    out_specs = query_specs  # out
    return in_specs, out_specs


def create_cudnn_attn_mask(
    query_segment_ids: DeviceArray,
    kv_segment_ids: DeviceArray,
) -> DeviceArray:
    batch_size, query_seq_len = query_segment_ids.shape
    batch_size, kv_seq_len = query_segment_ids.shape
    mask = make_segment_mask(query_segment_ids, kv_segment_ids).reshape(
        batch_size, 1, query_seq_len, kv_seq_len
    )
    return mask.astype(jnp.bool)


def flash_attention_cudnn(
    query: DeviceArray,
    key: DeviceArray,
    value: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_segment_ids: DeviceArray,
    mask: DeviceArray | None = None,
    scale: float = 1.0,
    is_causal: bool = False,
    skip_pad_tokens: bool = True,
    mesh: Mesh | None = None,
) -> DeviceArray:
    def _flash_attn_cudnn(
        query: DeviceArray,
        key: DeviceArray,
        value: DeviceArray,
        mask: DeviceArray | None,
        *args: object,
        **kwargs: object,
    ):
        if mask is not None:
            num_heads = query.shape[2]
            mask = jnp.repeat(mask, repeats=num_heads, axis=1)

        result = jax.nn.dot_product_attention(
            query,
            key,
            value,
            *args,
            **kwargs,
            mask=mask,
        )
        return result

    if skip_pad_tokens:
        query_seq_lengths = jnp.sum(query_segment_ids != PADDING_SEGMENT_ID, axis=-1)
        kv_seq_lengths = jnp.sum(kv_segment_ids != PADDING_SEGMENT_ID, axis=-1)
    else:
        query_seq_lengths = None
        kv_seq_lengths = None

    flash_attention_fn = functools.partial(
        _flash_attn_cudnn,
        is_causal=is_causal,
        scale=scale,
        implementation="cudnn",
        query_seq_lengths=query_seq_lengths,
        key_value_seq_lengths=kv_seq_lengths,
    )

    in_specs, out_specs = _make_flash_attention_cudnn_partition_specs(
        use_mask=mask is not None,
    )
    flash_attention_fn = shard_map(
        flash_attention_fn,
        in_specs=in_specs,
        out_specs=out_specs,
        mesh=mesh,
    )
    result = flash_attention_fn(
        query,
        key,
        value,
        mask,
    )
    return result
