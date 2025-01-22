from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

from kvax.utils.typing import DeviceArray

PADDING_SEGMENT_ID = -1
H100_NAME = "H100"


@dataclass(frozen=True)
class FlashAttentionParamsConfig:
    query_block_size: int = 64
    kv_block_size: int = 64
    num_warps: int = 4
    num_stages: int = 3


def get_default_flash_attention_params(
    backward: bool,
) -> FlashAttentionParamsConfig:
    devices = jax.devices(backend="gpu")

    if len(devices) == 0:
        raise ValueError("There is must be at least one GPU.")

    # Define parameters based on GPU type
    if H100_NAME in devices[0].device_kind:
        query_block_size = 64 if backward else 128
        params = FlashAttentionParamsConfig(
            query_block_size=query_block_size,
            kv_block_size=128,
            num_warps=8,
            num_stages=3,
        )
    else:
        params = FlashAttentionParamsConfig()

    return params


def pad_to_block_size(
    embeddings: Sequence[DeviceArray] | None,
    positions: DeviceArray | None,
    segment_ids: DeviceArray | None,
    block_size: int,
    pos_fill_value: int,
    transposed_inputs: bool = False,
):
    seq_len = positions.shape[1]
    padded_seq_len = (seq_len + block_size - 1) // block_size * block_size
    pad_len = padded_seq_len - seq_len

    if transposed_inputs:
        embeddings_axis = ((0, 0), (0, 0), (0, pad_len), (0, 0))
    else:
        embeddings_axis = ((0, 0), (0, pad_len), (0, 0), (0, 0))

    if pad_len > 0:
        if embeddings is not None:
            embeddings = [jnp.pad(e, embeddings_axis) for e in embeddings]

        if positions is not None:
            positions = jnp.pad(
                positions, ((0, 0), (0, pad_len)), constant_values=pos_fill_value
            )
        if segment_ids is not None:
            segment_ids = jnp.pad(
                segment_ids, ((0, 0), (0, pad_len)), constant_values=PADDING_SEGMENT_ID
            )

    return embeddings, positions, segment_ids
