import chex
import jax.numpy as jnp
from jax.sharding import Mesh

from kvax.utils.sharding import get_query_context_mesh_axis_size
from kvax.utils.typing import DeviceArray


def generate_shard_indices(num_shards: int, reverse: bool = False) -> jnp.ndarray:
    # Create indexes like: ((0, num_shards-1), (1, num_shards-2), ...)
    # This is needed to make a balanced load on all GPUs in attention
    indices = jnp.array(
        [(i, num_shards - i - 1) for i in range(num_shards // 2)]
    ).flatten()
    return jnp.argsort(indices) if reverse else indices


def apply_permutation(
    inputs: tuple[DeviceArray, ...],
    num_shards: int,
    reverse_permutation: bool,
) -> tuple[DeviceArray, ...]:
    batch_size = inputs[0].shape[0]
    shard_indices = generate_shard_indices(num_shards, reverse=reverse_permutation)[
        None, :
    ]

    outputs = []
    for input in inputs:
        idx_shape = (1, num_shards) + (1,) * len(input.shape[1:])
        idxs_exp = jnp.reshape(shard_indices, idx_shape)
        outputs.append(
            jnp.take_along_axis(
                jnp.reshape(
                    input,
                    (batch_size, num_shards, -1, *input.shape[2:]),
                ),
                idxs_exp,
                axis=1,
            ).reshape(input.shape)
        )
    return outputs


def permute_tokens_context_parallelism(
    inputs: DeviceArray | tuple[DeviceArray, ...],
    mesh: Mesh | None = None,
) -> DeviceArray | tuple[DeviceArray, ...]:
    """
    A function to permute tokens across the sequence length `(axis==1)` to balance
    computation of the attention operation between GPUs for the causal mask case.

    Args:
        inputs (DeviceArray | tuple[DeviceArray, ...]): An input tensor or tuple of
            tensors to permute.
        mesh (Mesh | None, optional): Device mesh configuration for distributed
            execution. If set to `None`, it uses the mesh from the global context.
            An exception is raised if `None` is provided and no mesh is available
            from the global context. The default is `None`.
    Returns:
        DeviceArray | tuple[DeviceArray, ...]: Permuted tensor or tuple of tensors.
    """
    context_axis_size = get_query_context_mesh_axis_size(mesh)
    num_shards = 2 * context_axis_size

    is_single_input = isinstance(inputs, DeviceArray)
    if is_single_input:
        inputs = (inputs,)

    chex.assert_equal_shape_prefix(inputs, 2)

    # Permute data
    outputs = apply_permutation(
        inputs,
        num_shards=num_shards,
        reverse_permutation=False,
    )
    return outputs[0] if is_single_input else outputs


def unpermute_tokens_context_parallelism(
    inputs: DeviceArray | tuple[DeviceArray, ...],
    mesh: Mesh | None = None,
) -> DeviceArray | tuple[DeviceArray, ...]:
    """
    A function to unpermute tokens across the sequence length `(axis==1)` after
    the `permute_tokens_context_parallelism` function to return them to their
    original order.

    Args:
        inputs (DeviceArray | tuple[DeviceArray, ...]): An input tensor or tuple of
            tensors to unpermute.
        mesh (Mesh | None, optional): Device mesh configuration for distributed
            execution. If set to `None`, it uses the mesh from the global context.
            An exception is raised if `None` is provided and no mesh is available
            from the global context. The default is `None`.
    Returns:
        DeviceArray | tuple[DeviceArray, ...]: A tensor or tuple of tensors with tokens
            in their original order.
    """
    context_axis_size = get_query_context_mesh_axis_size(mesh)
    num_shards = 2 * context_axis_size

    is_single_input = isinstance(inputs, DeviceArray)
    if is_single_input:
        inputs = (inputs,)

    chex.assert_equal_shape_prefix(inputs, 2)

    # Unpermute data
    outputs = apply_permutation(
        inputs,
        num_shards=num_shards,
        reverse_permutation=True,
    )
    return outputs[0] if is_single_input else outputs
