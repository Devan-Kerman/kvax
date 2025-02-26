import functools
import logging
import math
import random
import timeit
from typing import Callable, Literal

import chex
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PositionalSharding

from kvax.ops.flash_attention_cudnn import create_cudnn_attn_mask, flash_attention_cudnn
from kvax.ops.flash_attention_triton import flash_attention_triton
from kvax.ops.mask_creator import (
    calculate_mask_sparsity,
    create_attention_mask,
    print_mask,
)
from kvax.utils.common import PADDING_SEGMENT_ID, get_default_flash_attention_params
from kvax.utils.permutation import (
    permute_tokens_context_parallelism,
    unpermute_tokens_context_parallelism,
)
from kvax.utils.specs import attention_specs
from kvax.utils.typing import DeviceArray, PRNGKey, Specs

logger = logging.getLogger(__name__)

AttentionInputsDict = tuple[dict[str, tuple[DeviceArray, ...]], Mesh, Specs, Specs]


def make_random_attention_inputs_with_sharding(
    batch_size: int,
    query_seq_len: int,
    kv_seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    rng: PRNGKey,
    num_pad_tokens: int = 0,
    shard_kv: bool = True,
    random_num_pad_tokens_in_batch: bool = False,
    start_query_position: int | tuple[int] = 0,
    parallelism_type: Literal["context", "tensor"] = "tensor",
    list_of_segment_lengths: list[int] | None = None,
) -> AttentionInputsDict:
    is_inference_mode = query_seq_len == 1
    if is_inference_mode and random_num_pad_tokens_in_batch:
        raise ValueError(
            "Inference mode and random_num_pad_tokens_in_batch flag are not supported."
        )

    q_shape = batch_size, query_seq_len, num_query_heads, qk_head_dim
    k_shape = batch_size, kv_seq_len, num_kv_heads, qk_head_dim
    v_shape = batch_size, kv_seq_len, num_kv_heads, value_head_dim
    rng_q, rng_k, rng_v = jax.random.split(rng, 3)
    query = jax.random.normal(rng_q, q_shape, dtype=jnp.bfloat16)
    key = jax.random.normal(rng_k, k_shape, dtype=jnp.bfloat16)
    value = jax.random.normal(rng_v, v_shape, dtype=jnp.bfloat16)
    scale = 1.0 / math.sqrt(qk_head_dim)

    q_pos = jnp.broadcast_to(jnp.arange(query_seq_len), (batch_size, query_seq_len))
    kv_pos = jnp.broadcast_to(jnp.arange(kv_seq_len), (batch_size, kv_seq_len))

    if isinstance(start_query_position, tuple):
        start_query_position = jnp.array(start_query_position, dtype=jnp.int32).reshape(
            -1, 1
        )
        chex.assert_shape(start_query_position, (batch_size, 1))
    else:
        start_query_position = jnp.broadcast_to(
            jnp.array(start_query_position, dtype=jnp.int32), (batch_size, 1)
        )

    q_pos += start_query_position

    q_segment_ids = jnp.zeros((batch_size, query_seq_len), dtype=jnp.int32)
    kv_segment_ids = jnp.zeros((batch_size, kv_seq_len), dtype=jnp.int32)

    # In the inference mode we ignore num_pad_tokens value.
    # All tokens with posinion > query_position + 1 are pad tokens.
    if is_inference_mode:
        num_pad_tokens = kv_seq_len - start_query_position[:, 0] - 1
    else:
        num_pad_tokens = jnp.broadcast_to(
            jnp.array(num_pad_tokens, dtype=jnp.int32), (batch_size,)
        )

    if list_of_segment_lengths:
        # Handle the case where the last segment is -1
        if list_of_segment_lengths[-1] == -1:
            last_segment_length = query_seq_len - np.sum(list_of_segment_lengths[:-1])
            if last_segment_length < 0:
                raise ValueError(
                    "Sum of all sizes in list_of_segment_lengths exceed seq_len."
                )
            list_of_segment_lengths[-1] = last_segment_length

        # Prepend 0 to simplify cumsum logic
        list_of_segment_lengths = [0] + list_of_segment_lengths
        cumulative_lengths = np.cumsum(list_of_segment_lengths)
        total_length = cumulative_lengths[-1]

        # Validate that the total length does not exceed seq_lens
        if total_length > kv_seq_len or total_length > query_seq_len:
            raise ValueError(
                "Sum of all sizes in list_of_segment_lengths exceed seq_len."
            )

        for i, (start, end) in enumerate(
            zip(cumulative_lengths[:-1], cumulative_lengths[1:])
        ):
            q_segment_ids = q_segment_ids.at[:, start:end].set(i)
            kv_segment_ids = kv_segment_ids.at[:, start:end].set(i)

    if random_num_pad_tokens_in_batch:
        max_num_pad_tokens = num_pad_tokens
        num_pad_tokens = jax.random.randint(rng, (batch_size,), 0, max_num_pad_tokens)

        # Make the number of pad tokens largest in the first batch sample because
        # this leads to errors in attention op.
        max_index = jnp.argmax(num_pad_tokens, axis=0)
        num_pad_tokens_copy = num_pad_tokens.copy()
        num_pad_tokens_copy = num_pad_tokens_copy.at[0].set(num_pad_tokens[max_index])
        num_pad_tokens = num_pad_tokens_copy.at[max_index].set(num_pad_tokens[0])

        # Make the max distance between min and max number of pad tokens 256.
        # This need to make sure they get into different blocks in FA.
        if (max_num_pad_tokens < 256).any():
            raise ValueError(
                "Maximum number of pad tokens should be greater than 256 if "
                "the flag random_num_pad_tokens_in_batch is True"
            )

        max_index = jnp.argmax(num_pad_tokens, axis=0)
        min_index = jnp.argmin(num_pad_tokens, axis=0)
        if num_pad_tokens[max_index] - num_pad_tokens[min_index] < 256:
            num_pad_tokens = num_pad_tokens.at[min_index].set(
                jnp.maximum(num_pad_tokens[min_index] - 128, 0)
            )
            diff = num_pad_tokens[max_index] - num_pad_tokens[min_index]
            num_pad_tokens = num_pad_tokens.at[max_index].set(
                jnp.minimum(num_pad_tokens[max_index] + 256 - diff, kv_seq_len - 1)
            )

        if query_seq_len > 1:
            query_mask = (
                jnp.arange(query_seq_len) >= query_seq_len - num_pad_tokens[:, None]
            )
            q_segment_ids = q_segment_ids.at[query_mask].set(PADDING_SEGMENT_ID)
        kv_mask = jnp.arange(kv_seq_len) >= kv_seq_len - num_pad_tokens[:, None]
        kv_segment_ids = kv_segment_ids.at[kv_mask].set(PADDING_SEGMENT_ID)
    else:
        for i in range(batch_size):
            if query_seq_len > 1:
                q_segment_ids = q_segment_ids.at[
                    :, query_seq_len - num_pad_tokens[i] :
                ].set(PADDING_SEGMENT_ID)
            kv_segment_ids = kv_segment_ids.at[i, kv_seq_len - num_pad_tokens[i] :].set(
                PADDING_SEGMENT_ID
            )

    logger.info(f"Query shape: {q_shape}")
    logger.info(f"Key shape: {k_shape}")
    logger.info(f"Value shape: {v_shape}")

    inputs_dict = {
        "data": (query, key, value, q_pos, q_segment_ids, kv_pos, kv_segment_ids),
        "scale": scale,
    }

    # Shard query, key and value across available devices
    inputs_dict_sharded, mesh, query_specs, kv_specs = shard_input_data(
        inputs_dict,
        parallelism_type=parallelism_type,
        shard_kv=shard_kv,
    )

    return inputs_dict_sharded, mesh, query_specs, kv_specs


def shard_input_data(
    inputs: dict[str, tuple[DeviceArray, ...]],
    parallelism_type: Literal["context", "tensor"] = "tensor",
    shard_kv: bool = True,
) -> AttentionInputsDict:
    (query, key, value, q_pos, q_sids, kv_pos, kv_sids) = inputs["data"]
    # Shard query, key and value across available devices
    num_devices = len(jax.devices())
    if parallelism_type == "tensor":
        # input axes: (batch, sequence, heads, head_dim)
        # Sharding tensors across heads
        device_mesh = mesh_utils.create_device_mesh(mesh_shape=(1, 1, num_devices, 1))
        mesh = Mesh(device_mesh.reshape(1, num_devices), axis_names=("data", "model"))
        query_specs = ("data", None, "model", None)
    elif parallelism_type == "context":
        # input axes: (batch, sequence, heads, head_dim)
        # Sharding tensors across sequence
        device_mesh = mesh_utils.create_device_mesh(mesh_shape=(1, num_devices, 1, 1))
        mesh = Mesh(
            device_mesh.reshape(1, num_devices, 1),
            axis_names=("data", "context", "model"),
        )
        query_specs = ("data", "context", None, None)
    else:
        raise ValueError(f"Parallelism type {parallelism_type} is not supported")

    sharding = PositionalSharding(device_mesh)
    query = jax.device_put(query, sharding)
    logger.info(f"Query head dimension will be sharded across {num_devices} device(s)")
    if shard_kv:
        key = jax.device_put(key, sharding)
        value = jax.device_put(value, sharding)
        kv_specs = query_specs
        logger.info(f"KV head dimension will be sharded across {num_devices} device(s)")
    else:
        # This makes sense if the number of KV heads is small (e.g. 1)
        replicated_sharding = PositionalSharding(device_mesh).replicate()
        replicated_sharding = replicated_sharding.reshape([1] * key.ndim)

        key = jax.device_put(key, replicated_sharding)
        value = jax.device_put(value, replicated_sharding)
        kv_specs = ("data", None, None, None)
        logger.info("KV head dimension will be replicated across devices")

    # Replicate other inputs
    replicated_sharding = PositionalSharding(device_mesh).replicate()
    replicated_sharding = replicated_sharding.reshape([1] * q_pos.ndim)

    q_pos = jax.device_put(q_pos, replicated_sharding)
    q_sids = jax.device_put(q_sids, replicated_sharding)
    kv_pos = jax.device_put(kv_pos, replicated_sharding)
    kv_sids = jax.device_put(kv_sids, replicated_sharding)

    sharded_inputs = {
        "data": (query, key, value, q_pos, q_sids, kv_pos, kv_sids),
        "scale": inputs["scale"],
    }
    return sharded_inputs, mesh, query_specs, kv_specs


def format_duration(seconds: int | float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:.2f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    elif seconds < 3600.0:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes} m {seconds} s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours} h {minutes} m {seconds} s"


def format_tflops(tflops: int | float) -> str:
    if tflops > 1.0:
        return f"{tflops:.2f} TFLOPs"
    else:
        return f"{tflops * 1000.0:.2f} GFLOPs"


def benchmark_jax_func(
    fn_name: str,
    fn: Callable,
    fn_args: tuple,
    fn_kwargs: dict | None = None,
    num_iters: int = 100,
    profile: bool = False,
    tflops_per_iter: float | None = None,
) -> DeviceArray:
    if fn_kwargs is not None:
        # Partially apply kwargs so that they are considered static arguments
        fn = functools.partial(fn, **fn_kwargs)

    fn_compiled = jax.jit(fn)

    # Warmup. Calling compile() is not enough for triton for some reason.
    result = fn_compiled(*fn_args)
    result[0].block_until_ready()

    if profile:
        jax.profiler.start_trace("./profile", create_perfetto_link=True)

    elapsed = timeit.timeit(
        lambda: fn_compiled(*fn_args)[0].block_until_ready(), number=num_iters
    )

    if profile:
        jax.profiler.stop_trace()

    message = f"{fn_name} took {format_duration(elapsed / num_iters)} per iteration"

    if tflops_per_iter is not None:
        tflops_achieved = tflops_per_iter * num_iters / elapsed
        message += f", achieved {format_tflops(tflops_achieved)}"

    logging.info(message)
    return result


def generate_random_segments_lengths(
    seq_len: int,
    num_segments: int,
) -> list[int] | None:
    if num_segments == 1:
        return None

    random.seed(0)
    lengths = [1] * num_segments
    remaining_length = seq_len - num_segments

    for _ in range(remaining_length):
        index = random.randint(0, num_segments - 1)
        lengths[index] += 1

    return lengths


def disable_compile_cache():
    jax.config.update("jax_enable_compilation_cache", False)


def benchmark_flash_attention_cudnn_fwd(
    inputs_dict: AttentionInputsDict,
    benchmark_fn: Callable,
    tflops_per_iter: int,
    num_segments: int,
    is_causal: bool,
) -> DeviceArray:
    attn_inputs_dict, mesh, query_specs, kv_specs = inputs_dict
    attn_inputs = attn_inputs_dict["data"]
    scale = attn_inputs_dict["scale"]
    query, key, value, _, q_sids, _, kv_sids = attn_inputs

    with attention_specs(query_specs, kv_specs):
        if num_segments > 1:
            cudnn_mask = create_cudnn_attn_mask(
                q_sids,
                kv_sids,
            )
        else:
            cudnn_mask = None

        cudnn_sparcity = 0.5 if is_causal else 1.0
        result_cudnn = benchmark_fn(
            "flash_attention_cudnn",
            fn=flash_attention_cudnn,
            fn_args=(query, key, value, q_sids, kv_sids),
            fn_kwargs={
                "mask": cudnn_mask,
                "is_causal": is_causal,
                "scale": scale,
                "mesh": mesh,
            },
            profile=False,
            tflops_per_iter=cudnn_sparcity * tflops_per_iter,
        )

    result_cudnn = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, result_cudnn
    )
    return result_cudnn


def benchmark_flash_attention_triton_fwd(
    inputs_dict: AttentionInputsDict,
    benchmark_fn: Callable,
    tflops_per_iter: int,
    assume_sequential_positions: bool,
    permute_tokens_for_load_balance: bool,
    query_seq_len: int,
    kv_seq_len: int,
    show_attention_mask: bool,
    name: str = "flash_attention_triton",
) -> DeviceArray:
    attn_inputs_dict, mesh, query_specs, kv_specs = inputs_dict
    attn_inputs = attn_inputs_dict["data"]
    scale = attn_inputs_dict["scale"]
    query, key, value, q_pos, q_sids, kv_pos, kv_sids = attn_inputs

    fwd_params = get_default_flash_attention_params(backward=False)

    with attention_specs(query_specs, kv_specs):
        if permute_tokens_for_load_balance:
            attn_inputs = permute_tokens_context_parallelism(
                (query, key, value, q_pos, q_sids),
                mesh=mesh,
            )
            attn_inputs = (*attn_inputs, kv_pos, kv_sids)
            _, _, _, q_pos, q_sids, kv_pos, kv_sids = attn_inputs

        mask = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=fwd_params,
            mesh=mesh,
        )
        if show_attention_mask:
            print_mask(mask[0], kv_seq_len // fwd_params.kv_block_size)
        result_triton = benchmark_fn(
            name,
            fn=flash_attention_triton,
            fn_args=attn_inputs,
            fn_kwargs={
                "mask": mask,
                "fwd_params": fwd_params,
                "assume_sequential_positions": assume_sequential_positions,
                "permute_tokens_for_load_balance": permute_tokens_for_load_balance,
                "scale": scale,
                "mesh": mesh,
            },
            profile=False,
            tflops_per_iter=tflops_per_iter
            * calculate_mask_sparsity(
                mask[0],
                fwd_params.query_block_size,
                fwd_params.kv_block_size,
                query_seq_len,
                kv_seq_len,
            ),
        )
        if permute_tokens_for_load_balance:
            result_triton = unpermute_tokens_context_parallelism(
                result_triton,
                mesh=mesh,
            )

    result_triton = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, result_triton
    )
    return result_triton


def benchmark_flash_attention_cudnn_bwd(
    inputs_dict: AttentionInputsDict,
    benchmark_fn: Callable,
    tflops_per_iter: int,
    num_segments: int,
    is_causal: bool,
) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    attn_inputs_dict, mesh, query_specs, kv_specs = inputs_dict
    attn_inputs = attn_inputs_dict["data"]
    scale = attn_inputs_dict["scale"]
    query, key, value, _, q_sids, _, kv_sids = attn_inputs

    def mha_attention_cudnn_fwd_bwd(*args, **kwargs):
        return flash_attention_cudnn(
            *args,
            **kwargs,
            mesh=mesh,
        ).sum()

    with attention_specs(query_specs, kv_specs):
        if num_segments > 1:
            cudnn_mask = create_cudnn_attn_mask(
                q_sids,
                kv_sids,
            )
        else:
            cudnn_mask = None

        cudnn_sparcity = 0.5 if is_causal else 1.0
        result_cudnn = benchmark_fn(
            "flash_attention_cudnn",
            fn=jax.grad(mha_attention_cudnn_fwd_bwd, argnums=(0, 1, 2)),
            fn_args=(query, key, value, q_sids, kv_sids),
            fn_kwargs={
                "mask": cudnn_mask,
                "is_causal": is_causal,
                "scale": scale,
            },
            profile=False,
            tflops_per_iter=cudnn_sparcity * tflops_per_iter,
        )

    dquery_cudnn, dkey_cudnn, dvalue_cudnn = result_cudnn

    dquery_cudnn = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dquery_cudnn
    )
    dkey_cudnn = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dkey_cudnn
    )
    dvalue_cudnn = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dvalue_cudnn
    )
    return dquery_cudnn, dkey_cudnn, dvalue_cudnn


def benchmark_flash_attention_triton_bwd(
    inputs_dict: AttentionInputsDict,
    benchmark_fn: Callable,
    tflops_per_iter: int,
    assume_sequential_positions: bool,
    permute_tokens_for_load_balance: bool,
    query_seq_len: int,
    kv_seq_len: int,
    show_attention_mask: bool,
    name: str = "flash_attention_triton",
) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    attn_inputs_dict, mesh, query_specs, kv_specs = inputs_dict
    attn_inputs = attn_inputs_dict["data"]
    scale = attn_inputs_dict["scale"]
    query, key, value, q_pos, q_sids, kv_pos, kv_sids = attn_inputs

    fwd_params = get_default_flash_attention_params(backward=False)
    bwd_params = get_default_flash_attention_params(backward=True)

    def flash_mha_attention_triton_fwd_bwd(*args, **kwargs):
        return flash_attention_triton(
            *args,
            **kwargs,
            fwd_params=fwd_params,
            mesh=mesh,
        ).sum()

    with attention_specs(query_specs, kv_specs):
        if permute_tokens_for_load_balance:
            attn_inputs = permute_tokens_context_parallelism(
                (query, key, value, q_pos, q_sids),
                mesh=mesh,
            )
            attn_inputs = (*attn_inputs, kv_pos, kv_sids)
            _, _, _, q_pos, q_sids, kv_pos, kv_sids = attn_inputs

        mask = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            calc_bwd_mask=True,
            mesh=mesh,
        )
        if show_attention_mask:
            print_mask(mask[0], kv_seq_len // fwd_params.kv_block_size)
            print_mask(mask[1], kv_seq_len // bwd_params.query_block_size)
            print_mask(mask[2], query_seq_len // bwd_params.query_block_size)
        result_triton = benchmark_fn(
            name,
            fn=jax.grad(
                flash_mha_attention_triton_fwd_bwd,
                argnums=(0, 1, 2),
            ),
            fn_args=attn_inputs,
            fn_kwargs={
                "mask": mask,
                "bwd_params": bwd_params,
                "assume_sequential_positions": assume_sequential_positions,
                "permute_tokens_for_load_balance": permute_tokens_for_load_balance,
                "scale": scale,
            },
            profile=False,
            tflops_per_iter=tflops_per_iter
            * calculate_mask_sparsity(
                mask[1],
                bwd_params.query_block_size,
                bwd_params.kv_block_size,
                query_seq_len,
                kv_seq_len,
            ),
        )
        if permute_tokens_for_load_balance:
            (
                dquery_triton,
                dkey_triton,
                dvalue_triton,
                q_sids,
            ) = unpermute_tokens_context_parallelism(
                (*result_triton, q_sids),
                mesh=mesh,
            )
            result_triton = dquery_triton, dkey_triton, dvalue_triton

    dquery_triton, dkey_triton, dvalue_triton = result_triton

    dquery_triton = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dquery_triton
    )
    dkey_triton = jnp.where(
        kv_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dkey_triton
    )
    dvalue_triton = jnp.where(
        kv_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dvalue_triton
    )

    return dquery_triton, dkey_triton, dvalue_triton


def check_outputs_fwd(
    results_ref: DeviceArray,
    results: DeviceArray,
    atol: float = 2e-2,
    rtol: float = 1e-5,
) -> None:
    assert jnp.isfinite(results).all()
    assert jnp.isfinite(results_ref).all()
    assert jnp.allclose(results, results_ref, atol=atol, rtol=rtol)

    logger.info("Validation successful.")


def check_outputs_bwd(
    results_ref: DeviceArray,
    results: DeviceArray,
    query_atol: float = 3e-2,
    query_rtol: float = 1e-2,
    kv_atol: float = 3e-2,
    kv_rtol: float = 3e-2,
) -> None:
    dq, dk, dv = results
    dq_ref, dk_ref, dv_ref = results_ref

    assert jnp.allclose(dq, dq_ref, atol=query_atol, rtol=query_rtol)
    assert jnp.allclose(dk, dk_ref, atol=kv_atol, rtol=kv_rtol)
    assert jnp.allclose(dv, dv_ref, atol=kv_atol, rtol=kv_rtol)

    logger.info("Validation successful.")
