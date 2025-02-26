import functools
import logging

import click
import jax

from kvax.utils.benchmarking import (
    benchmark_flash_attention_cudnn_bwd,
    benchmark_flash_attention_cudnn_fwd,
    benchmark_flash_attention_triton_bwd,
    benchmark_flash_attention_triton_fwd,
    benchmark_jax_func,
    check_outputs_bwd,
    check_outputs_fwd,
    disable_compile_cache,
    generate_random_segments_lengths,
    make_random_attention_inputs_with_sharding,
    shard_input_data,
)
from kvax.utils.math import get_multi_head_attention_tflops

disable_compile_cache()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command("mha")
@click.option("--num-iters", type=int, default=10)
@click.option("--batch-size", type=int, default=4)
@click.option("--query-seq-len", type=int, default=8192)
@click.option("--kv-seq-len", type=int, default=8192)
@click.option("--num-heads", type=int, default=16)
@click.option("--num-kv-heads", type=int, default=16)
@click.option("--qk-head-dim", type=int, default=128)
@click.option("--value-head-dim", type=int, default=128)
@click.option("--assume-sequential-positions", type=bool, default=True)
@click.option("--num-segments", type=int, default=1)
@click.option("--num-pad-tokens", type=int, default=0)
@click.option("--show-attention-mask", is_flag=True)
def mha_benchmark(
    num_iters: int,
    batch_size: int,
    query_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    assume_sequential_positions: bool,
    num_segments: int,
    num_pad_tokens: int,
    show_attention_mask: bool,
):
    # Create inputs
    inputs_dict = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        num_pad_tokens=num_pad_tokens,
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        rng=jax.random.PRNGKey(0),
    )

    # Calculate the FLOPs per iteration for each GPU
    tflops_per_iter = get_multi_head_attention_tflops(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        causal=False,
        num_gpus=len(jax.devices()),
    )

    # Predefine number of iterations
    benchmark_fn = functools.partial(
        benchmark_jax_func,
        num_iters=num_iters,
    )

    # Benchmarking
    result_cudnn = benchmark_flash_attention_cudnn_fwd(
        inputs_dict,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        num_segments=num_segments,
        is_causal=True,
    )
    result_triton = benchmark_flash_attention_triton_fwd(
        inputs_dict,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        assume_sequential_positions=assume_sequential_positions,
        # This is always False when context parallelism is not enabled
        permute_tokens_for_load_balance=False,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        show_attention_mask=show_attention_mask,
    )

    # Results validation
    check_outputs_fwd(result_cudnn, result_triton)


@cli.command("mha_bwd")
@click.option("--num-iters", type=int, default=10)
@click.option("--batch-size", type=int, default=4)
@click.option("--query-seq-len", type=int, default=8192)
@click.option("--kv-seq-len", type=int, default=8192)
@click.option("--num-heads", type=int, default=16)
@click.option("--num-kv-heads", type=int, default=16)
@click.option("--qk-head-dim", type=int, default=128)
@click.option("--value-head-dim", type=int, default=128)
@click.option("--assume-sequential-positions", type=bool, default=True)
@click.option("--num-segments", type=int, default=1)
@click.option("--num-pad-tokens", type=int, default=0)
@click.option("--show-attention-mask", is_flag=True)
def mha_benchmark_bwd(
    num_iters: int,
    batch_size: int,
    query_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    assume_sequential_positions: bool,
    num_segments: int,
    num_pad_tokens: int,
    show_attention_mask: bool,
):
    # Create inputs
    inputs_dict = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        num_pad_tokens=num_pad_tokens,
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        rng=jax.random.PRNGKey(0),
    )

    # Calculate the FLOPs per iteration for each GPU
    tflops_per_iter = get_multi_head_attention_tflops(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        causal=False,
        forward_pass=True,
        backward_pass=True,
        num_gpus=len(jax.devices()),
    )

    # Predefine number of iterations
    benchmark_fn = functools.partial(
        benchmark_jax_func,
        num_iters=num_iters,
    )

    # Benchmarking
    result_cudnn = benchmark_flash_attention_cudnn_bwd(
        inputs_dict,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        num_segments=num_segments,
        is_causal=True,
    )
    result_triton = benchmark_flash_attention_triton_bwd(
        inputs_dict,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        assume_sequential_positions=assume_sequential_positions,
        # This is always False when context parallelism is not enabled
        permute_tokens_for_load_balance=False,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        show_attention_mask=show_attention_mask,
    )

    # Results validation
    check_outputs_bwd(result_cudnn, result_triton)


@cli.command("mha_cp")
@click.option("--num-iters", type=int, default=10)
@click.option("--batch-size", type=int, default=4)
@click.option("--query-seq-len", type=int, default=32768)
@click.option("--kv-seq-len", type=int, default=32768)
@click.option("--num-heads", type=int, default=32)
@click.option("--num-kv-heads", type=int, default=8)
@click.option("--qk-head-dim", type=int, default=128)
@click.option("--value-head-dim", type=int, default=128)
@click.option("--assume-sequential-positions", type=bool, default=True)
@click.option("--num-segments", type=int, default=1)
@click.option("--permute-tokens-for-load-balance", type=bool, default=True)
@click.option("--show-attention-mask", is_flag=True)
def mha_cp_fwd_benchmark(
    num_iters: int,
    batch_size: int,
    query_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    assume_sequential_positions: bool,
    num_segments: int,
    permute_tokens_for_load_balance: bool,
    show_attention_mask: bool,
):
    # Create inputs
    inputs_dict_context = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        shard_kv=False,
        parallelism_type="context",
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        rng=jax.random.PRNGKey(0),
    )

    # Calculate the FLOPs per iteration for each GPU
    tflops_per_iter = get_multi_head_attention_tflops(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        causal=False,
        num_gpus=len(jax.devices()),
    )

    # Predefine number of iterations
    benchmark_fn = functools.partial(
        benchmark_jax_func,
        num_iters=num_iters,
    )

    # Benchmarking context parallelism
    result_cp = benchmark_flash_attention_triton_fwd(
        inputs_dict_context,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        assume_sequential_positions=assume_sequential_positions,
        permute_tokens_for_load_balance=permute_tokens_for_load_balance,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        show_attention_mask=show_attention_mask,
        name="flash_attention_triton_cp",
    )

    # Reshard the same inputs
    attn_inputs_tensor = shard_input_data(
        inputs_dict_context[0],
        parallelism_type="tensor",
    )

    # Benchmarking tensor parallelism
    result_tp = benchmark_flash_attention_triton_fwd(
        attn_inputs_tensor,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        assume_sequential_positions=assume_sequential_positions,
        # This is always False when context parallelism is not enabled
        permute_tokens_for_load_balance=False,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        show_attention_mask=show_attention_mask,
        name="flash_attention_triton_tp",
    )

    # Results validation
    check_outputs_fwd(result_cp, result_tp)


@cli.command("mha_cp_bwd")
@click.option("--num-iters", type=int, default=10)
@click.option("--batch-size", type=int, default=4)
@click.option("--query-seq-len", type=int, default=32768)
@click.option("--kv-seq-len", type=int, default=32768)
@click.option("--num-heads", type=int, default=32)
@click.option("--num-kv-heads", type=int, default=8)
@click.option("--qk-head-dim", type=int, default=128)
@click.option("--value-head-dim", type=int, default=128)
@click.option("--assume-sequential-positions", type=bool, default=True)
@click.option("--num-segments", type=int, default=1)
@click.option("--permute-tokens-for-load-balance", type=bool, default=True)
@click.option("--show-attention-mask", is_flag=True)
def mha_cp_bwd_benchmark(
    num_iters: int,
    batch_size: int,
    query_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    assume_sequential_positions: bool,
    num_segments: int,
    permute_tokens_for_load_balance: bool,
    show_attention_mask: bool,
):
    # Create inputs
    inputs_dict_context = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        shard_kv=False,
        parallelism_type="context",
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        rng=jax.random.PRNGKey(0),
    )

    # Calculate the FLOPs per iteration for each GPU
    tflops_per_iter = get_multi_head_attention_tflops(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        causal=False,
        forward_pass=True,
        backward_pass=True,
        num_gpus=len(jax.devices()),
    )

    # Predefine number of iterations
    benchmark_fn = functools.partial(
        benchmark_jax_func,
        num_iters=num_iters,
    )

    # Benchmarking context parallelism
    result_cp = benchmark_flash_attention_triton_bwd(
        inputs_dict_context,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        assume_sequential_positions=assume_sequential_positions,
        permute_tokens_for_load_balance=permute_tokens_for_load_balance,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        show_attention_mask=show_attention_mask,
        name="flash_attention_triton_cp",
    )

    # Reshard the same inputs
    attn_inputs_tensor = shard_input_data(
        inputs_dict_context[0],
        parallelism_type="tensor",
    )

    # Benchmarking tensor parallelism
    result_tp = benchmark_flash_attention_triton_bwd(
        attn_inputs_tensor,
        benchmark_fn,
        tflops_per_iter=tflops_per_iter,
        assume_sequential_positions=assume_sequential_positions,
        # This is always False when context parallelism is not enabled
        permute_tokens_for_load_balance=False,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        show_attention_mask=show_attention_mask,
        name="flash_attention_triton_tp",
    )

    # Results validation
    check_outputs_bwd(result_cp, result_tp)


if __name__ == "__main__":
    cli()
