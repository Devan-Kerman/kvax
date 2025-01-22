def get_multi_head_attention_tflops(
    batch_size: int,
    query_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    causal: bool,
    num_gpus: int = 1,
    forward_pass: bool = True,
    backward_pass: bool = False,
) -> float:
    if not forward_pass and not backward_pass:
        raise ValueError(
            "At least one of the forward_pass or backward_pass should be True"
        )

    flops_per_batch = 2 * (qk_head_dim + value_head_dim) * query_seq_len * kv_seq_len
    result = batch_size * num_heads * flops_per_batch / 1e12

    if causal:
        result *= 0.5
    if forward_pass and backward_pass:
        result *= 3.5
    elif backward_pass:
        result *= 2.5

    result /= num_gpus

    return result
