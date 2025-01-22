from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Generator

from kvax.utils.typing import Specs


@dataclass
class AttentionSpecs:
    query_specs: Specs | None = None
    kv_specs: Specs | None = None


class Axes(IntEnum):
    batch = 0
    query_sequence = 1
    kv_sequence = 1
    heads = 2
    kv_heads = 2
    head_dim = 3


# Global context for attention axes.
_attention_specs = AttentionSpecs()


def _validate_attention_specs(query_specs: Specs, kv_specs: Specs) -> None:
    if len(query_specs) != 4:
        raise ValueError(
            f"query_specs must have exactly 4 dimensions, got {len(query_specs)}."
        )
    if len(kv_specs) != 4:
        raise ValueError(
            f"kv_specs must have exactly 4 dimensions, got {len(kv_specs)}."
        )
    if query_specs[Axes.batch] != kv_specs[Axes.batch]:
        raise ValueError(
            "Batch dimension sharding must match between query_specs and kv_specs."
        )
    if query_specs[Axes.head_dim] != kv_specs[Axes.head_dim]:
        raise ValueError(
            "head_dim dimension sharding must match between query_specs and kv_specs."
        )
    if not kv_specs[Axes.kv_sequence] is None:
        raise ValueError(
            "Context parallelism with sharded key/value tensors is not supported."
        )
    if not kv_specs[Axes.head_dim] is None:
        raise ValueError("Sharding of attention_head_dim is not supported.")


def set_attention_specs(query_specs: Specs, kv_specs: Specs) -> None:
    """
    Sets the shard map specifications for query, key, and value tensors.
    The expected format is as follows:
      - query: (batch, query_sequence, heads, attention_head_dim)
      - kv: (batch, kv_sequence, kv_heads, attention_head_dim)

    Args:
        query_specs (Specs): Specifications for sharding the query tensor.
        kv_specs (Specs): Specifications for sharding the key and value tensors.

    Returns:
        None
    """
    _validate_attention_specs(query_specs, kv_specs)
    _attention_specs.query_specs = query_specs
    _attention_specs.kv_specs = kv_specs


def get_attention_specs() -> AttentionSpecs:
    """
    Returns the shard map specifications for query, key, and value tensors.
    The specifications are returned in the following format:
      - query: (batch, query_sequence, heads, attention_head_dim)
      - kv: (batch, kv_sequence, kv_heads, attention_head_dim)

    Returns:
        AttentionSpecs: A specifications for query and kv tensors sharding.
    """
    if _attention_specs.query_specs is None or _attention_specs.kv_specs is None:
        raise ValueError(
            "Attention specifications must be initialised "
            "before calling get_attention_specs."
        )
    return _attention_specs


@contextmanager
def attention_specs(query_specs: Specs, kv_specs: Specs) -> Generator[None, None, None]:
    """
    A context manager for setting the attention specifications
    for query and key, value tensors.

    Args:
        query_specs (Specs): Specifications for sharding the query tensor.
        kv_specs (Specs): Specifications for sharding the key and value tensors.

    Yields:
        None
    """
    old_query_specs = _attention_specs.query_specs
    old_kv_specs = _attention_specs.kv_specs

    set_attention_specs(query_specs, kv_specs)

    try:
        yield
    finally:
        _attention_specs.query_specs = old_query_specs
        _attention_specs.kv_specs = old_kv_specs
