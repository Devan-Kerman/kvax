from typing import Callable, ParamSpec, TypeVar

import jax
from jax.experimental.shard_map import shard_map as jax_shard_map
from jax.interpreters import pxla
from jax.sharding import Mesh, PartitionSpec

from kvax.utils.specs import Axes, get_attention_specs
from kvax.utils.typing import Specs

P = ParamSpec("P")
R = TypeVar("R")


def _get_global_mesh() -> Mesh:
    mesh_env = pxla.thread_resources.env
    return mesh_env.physical_mesh


def _global_mesh_defined() -> bool:
    return _get_global_mesh != ()


def _check_mesh_or_get_global(mesh: Mesh | None) -> Mesh:
    if not _global_mesh_defined() and mesh is None:
        raise ValueError(
            "Mesh weren't provided."
            " Either mesh should not be None or global Mesh needs to be set as"
            " a context."
        )

    if mesh is None:
        mesh = _get_global_mesh()
    return mesh


def shard_map(
    f: Callable[P, R],
    in_specs: Specs,
    out_specs: Specs,
    mesh: Mesh | None = None,
) -> Callable[P, R]:
    mesh = _check_mesh_or_get_global(mesh)

    def is_leaf(node):
        return isinstance(node, tuple) and (
            len(node) == 0 or isinstance(node[0], str) or (node[0] is None)
        )

    res_specs = []
    for specs in [in_specs, out_specs]:
        res_specs.append(
            jax.tree_util.tree_map(
                lambda node: node if node is None else PartitionSpec(*node),
                specs,
                is_leaf=is_leaf,
            )
        )
    in_specs, out_specs = res_specs

    return jax_shard_map(
        f,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )


def get_query_context_mesh_axis_name(mesh: Mesh | None) -> str:
    mesh = _check_mesh_or_get_global(mesh)
    _query_specs = get_attention_specs().query_specs
    return _query_specs[Axes.query_sequence]


def get_query_context_mesh_axis_size(mesh: Mesh | None) -> int:
    mesh = _check_mesh_or_get_global(mesh)

    query_context_axis = get_query_context_mesh_axis_name(mesh)

    if query_context_axis not in mesh.shape:
        raise ValueError(
            f"Mesh axis {query_context_axis} is not present in the device mesh shape."
        )

    return mesh.shape[query_context_axis]
