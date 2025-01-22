from dataclasses import dataclass

import chex
import jax
import jax.experimental
import jax.experimental.shard_map

DeviceArray = chex.ArrayDevice
PRNGKey = chex.PRNGKey
Specs = jax.experimental.shard_map.Specs

FlattenedAttentionMask = tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray]


@dataclass
class AttentionMask:
    lower_bounds: DeviceArray
    upper_bounds: DeviceArray
    lower_full_bounds: DeviceArray
    upper_full_bounds: DeviceArray

    def flatten(self) -> FlattenedAttentionMask:
        return (
            self.lower_bounds,
            self.upper_bounds,
            self.lower_full_bounds,
            self.upper_full_bounds,
        )

    @classmethod
    def unflatten(cls, flatten_mask: FlattenedAttentionMask) -> "AttentionMask":
        return cls(
            lower_bounds=flatten_mask[0],
            upper_bounds=flatten_mask[1],
            lower_full_bounds=flatten_mask[2],
            upper_full_bounds=flatten_mask[3],
        )
