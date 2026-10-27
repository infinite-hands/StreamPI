from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


def ema_update(decay: float, ema_params: at.PyTree, new_params: at.PyTree) -> at.PyTree:
    """One EMA step, computed in float32 and stored in each leaf's own dtype.

    A frozen bfloat16 weight (old == new) therefore stays bit-identical; doing the arithmetic in bfloat16 adds rounding
    noise to it every step, which a 0.99 decay lets accumulate.
    """

    def step(old, new):
        return (decay * old.astype(jnp.float32) + (1 - decay) * new.astype(jnp.float32)).astype(old.dtype)

    return jax.tree.map(step, ema_params, new_params)
