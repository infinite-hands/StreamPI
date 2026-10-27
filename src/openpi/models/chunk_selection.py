"""Choosing one action chunk among several candidates sampled from the same observation."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

# (batch, candidates, action_horizon, action_dim) -> (batch, candidates); higher is better. A learned critic plugs in here.
Scorer = Callable[[jax.Array], jax.Array]


def medoid_scores(candidates: jax.Array, *, rows: int | None = None) -> jax.Array:
    """Minus each candidate's mean distance to the others over its first `rows` action rows (the rows that get played).

    The highest score is the most central candidate, so a chunk from a stray mode of the flow is not the one chosen.
    """
    flat = candidates[:, :, :rows].reshape(candidates.shape[0], candidates.shape[1], -1)
    distance = jnp.linalg.norm(flat[:, :, None] - flat[:, None], axis=-1)
    return -distance.sum(axis=-1) / max(candidates.shape[1] - 1, 1)


def select_best(candidates: jax.Array, scorer: Scorer = medoid_scores) -> tuple[jax.Array, jax.Array]:
    """The best-scoring chunk of each row, (batch, action_horizon, action_dim), and its index, (batch,)."""
    best = jnp.argmax(scorer(candidates), axis=-1)
    chosen = jnp.take_along_axis(candidates, best[:, None, None, None], axis=1)[:, 0]
    return chosen, best
