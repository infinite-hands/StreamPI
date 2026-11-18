"""Helpers for tests that need a small Pi0 whose output really depends on its inputs."""

import flax.nnx as nnx
import jax


def awaken(model, key, scale: float = 0.1):
    """Gives the zero-initialised adaRMS modulation kernels and SigLIP head random values.

    A fresh pi0.5 gates every attention and FFN update with a kernel that starts at zero, and SigLIP's output head
    starts at zero too, so its output ignores the prompt and the images entirely and any test that compares outputs
    of a fresh model proves nothing.
    """
    state = nnx.state(model, nnx.Param)

    def fill(path, leaf):
        names = [str(getattr(part, "key", getattr(part, "name", part))) for part in path]
        if ("Dense_0" in names or "head" in names) and "kernel" in names:
            leaf_key = jax.random.fold_in(key, sum(ord(char) * (index + 1) for index, char in enumerate("/".join(names))))
            return scale * jax.random.normal(leaf_key, leaf.shape, leaf.dtype)
        return leaf

    nnx.update(model, jax.tree_util.tree_map_with_path(fill, state))
    return model
