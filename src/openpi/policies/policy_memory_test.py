"""Policy.infer's memory resets: the KV window every hist_horizon calls, the compressed memory only at an episode's start."""

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.policies import policy as _policy


class _KvOnly(nnx.Module):
    """Counts the calls each memory has lived through, so a reset shows as the count starting over. This model has no
    compressed memory to reset: the count in `long_frames` just keeps running."""

    hist_horizon = 2

    def __init__(self):
        self.weight = nnx.Param(jnp.zeros(()))

    def reset_memory(self, memory):
        memory["memory_kv_cache"] = None

    def sample_actions(self, rng, observation, *, memory=None, **_):
        kv = memory.get("memory_kv_cache")
        long = memory.get("long_frames")
        memory["memory_kv_cache"] = jnp.asarray(1 if kv is None else kv + 1)
        memory["long_frames"] = jnp.asarray(1 if long is None else long + 1)
        return jnp.zeros((observation.state.shape[0], 2, 3)), memory


class _Counter(_KvOnly):
    def reset_long_memory(self, memory):
        memory["long_frames"] = None


def _observation(step: int) -> dict:
    return {"image": {"cam": np.zeros((1, 8, 8, 3), np.uint8)}, "image_mask": {"cam": np.True_},
            "state": np.zeros(3, np.float32), "step": step}


def _counts(policy, steps):
    seen = []
    for step in steps:
        policy.infer(_observation(step))
        seen.append((int(policy.memory["memory_kv_cache"]), int(policy.memory["long_frames"])))
    return seen


def test_the_kv_window_resets_every_hist_horizon_calls_and_the_long_memory_only_at_step_zero():
    policy = _policy.Policy(_Counter())
    # KV count: 1, 2 per window of two calls. Long count: unbroken through the reset at step 2, restarted at step 0.
    assert _counts(policy, [0, 1, 2, 3]) == [(1, 1), (2, 2), (1, 3), (2, 4)]
    assert _counts(policy, [0, 1]) == [(1, 1), (2, 2)]


def test_a_model_without_a_long_memory_is_left_alone():
    assert not hasattr(_KvOnly(), "reset_long_memory")
    # Step 0 comes again, and Policy neither fails nor clears anything the model did not offer to clear.
    assert _counts(_policy.Policy(_KvOnly()), [0, 1, 2, 0]) == [(1, 1), (2, 2), (1, 3), (1, 4)]
