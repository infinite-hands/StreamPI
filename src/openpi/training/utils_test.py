import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import utils

DECAY = 0.99


def test_ema_update_leaves_frozen_bfloat16_weights_bit_identical_and_tracks_the_trained_ones():
    rng = np.random.default_rng(0)
    frozen = jnp.asarray(rng.normal(size=1000) * 0.05, dtype=jnp.bfloat16)
    ema = {"frozen": frozen, "trained": jnp.zeros(4, jnp.float32)}
    update = jax.jit(lambda ema, new: utils.ema_update(DECAY, ema, new))

    expected = np.zeros(4)
    for step in range(500):
        target = np.full(4, float(step % 7))
        ema = update(ema, {"frozen": frozen, "trained": jnp.asarray(target, jnp.float32)})
        expected = DECAY * expected + (1 - DECAY) * target

    assert ema["frozen"].dtype == jnp.bfloat16
    assert np.array_equal(np.asarray(ema["frozen"]), np.asarray(frozen))
    assert np.allclose(np.asarray(ema["trained"]), expected, atol=1e-4)


def test_the_bfloat16_arithmetic_this_replaces_really_does_drift():
    rng = np.random.default_rng(0)
    frozen = jnp.asarray(rng.normal(size=10_000) * 0.05, dtype=jnp.bfloat16)
    old_style = jax.jit(lambda old, new: DECAY * old + (1 - DECAY) * new)
    ema = frozen
    for _ in range(500):
        ema = old_style(ema, frozen)
    drift = float(jnp.sqrt(jnp.mean((ema.astype(jnp.float32) - frozen.astype(jnp.float32)) ** 2)))
    assert drift / float(jnp.sqrt(jnp.mean(frozen.astype(jnp.float32) ** 2))) > 0.01
