"""A shorter prompt padding is the same model: padded tokens are masked and RoPE positions skip them.
Dummy gemma variants, CPU."""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config

ACTION_DIM, HORIZON, HIST, REAL_TOKENS, BATCH = 14, 4, 3, 10, 2


def _config(max_token_len: int) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=ACTION_DIM,
        action_horizon=HORIZON, hist_horizon=HIST, max_token_len=max_token_len, dtype="float32")


def _observation(max_token_len: int, pad_id: int, *, frames: int = HIST, seed: int = 0) -> _model.Observation:
    tokens = np.full((BATCH, max_token_len), pad_id, np.int32)
    tokens[:, :REAL_TOKENS] = np.random.default_rng(100).integers(3, 100, size=(BATCH, REAL_TOKENS))
    mask = np.zeros((BATCH, max_token_len), bool)
    mask[:, :REAL_TOKENS] = True
    with jax.ensure_compile_time_eval():
        images = {key: jnp.asarray(np.random.default_rng(seed + index).uniform(-1, 1, size=(BATCH, frames, *_model.IMAGE_RESOLUTION, 3)),
                                   dtype=jnp.float32)
                  for index, key in enumerate(_model.IMAGE_KEYS)}
        state = jnp.asarray(np.random.default_rng(seed + 50).normal(size=(BATCH, ACTION_DIM)), dtype=jnp.float32)
    return _model.Observation(
        images=images, image_masks={key: jnp.ones((BATCH,), dtype=bool) for key in _model.IMAGE_KEYS}, state=state,
        tokenized_prompt=jnp.asarray(tokens), tokenized_prompt_mask=jnp.asarray(mask))


def test_training_loss_ignores_how_far_the_prompt_is_padded():
    key = jax.random.key(0)
    short, long = _config(24).create(key), _config(56).create(key)
    actions = jnp.asarray(np.random.default_rng(1).normal(size=(BATCH, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    loss_short = short.compute_loss(key, _observation(24, pad_id=0), actions)
    loss_long = long.compute_loss(key, _observation(56, pad_id=7), actions)
    assert loss_short.shape == (BATCH, HORIZON)
    assert np.allclose(np.asarray(loss_short), np.asarray(loss_long), rtol=1e-4, atol=1e-5), (loss_short, loss_long)


def test_serving_with_a_carried_memory_ignores_how_far_the_prompt_is_padded():
    key = jax.random.key(0)
    models = {24: _config(24).create(key), 56: _config(56).create(key)}
    memories = {length: {} for length in models}
    for length, model in models.items():
        model.reset_memory(memories[length])
    for call in range(2):
        outputs = {}
        for length, model in models.items():
            observation = _observation(length, pad_id=0 if length == 24 else 7, frames=1, seed=10 * call)
            outputs[length], memories[length] = model.sample_actions(jax.random.key(call), observation, num_steps=2,
                                                                     memory=memories[length])
        assert np.allclose(np.asarray(outputs[24]), np.asarray(outputs[56]), rtol=1e-4, atol=1e-5), call
