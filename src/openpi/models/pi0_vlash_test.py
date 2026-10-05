"""VLASH's shared-observation training on pi0.5: the state as adaRMS conditioning, and several
(state, action chunk) branches behind one observation. Dummy gemma variants, CPU."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config

ACTION_DIM, HORIZON, TOKENS = 14, 4, 16


def _config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=ACTION_DIM,
        action_horizon=HORIZON, hist_horizon=1, max_token_len=TOKENS, **overrides)


def _observation(batch: int, branches: int = 0, image_keys=_model.IMAGE_KEYS, masked=()) -> _model.Observation:
    state_shape = (batch, branches, ACTION_DIM) if branches else (batch, ACTION_DIM)
    with jax.ensure_compile_time_eval():
        state = jnp.asarray(np.random.default_rng(0).normal(size=state_shape), dtype=jnp.float32)
        # Seeded per camera, so the same camera carries the same image whichever cameras a fixture has.
        images = {key: jnp.asarray(np.random.default_rng(_model.IMAGE_KEYS.index(key) + 1)
                                   .uniform(-1, 1, size=(batch, 1, *_model.IMAGE_RESOLUTION, 3)), dtype=jnp.float32)
                  for key in image_keys}
    return _model.Observation(
        images={key: (jnp.zeros_like(image) if key in masked else image) for key, image in images.items()},
        image_masks={key: jnp.full((batch,), key not in masked, dtype=bool) for key in image_keys},
        state=state,
        tokenized_prompt=jnp.ones((batch, TOKENS), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch, TOKENS), dtype=bool),
    )


def test_single_camera_model_equals_masked_three_camera_model():
    """Skipping the cameras a recipe masks is the same model: a masked camera's tokens are padding."""
    key = jax.random.key(0)
    three = _config(dtype="float32").create(key)
    one = _config(dtype="float32", image_keys=("left_wrist_0_rgb",)).create(key)
    actions = jnp.asarray(np.random.default_rng(1).normal(size=(2, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    masked = _observation(2, masked=("base_0_rgb", "right_wrist_0_rgb"))
    only = _observation(2, image_keys=("left_wrist_0_rgb",))
    assert np.allclose(np.asarray(masked.images["left_wrist_0_rgb"]), np.asarray(only.images["left_wrist_0_rgb"]))
    loss_three, loss_one = three.compute_loss(key, masked, actions), one.compute_loss(key, only, actions)
    assert loss_one.shape == (2, HORIZON)
    assert np.allclose(np.asarray(loss_three), np.asarray(loss_one), rtol=1e-3, atol=1e-4), (loss_three, loss_one)
    # The one-camera model also accepts an observation that still carries the other cameras (serving sends all three).
    loss_extra = one.compute_loss(key, masked, actions)
    assert np.allclose(np.asarray(loss_extra), np.asarray(loss_one), rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError, match="image_keys"):
        _config(image_keys=("nose_cam",))


def test_config_rules():
    assert _config().discrete_state_input and not _config(state_cond=True).discrete_state_input
    with pytest.raises(ValueError, match="state_cond"):
        _config(vlash_branches=3)
    with pytest.raises(ValueError, match="discrete_state_input"):
        _config(state_cond=True, discrete_state_input=True)
    with pytest.raises(ValueError, match="pi0.5"):
        _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", pi05=False, state_cond=True)
    spec, actions = _config(state_cond=True, vlash_branches=3).inputs_spec(batch_size=2)
    assert spec.state.shape == (2, 3, ACTION_DIM) and actions.shape == (2, 3, HORIZON, ACTION_DIM)


def test_branch_isolation_mask():
    mask = np.asarray(_pi0.branch_isolation_mask(prefix_len=3, branches=2, branch_len=2))
    assert mask.shape == (7, 7)
    assert mask[:3].all() and mask[:, :3].all(), "the prefix rows and columns are untouched"
    assert mask[3:5, 3:5].all() and mask[5:7, 5:7].all(), "a branch sees itself"
    assert not mask[3:5, 5:7].any() and not mask[5:7, 3:5].any(), "never a sibling"


def test_state_cond_starts_as_the_time_conditioning():
    key = jax.random.key(0)
    plain = _config().create(key)
    conditioned = _config(state_cond=True).create(key)
    obs = _observation(2)
    x_t = jnp.zeros((2, HORIZON, ACTION_DIM))
    time = jnp.full((2, HORIZON), 0.5)
    _, _, _, cond_plain = plain.embed_suffix(obs, x_t, time)
    _, _, _, cond_state = conditioned.embed_suffix(obs, x_t, time)
    assert np.allclose(np.asarray(cond_plain), np.asarray(cond_state)), "zero-initialised: no change at step 0"
    masked = _config(state_cond=True, state_cond_dims=(0, 1, 2)).create(key)
    kernel = np.asarray(masked.state_mlp_out.kernel.value)
    assert not kernel.any()


def test_branched_loss_isolates_branches():
    key = jax.random.key(0)
    model = _config(state_cond=True, vlash_branches=3).create(key)
    obs = _observation(1, branches=3)
    actions = jnp.asarray(np.random.default_rng(1).normal(size=(1, 3, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    loss = model.compute_loss(key, obs, actions)
    assert loss.shape == (1, 3, HORIZON) and bool(jnp.isfinite(loss).all())
    # Changing branch 1's actions and state leaves branch 0's and 2's losses exactly as they were.
    other = actions.at[:, 1].set(actions[:, 1] + 1.0)
    other_obs = dataclasses.replace(obs, state=obs.state.at[:, 1].add(1.0))
    loss_other = model.compute_loss(key, other_obs, other)
    assert np.allclose(np.asarray(loss[:, 0]), np.asarray(loss_other[:, 0]))
    assert np.allclose(np.asarray(loss[:, 2]), np.asarray(loss_other[:, 2]))
    assert not np.allclose(np.asarray(loss[:, 1]), np.asarray(loss_other[:, 1]))
    # The plain layout still serves: one state, one chunk.
    single = _config(state_cond=True).create(key)
    plain_loss = single.compute_loss(key, _observation(1), actions[:, 0])
    assert plain_loss.shape == (1, HORIZON)
    sampled, _memory = single.sample_actions(key, _observation(1), num_steps=2, memory={"memory_tokens": None,
                                                                                         "memory_kv_cache": None,
                                                                                         "memory_prefix_mask": None})
    assert sampled.shape == (1, HORIZON, ACTION_DIM)


def test_packed_model_serves_one_state():
    """A model trained with branches serves one state per call: the branch count shapes training only."""
    key = jax.random.key(0)
    packed = _config(state_cond=True, vlash_branches=9, state_cond_dims=tuple(range(7))).create(key)
    obs = _observation(1)
    sampled, _memory = packed.sample_actions(key, obs, num_steps=2, memory={"memory_tokens": None,
                                                                          "memory_kv_cache": None,
                                                                          "memory_prefix_mask": None})
    assert sampled.shape == (1, HORIZON, ACTION_DIM) and bool(jnp.isfinite(sampled).all())
    branched = _observation(1, branches=9)
    actions = jnp.zeros((1, 9, HORIZON, ACTION_DIM), dtype=jnp.float32)
    assert packed.compute_loss(key, branched, actions).shape == (1, 9, HORIZON)
