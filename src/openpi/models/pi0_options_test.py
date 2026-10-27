"""The opt-in loss weight and image resolution of Pi0Config: defaults change nothing, options do what they say.
Dummy gemma variants, CPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import pi0_testing

ACTION_DIM, HORIZON, TOKENS = 14, 4, 16
CAMERA = "right_wrist_0_rgb"


def _config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=ACTION_DIM,
        action_horizon=HORIZON, hist_horizon=1, max_token_len=TOKENS, dtype="float32", image_keys=(CAMERA,), **overrides)


def _observation(side: int, frames: int = 1) -> _model.Observation:
    with jax.ensure_compile_time_eval():
        image = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, size=(1, frames, side, side, 3)), dtype=jnp.float32)
        state = jnp.asarray(np.random.default_rng(2).normal(size=(1, ACTION_DIM)), dtype=jnp.float32)
    return _model.Observation(
        images={CAMERA: image}, image_masks={CAMERA: jnp.ones((1,), dtype=bool)}, state=state,
        tokenized_prompt=jnp.ones((1, TOKENS), dtype=jnp.int32), tokenized_prompt_mask=jnp.ones((1, TOKENS), dtype=bool))


def _loss(config, actions):
    key = jax.random.key(0)
    return config.create(key).compute_loss(key, _observation(224), actions)


def test_gripper_loss_weight_is_a_no_op_at_one_and_linear_otherwise():
    actions = jnp.asarray(np.random.default_rng(3).normal(size=(1, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    plain = _loss(_config(), actions)
    one = _loss(_config(gripper_loss_dim=13), actions)
    two = _loss(_config(gripper_loss_dim=13, gripper_loss_weight=2.0), actions)
    three = _loss(_config(gripper_loss_dim=13, gripper_loss_weight=3.0), actions)

    assert np.array_equal(np.asarray(plain), np.asarray(one))
    assert not np.allclose(np.asarray(plain), np.asarray(two))
    # Each weight adds (w - 1) times the gripper's squared error, so the loss is linear in the weight.
    assert np.allclose(np.asarray(three - plain), 2 * np.asarray(two - plain), rtol=1e-4, atol=1e-6)


def test_config_rules_for_the_options():
    with pytest.raises(ValueError, match="gripper_loss_dim"):
        _config(gripper_loss_weight=3.0)
    with pytest.raises(ValueError, match="positive"):
        _config(gripper_loss_dim=13, gripper_loss_weight=0.0)
    with pytest.raises(ValueError, match="outside"):
        _config(gripper_loss_dim=14)
    with pytest.raises(ValueError, match="multiple of 14"):
        _config(image_resolution=(300, 300))
    with pytest.raises(ValueError, match="square"):
        _config(image_resolution=(224, 336))
    spec, _ = _config(image_resolution=(336, 336)).inputs_spec(batch_size=1)
    assert spec.images[CAMERA].shape == (1, 336, 336, 3)


def test_a_336_px_model_reads_576_image_tokens_and_resizes_a_224_px_camera():
    config = _config(image_resolution=(336, 336))
    model = config.create(jax.random.key(0))
    observation = _model.preprocess_observation(None, _observation(224, frames=2), image_keys=(CAMERA,),
                                                image_resolution=model.image_resolution)
    assert observation.images[CAMERA].shape[2:4] == (336, 336)

    tokens, _, _, per_frame, frames = model.embed_prefix(observation)
    assert (frames, per_frame) == (2, 576 + TOKENS)
    assert tokens.shape[1] == 2 * (576 + TOKENS)

    actions = jnp.zeros((1, HORIZON, ACTION_DIM), dtype=jnp.float32)
    assert np.isfinite(np.asarray(model.compute_loss(jax.random.key(0), _observation(224, frames=2), actions))).all()


def test_wrist_augmentation_runs_inside_the_training_loss():
    key = jax.random.key(0)
    actions = jnp.zeros((1, HORIZON, ACTION_DIM), dtype=jnp.float32)
    # Awakened models, so the loss depends on the image at all.
    augmented = pi0_testing.awaken(_config(wrist_augmentation=True).create(key), jax.random.key(7))
    plain = pi0_testing.awaken(_config().create(key), jax.random.key(7))

    loss = augmented.compute_loss(key, _observation(224), actions, train=True)
    reference = plain.compute_loss(key, _observation(224), actions, train=True)
    evaluation = augmented.compute_loss(key, _observation(224), actions, train=False)
    evaluation_reference = plain.compute_loss(key, _observation(224), actions, train=False)

    assert np.isfinite(np.asarray(loss)).all()
    assert not np.allclose(np.asarray(loss), np.asarray(reference), atol=1e-6)
    assert np.array_equal(np.asarray(evaluation), np.asarray(evaluation_reference))


def test_a_two_camera_model_reads_both_cameras_in_every_frame_block():
    keys = ("base_0_rgb", CAMERA)
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=ACTION_DIM, action_horizon=HORIZON,
        hist_horizon=2, max_token_len=TOKENS, dtype="float32", image_keys=keys)
    model = pi0_testing.awaken(config.create(jax.random.key(0)), jax.random.key(7))
    observation = _observation(224, frames=2)
    observation = _model.Observation(
        images={key: observation.images[CAMERA] * (1.0 if key == CAMERA else -1.0) for key in keys},
        image_masks={key: jnp.ones((1,), dtype=bool) for key in keys}, state=observation.state,
        tokenized_prompt=observation.tokenized_prompt, tokenized_prompt_mask=observation.tokenized_prompt_mask)

    tokens, _, _, per_frame, frames = model.embed_prefix(observation)

    assert (frames, per_frame) == (2, 2 * 256 + TOKENS)
    assert tokens.shape[1] == 2 * (2 * 256 + TOKENS)
    actions = jnp.zeros((1, HORIZON, ACTION_DIM), dtype=jnp.float32)
    base = model.compute_loss(jax.random.key(0), observation, actions)
    flipped = _model.Observation(
        images={**observation.images, "base_0_rgb": -observation.images["base_0_rgb"]}, image_masks=observation.image_masks,
        state=observation.state, tokenized_prompt=observation.tokenized_prompt, tokenized_prompt_mask=observation.tokenized_prompt_mask)
    assert not np.allclose(np.asarray(base), np.asarray(model.compute_loss(jax.random.key(0), flipped, actions)), atol=1e-6)
