"""sample_action_candidates: several chunks from one prefix pass, each what sample_actions gives for its noise."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import chunk_selection
from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import pi0_testing

ACTION_DIM, HORIZON, TOKENS, BATCH, CANDIDATES = 14, 4, 16, 2, 3
CAMERA = "right_wrist_0_rgb"
MEMORY_KEYS = ("memory_tokens", "memory_kv_cache", "memory_prefix_mask")


def _model_under_test():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=ACTION_DIM, action_horizon=HORIZON,
        hist_horizon=1, max_token_len=TOKENS, dtype="float32", image_keys=(CAMERA,))
    return pi0_testing.awaken(config.create(jax.random.key(0)), jax.random.key(7))


def _observation(rows: list[int]) -> _model.Observation:
    with jax.ensure_compile_time_eval():
        images = jnp.asarray(np.stack([np.random.default_rng(row).uniform(-1, 1, size=(1, 224, 224, 3)) for row in rows]), jnp.float32)
        state = jnp.asarray(np.stack([np.random.default_rng(50 + row).normal(size=ACTION_DIM) for row in rows]), jnp.float32)
    return _model.Observation(
        images={CAMERA: images}, image_masks={CAMERA: jnp.ones((len(rows),), dtype=bool)}, state=state,
        tokenized_prompt=jnp.ones((len(rows), TOKENS), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((len(rows), TOKENS), dtype=bool))


def _fresh_memory(model):
    memory = dict.fromkeys(MEMORY_KEYS)
    model.reset_memory(memory)
    return memory


def test_each_candidate_is_what_sample_actions_gives_for_its_noise():
    model = _model_under_test()
    observation = _observation([1, 2])
    noise = jax.random.normal(jax.random.key(3), (BATCH, CANDIDATES, HORIZON, ACTION_DIM))

    candidates, _ = model.sample_action_candidates(
        jax.random.key(0), observation, num_candidates=CANDIDATES, num_steps=2, noise=noise, memory=_fresh_memory(model))

    assert candidates.shape == (BATCH, CANDIDATES, HORIZON, ACTION_DIM)
    for index in range(CANDIDATES):
        single, _ = model.sample_actions(jax.random.key(0), observation, num_steps=2, noise=noise[:, index], memory=_fresh_memory(model))
        assert np.allclose(np.asarray(candidates[:, index]), np.asarray(single), rtol=1e-4, atol=1e-5), index
    assert not np.allclose(np.asarray(candidates[:, 0]), np.asarray(candidates[:, 1]), atol=1e-4)
    assert not np.allclose(np.asarray(candidates[0]), np.asarray(candidates[1]), atol=1e-4)


def test_the_prefix_memory_is_saved_once_and_matches_a_single_call():
    model = _model_under_test()
    observation = _observation([1, 2])
    noise = jax.random.normal(jax.random.key(3), (BATCH, CANDIDATES, HORIZON, ACTION_DIM))
    many, single = _fresh_memory(model), _fresh_memory(model)

    model.sample_action_candidates(jax.random.key(0), observation, num_candidates=CANDIDATES, num_steps=2, noise=noise, memory=many)
    model.sample_actions(jax.random.key(0), observation, num_steps=2, noise=noise[:, 0], memory=single)

    assert many["memory_prefix_mask"].shape == single["memory_prefix_mask"].shape
    assert np.array_equal(np.asarray(many["memory_prefix_mask"]), np.asarray(single["memory_prefix_mask"]))
    for left, right in zip(jax.tree.leaves(many["memory_kv_cache"]), jax.tree.leaves(single["memory_kv_cache"]), strict=True):
        assert np.allclose(np.asarray(left), np.asarray(right), atol=1e-6)


def test_candidates_need_matching_noise():
    model = _model_under_test()
    wrong = jnp.zeros((BATCH, CANDIDATES + 1, HORIZON, ACTION_DIM))
    with pytest.raises(ValueError, match="num_candidates"):
        model.sample_action_candidates(
            jax.random.key(0), _observation([1, 2]), num_candidates=CANDIDATES, noise=wrong, memory=_fresh_memory(model))


def test_the_medoid_is_not_the_outlier_and_a_custom_scorer_plugs_in():
    rng = np.random.default_rng(0)
    cluster = rng.normal(0, 0.01, size=(1, 4, HORIZON, ACTION_DIM)).astype(np.float32)
    outlier = np.full((1, 1, HORIZON, ACTION_DIM), 5.0, np.float32)
    candidates = jnp.asarray(np.concatenate([cluster[:, :2], outlier, cluster[:, 2:]], axis=1))

    chosen, index = chunk_selection.select_best(candidates)
    assert int(index[0]) != 2
    assert np.allclose(np.asarray(chosen[0]), np.asarray(candidates[0, int(index[0])]))

    only_rows = chunk_selection.medoid_scores(candidates, rows=1)
    assert only_rows.shape == (1, 5)
    assert int(jnp.argmax(only_rows[0])) != 2

    chosen_by_critic, index = chunk_selection.select_best(candidates, scorer=lambda c: c.mean(axis=(2, 3)))
    assert int(index[0]) == 2
    assert np.allclose(np.asarray(chosen_by_critic[0]), 5.0)
