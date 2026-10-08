"""RACE (arXiv 2610.05719) on pi0.5: the zero-initialised transition modulation, the auxiliary pass and timing
head, and the serve output. Dummy gemma variants, float32, CPU.

A freshly initialised pi0.5 is prefix-blind -- its adaRMS modulation kernels start at zero, so every residual
gate is 0 and the action expert passes its input straight through. The fixtures therefore randomise those
kernels first, or "RACE changes nothing" would hold trivially."""

import dataclasses

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.policies import agilex_policy
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import weight_loaders

ACTION_DIM, HORIZON, TOKENS, HIST = 14, 4, 16, 2
CAMERA = "left_wrist_0_rgb"
ROWS = HORIZON + 2


def _config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(**{
        "paligemma_variant": "dummy", "action_expert_variant": "dummy", "pi05": True, "action_dim": ACTION_DIM,
        "action_horizon": HORIZON, "hist_horizon": HIST, "max_token_len": TOKENS, "dtype": "float32",
        "image_keys": (CAMERA,), **overrides})


def _params(model) -> dict:
    return nnx.state(model).to_pure_dict()


def _with_params(model, params: dict):
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    return nnx.merge(graphdef, state)


def _randomised(params: dict, pattern: str, seed: int, scale: float = 0.1) -> dict:
    """`params` with every kernel whose path contains `pattern` replaced by N(0, scale^2) draws."""
    rng = np.random.default_rng(seed)
    flat = traverse_util.flatten_dict(params, sep="/")
    for key, value in flat.items():
        if pattern in key and key.endswith("kernel"):
            flat[key] = jnp.asarray(rng.normal(scale=scale, size=value.shape), dtype=value.dtype)
    return traverse_util.unflatten_dict(flat, sep="/")


def _observation(batch: int, frames: int, *, seed: int = 0, window=None, window_mask=None) -> _model.Observation:
    rng = np.random.default_rng(seed)
    with jax.ensure_compile_time_eval():
        image = jnp.asarray(rng.uniform(-1, 1, size=(batch, frames, *_model.IMAGE_RESOLUTION, 3)), dtype=jnp.float32)
        state = jnp.asarray(rng.normal(size=(batch, ACTION_DIM)), dtype=jnp.float32)
    return _model.Observation(
        images={CAMERA: image},
        image_masks={CAMERA: jnp.ones((batch,), dtype=bool)},
        state=state,
        tokenized_prompt=jnp.asarray(rng.integers(0, 1000, size=(batch, TOKENS)), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray(np.arange(TOKENS)[None, :] < TOKENS - 3).repeat(batch, 0),
        transition_window=window,
        transition_window_mask=window_mask,
    )


def _fresh_memory() -> dict:
    return dict(memory_tokens=None, memory_kv_cache=None, memory_prefix_mask=None)


@pytest.fixture(scope="module")
def models():
    """(base, race at its initialisation, race with its modulation randomised): the race models are built the
    way training builds them -- fresh weights, then the base checkpoint merged in by CheckpointWeightLoader's
    rule, which lets RACE's weights be missing."""
    key = jax.random.key(0)
    base = _config().create(key)
    base_params = _randomised(_params(base), "Dense_0", seed=1)   # the adaRMS modulation kernels
    base = _with_params(base, base_params)

    race = _config(race=True).create(jax.random.key(7))
    merged = weight_loaders._merge_params(base_params, _params(race), missing_regex=_pi0_config.RACE_PARAMS_REGEX)
    race = _with_params(race, merged)
    live = _with_params(race, _randomised(_params(race), "race_modulation", seed=2))
    return base, race, live


def _targets(batch: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    window = jnp.asarray(rng.uniform(0, 1, size=(batch, ROWS)), dtype=jnp.float32)
    return window, jnp.ones((batch, ROWS), dtype=bool)


def test_race_weights_exist_and_start_at_zero(models):
    _base, race, _live = models
    flat = traverse_util.flatten_dict(_params(race), sep="/")
    depth, width = 4, 64   # the dummy expert
    assert flat["PaliGemma/llm/layers/pre_attention_norm_1/race_modulation/kernel"].shape == (depth, width, 3 * width)
    assert flat["PaliGemma/llm/layers/pre_ffw_norm_1/race_modulation/kernel"].shape == (depth, width, 3 * width)
    assert flat["PaliGemma/llm/final_norm_1/race_modulation/kernel"].shape == (width, 3 * width)
    for key, value in flat.items():
        if "race_modulation" in key:
            assert not np.asarray(value).any(), key
    assert not any("race_modulation" in key and "_1/" not in key for key in flat), "only the action expert's norms"
    np.testing.assert_allclose(np.asarray(jax.nn.sigmoid(flat["race_gate_logits"])), 0.5)
    assert flat["race_head_proj/kernel"].shape == (width, 16)   # d_a -> d_z (the dummy VLM's kv width)


def test_step0_equals_the_base_model(models):
    """(a) RACE at initialisation, loaded over a non-RACE checkpoint: the base model's flow loss and actions."""
    base, race, _live = models
    window, inside = _targets(2)
    obs = _observation(2, HIST, window=window, window_mask=inside)
    actions = jnp.asarray(np.random.default_rng(4).normal(size=(2, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    key = jax.random.key(5)
    base_loss = base.compute_loss(key, obs, actions, train=True)
    race_loss = race.compute_loss(key, obs, actions, train=True)
    assert np.array_equal(np.asarray(base_loss), np.asarray(race_loss)), np.abs(base_loss - race_loss).max()
    total, parts = race.training_loss(key, obs, actions)
    assert np.array_equal(np.asarray(parts["flow_loss"]), np.asarray(jnp.mean(base_loss)))
    np.testing.assert_allclose(
        float(total), float(parts["flow_loss"] + 0.1 * parts["aux_loss"] + 0.05 * parts["timing_loss"]), rtol=1e-6)
    base_total, base_parts = base.training_loss(key, obs, actions)
    assert base_parts == {} and np.array_equal(np.asarray(base_total), np.asarray(jnp.mean(base_loss)))

    serve_obs = _observation(2, 1, seed=6)
    noise = jnp.asarray(np.random.default_rng(8).normal(size=(2, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    base_actions, _ = base.sample_actions(key, serve_obs, num_steps=10, noise=noise, memory=_fresh_memory())
    race_actions, _, scores = race.sample_actions(key, serve_obs, num_steps=10, noise=noise, memory=_fresh_memory())
    assert np.array_equal(np.asarray(base_actions), np.asarray(race_actions)), np.abs(base_actions - race_actions).max()
    assert scores.shape == (2, HORIZON) and bool(((scores >= 0) & (scores <= 1)).all())


def test_the_prior_reaches_the_actions_once_trained(models):
    """(b) With the modulation weights moved off zero, the prior changes the actions and the loss."""
    base, _race, live = models
    serve_obs = _observation(2, 1, seed=6)
    noise = jnp.asarray(np.random.default_rng(8).normal(size=(2, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    key = jax.random.key(5)
    base_actions, _ = base.sample_actions(key, serve_obs, num_steps=10, noise=noise, memory=_fresh_memory())
    live_actions, _, _ = live.sample_actions(key, serve_obs, num_steps=10, noise=noise, memory=_fresh_memory())
    assert np.abs(np.asarray(base_actions) - np.asarray(live_actions)).max() > 1e-3

    # The same pass under two different priors: the conditioning, not some other path, moves it.
    prefix_obs = _model.preprocess_observation(None, serve_obs, train=False, image_keys=(CAMERA,))
    tokens, mask, ar_mask, _, _ = live.embed_prefix_infer(prefix_obs, None)
    _, kv_cache = live.PaliGemma.llm([tokens, None], mask=_pi0.make_attn_mask(mask, ar_mask),
                                     positions=jnp.cumsum(mask, axis=1) - 1)

    def velocity(prior):
        cond = None if prior is None else live.race_step_conditioning(prior, jnp.float32(1.0), 10)
        return np.asarray(live._suffix_pass(prefix_obs, mask, kv_cache, noise, jnp.ones((2, HORIZON)), cond)[0])

    unmodulated, with_zero, with_one = velocity(None), velocity(jnp.zeros((2, HORIZON))), velocity(jnp.ones((2, HORIZON)))
    assert np.abs(with_zero - with_one).max() > 1e-3
    assert np.array_equal(unmodulated, with_zero), "a zero prior leaves the expert unmodified (W_l has no bias)"

    window, inside = _targets(2)
    obs = _observation(2, HIST, window=window, window_mask=inside)
    actions = jnp.asarray(np.random.default_rng(4).normal(size=(2, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    assert not np.allclose(np.asarray(base.compute_loss(key, obs, actions, train=True)),
                           np.asarray(live.compute_loss(key, obs, actions, train=True)))


def test_timing_loss_skips_rows_past_the_episode_end(models):
    """(c) L_timing averages over the chunk rows inside the episode; what a masked row holds does not matter."""
    _base, race, _live = models
    logits = jnp.asarray(np.random.default_rng(9).normal(size=(2, HORIZON)), dtype=jnp.float32)
    window = jnp.asarray(np.random.default_rng(10).uniform(0, 1, size=(2, ROWS)), dtype=jnp.float32)
    # Sample 0's episode ends after chunk row 1; sample 1 has every row.
    inside = jnp.asarray([[True, True, True, False, False, False], [True] * ROWS])
    obs = dataclasses.replace(_observation(2, 1), transition_window=window, transition_window_mask=inside)
    loss = float(race._race_timing_loss(logits, obs))

    target, counted = np.asarray(window)[:, 1:HORIZON + 1], np.asarray(inside)[:, 1:HORIZON + 1]
    probs = 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))
    bce = -(target * np.log(probs) + (1 - target) * np.log(1 - probs))
    np.testing.assert_allclose(loss, bce[counted].mean(), rtol=1e-5)
    assert counted.sum() == 6

    scrambled = dataclasses.replace(obs, transition_window=window.at[0, 3:].set(7.0))
    assert float(race._race_timing_loss(logits, scrambled)) == pytest.approx(loss, rel=1e-6)


def test_teacher_prior_jitter_and_padding(models):
    """Teacher forcing: rows 1..H of the window, frames outside the episode as 0, jittered by +-1 w.p. 0.5."""
    _base, race, _live = models
    batch = 512
    window = jnp.tile(jnp.arange(ROWS, dtype=jnp.float32)[None, :] / 10, (batch, 1))
    inside = jnp.ones((batch, ROWS), dtype=bool).at[:, -1].set(False)
    obs = dataclasses.replace(_observation(1, 1), transition_window=window, transition_window_mask=inside)
    plain = np.asarray(race._race_teacher_prior(jax.random.key(0), obs, train=False))
    np.testing.assert_allclose(plain, np.tile(np.arange(1, HORIZON + 1) / 10, (batch, 1)), rtol=1e-6)

    jittered = np.asarray(race._race_teacher_prior(jax.random.key(1), obs, train=True))
    shift = np.round((plain[:, 0] - jittered[:, 0]) * 10).astype(int)   # row 0 reads window row 1 - shift
    assert set(np.unique(shift)) <= {-1, 0, 1}
    assert abs((shift != 0).mean() - 0.5) < 0.08, (shift != 0).mean()
    assert abs((shift == 1).mean() - (shift == -1).mean()) < 0.08
    late = shift == -1   # reads one row later: the last chunk row is the window's last, outside the episode
    assert late.any() and (jittered[late, -1] == 0).all()


def test_gate_index():
    """Inference gates by the step's start time; accumulated float error in the loop's time does not move it."""
    for num_steps, expected in ((10, list(range(10))), (4, [0, 2, 5, 7]), (5, [0, 2, 4, 6, 8])):
        time, gates = jnp.float32(1.0), []
        for _ in range(num_steps):
            gates.append(int(_pi0.race_gate_index(time, num_steps)))
            time = time + jnp.float32(-1.0 / num_steps)
        assert gates == expected, (num_steps, gates)
    training = _pi0.Pi0._race_gate_for_training(jnp.asarray([1.0, 0.91, 0.9, 0.55, 0.101, 0.001]))
    assert np.asarray(training).tolist() == [0, 0, 1, 4, 8, 9]


def test_infer_returns_transition_scores_for_race_only(models):
    """(f) Policy.infer's output carries transition_scores (H,) through Unnormalize and AgilexOutputs for a RACE
    model, and nothing new for a base model."""
    base, _race, live = models
    stats = _normalize.NormStats(mean=np.zeros(ACTION_DIM), std=np.ones(ACTION_DIM),
                                 q01=-np.ones(ACTION_DIM), q99=np.ones(ACTION_DIM))
    outputs = [_transforms.Unnormalize({"state": stats, "actions": stats}, use_quantiles=True),
               agilex_policy.AgilexOutputs()]
    rng = np.random.default_rng(11)
    request = {
        "image": {CAMERA: rng.uniform(-1, 1, size=(*_model.IMAGE_RESOLUTION, 3)).astype(np.float32)},
        "image_mask": {CAMERA: np.True_},
        "state": rng.normal(size=ACTION_DIM).astype(np.float32),
        "tokenized_prompt": rng.integers(0, 1000, size=TOKENS).astype(np.int32),
        "tokenized_prompt_mask": np.ones(TOKENS, dtype=bool),
        "step": 0,
    }
    race_out = _policy.Policy(live, output_transforms=outputs, sample_kwargs={"num_steps": 3}).infer(dict(request))
    scores = race_out[_transforms.TRANSITION_SCORES_KEY]
    assert scores.shape == (HORIZON,) and scores.dtype == np.float32
    assert ((scores >= 0) & (scores <= 1)).all()
    assert race_out["actions"].shape == (HORIZON, 14)

    base_out = _policy.Policy(base, output_transforms=outputs, sample_kwargs={"num_steps": 3}).infer(dict(request))
    assert _transforms.TRANSITION_SCORES_KEY not in base_out
    assert set(base_out) == {"actions", "policy_timing"}


def test_streaming_memory_is_unchanged(models):
    """(g) save/fetch_memory: a RACE model (even with live modulation) keeps the same streaming memory as the
    base model over consecutive calls -- the prior reads the prefix and never writes it."""
    base, _race, live = models
    memories = []
    for model in (base, live):
        memory, key = _fresh_memory(), jax.random.key(12)
        for step in range(HIST):
            sampled = model.sample_actions(key, _observation(1, 1, seed=20 + step), num_steps=2, memory=memory)
            memory = sampled[1]
        memories.append(memory)
    base_memory, race_memory = memories
    assert set(base_memory) == set(race_memory)
    for name in ("memory_tokens", "memory_prefix_mask"):
        assert np.array_equal(np.asarray(base_memory[name]), np.asarray(race_memory[name])), name
    for base_cache, race_cache in zip(base_memory["memory_kv_cache"], race_memory["memory_kv_cache"], strict=True):
        assert np.array_equal(np.asarray(base_cache), np.asarray(race_cache))


def test_freeze_filter_keeps_race_weights_trainable():
    """A LoRA recipe freezes the LLM; RACE's modulation lives inside it and must still train."""
    config = dataclasses.replace(_config(race=True), paligemma_variant="gemma_2b_lora",
                                 action_expert_variant="gemma_300m_lora")
    frozen = nnx.filterlib.to_predicate(config.get_freeze_filter())
    param = nnx.Param(jnp.zeros(()))
    assert not frozen(("PaliGemma", "llm", "layers", "pre_ffw_norm_1", "race_modulation", "kernel"), param)
    assert not frozen(("race_head_cross", "query", "kernel"), param)
    assert frozen(("PaliGemma", "llm", "layers", "pre_ffw_norm_1", "Dense_0", "kernel"), param)
    base = dataclasses.replace(config, race=False)
    assert nnx.filterlib.to_predicate(base.get_freeze_filter())(
        ("PaliGemma", "llm", "layers", "pre_ffw_norm_1", "Dense_0", "kernel"), param)


def test_config_rules():
    with pytest.raises(ValueError, match="pi0"):
        _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", race=True)
    with pytest.raises(ValueError, match="vlash_branches"):
        _config(race=True, state_cond=True, vlash_branches=2)
    assert _config(race=True).race_aux_weight == 0.1 and _config(race=True).race_timing_weight == 0.05


def test_jitted_sampler_returns_scores(models):
    """The serve path jits sample_actions (nnx_utils.module_jit): the three-way return survives it."""
    _base, _race, live = models
    sample = nnx_utils.module_jit(live.sample_actions)
    actions, memory, scores = sample(jax.random.key(0), _observation(1, 1, seed=30), num_steps=2,
                                     memory=_fresh_memory())
    assert actions.shape == (1, HORIZON, ACTION_DIM) and scores.shape == (1, HORIZON)
    assert memory["memory_kv_cache"] is not None


def test_step0_equals_the_base_model_in_bfloat16():
    """(a) again in the dtype the YAM recipes train and serve in."""
    base = _config(dtype="bfloat16").create(jax.random.key(0))
    base_params = _randomised(_params(base), "Dense_0", seed=1)
    base = _with_params(base, base_params)
    race = _config(dtype="bfloat16", race=True).create(jax.random.key(7))
    race = _with_params(race, weight_loaders._merge_params(base_params, _params(race),
                                                           missing_regex=_pi0_config.RACE_PARAMS_REGEX))
    window, inside = _targets(2)
    obs = _observation(2, HIST, window=window, window_mask=inside)
    actions = jnp.asarray(np.random.default_rng(4).normal(size=(2, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    key = jax.random.key(5)
    base_loss = base.compute_loss(key, obs, actions, train=True)
    race_loss = race.compute_loss(key, obs, actions, train=True)
    assert np.array_equal(np.asarray(base_loss), np.asarray(race_loss)), np.abs(base_loss - race_loss).max()
    noise = jnp.asarray(np.random.default_rng(8).normal(size=(2, HORIZON, ACTION_DIM)), dtype=jnp.float32)
    base_actions, _ = base.sample_actions(key, _observation(2, 1, seed=6), num_steps=10, noise=noise,
                                          memory=_fresh_memory())
    race_actions, _, _ = race.sample_actions(key, _observation(2, 1, seed=6), num_steps=10, noise=noise,
                                             memory=_fresh_memory())
    assert np.array_equal(np.asarray(base_actions), np.asarray(race_actions)), np.abs(base_actions - race_actions).max()
