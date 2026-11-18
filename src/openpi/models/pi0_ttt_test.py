"""TTT layers (arXiv 2407.04620) on pi0.5: a model loaded over a checkpoint without them starts as that
checkpoint, the layers reach the loss and the actions once trained, and a served stream of single-frame calls
steps the same fast weights as the training pass over the same frames. Dummy gemma variants, float32, CPU."""

import os
import subprocess
import sys

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.models import pi0_testing
from openpi.models import ttt as _ttt
from openpi.training import weight_loaders

ACTION_DIM, HORIZON, TOKENS, HIST = 14, 4, 16, 3
CAMERA = "left_wrist_0_rgb"
TTT_WIDTH = 2 * _ttt.TTT_HEAD_DIM
DEPTH = 4   # the dummy expert's
EXACT_BFLOAT16_FLAG = "--xla_allow_excess_precision=false"


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


def _with_live_output(model, seed: int = 2, scale: float = 0.5):
    """`model` with every TTT layer's zero output projection replaced by N(0, scale^2) draws."""
    flat = traverse_util.flatten_dict(_params(model), sep="/")
    rng = np.random.default_rng(seed)
    for key, value in flat.items():
        if key.endswith(f"{_gemma.TTT_LAYER_NAME}/out_proj/kernel"):
            flat[key] = jnp.asarray(rng.normal(scale=scale, size=value.shape), dtype=value.dtype)
    return _with_params(model, traverse_util.unflatten_dict(flat, sep="/"))


def _observation(batch: int, frames: int, *, seed: int = 0) -> _model.Observation:
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
    )


def _actions(batch: int, seed: int = 4):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(batch, HORIZON, ACTION_DIM)), dtype=jnp.float32)


def _fresh_memory(model) -> dict:
    memory = {}
    model.reset_memory(memory)
    return memory


@pytest.fixture(scope="module")
def models():
    """(base, ttt at its initialisation, ttt with its output projection moved off zero): the ttt models are
    built the way training builds them -- fresh weights, then the base checkpoint merged in by
    CheckpointWeightLoader's rule, which lets the TTT weights be missing."""
    base = pi0_testing.awaken(_config().create(jax.random.key(0)), jax.random.key(1))
    ttt = _config(ttt_width=TTT_WIDTH).create(jax.random.key(7))
    merged = weight_loaders._merge_params(_params(base), _params(ttt), missing_regex=_pi0_config.TTT_PARAMS_REGEX)
    ttt = _with_params(ttt, merged)
    return base, ttt, _with_live_output(ttt)


def test_ttt_weights_sit_in_every_vlm_block_and_nowhere_else(models):
    base, ttt, _live = models
    flat = traverse_util.flatten_dict(_params(ttt), sep="/")
    layer = f"PaliGemma/llm/layers/{_gemma.TTT_LAYER_NAME}"
    heads, width = TTT_WIDTH // _ttt.TTT_HEAD_DIM, 64   # the dummy VLM's width
    assert flat[f"{layer}/w_init"].shape == (DEPTH, heads, _ttt.TTT_HEAD_DIM, _ttt.TTT_HEAD_DIM)
    assert flat[f"{layer}/q_proj/kernel"].shape == (DEPTH, width, TTT_WIDTH)
    assert flat[f"{layer}/out_proj/kernel"].shape == (DEPTH, TTT_WIDTH, width)
    assert not np.asarray(flat[f"{layer}/out_proj/kernel"]).any()
    new = set(flat) - set(traverse_util.flatten_dict(_params(base), sep="/"))
    assert new and all(key.startswith(layer + "/") for key in new), sorted(new)


def test_step0_equals_the_base_model(models):
    base, ttt, _live = models
    obs, actions, key = _observation(2, HIST), _actions(2), jax.random.key(5)
    base_loss = base.compute_loss(key, obs, actions, train=True)
    ttt_loss = ttt.compute_loss(key, obs, actions, train=True)
    assert np.array_equal(np.asarray(base_loss), np.asarray(ttt_loss)), np.abs(base_loss - ttt_loss).max()

    noise = _actions(2, seed=8)
    base_memory, ttt_memory = _fresh_memory(base), _fresh_memory(ttt)
    for step in range(HIST):
        serve_obs = _observation(2, 1, seed=10 + step)
        base_actions, base_memory = base.sample_actions(key, serve_obs, num_steps=4, noise=noise, memory=base_memory)
        ttt_actions, ttt_memory = ttt.sample_actions(key, serve_obs, num_steps=4, noise=noise, memory=ttt_memory)
        assert np.array_equal(np.asarray(base_actions), np.asarray(ttt_actions)), step
    assert _pi0.TTT_STATE_KEY not in base_memory


def test_the_layers_reach_the_loss_and_the_actions_once_trained(models):
    base, _ttt_model, live = models
    obs, actions, key = _observation(2, HIST), _actions(2), jax.random.key(5)
    assert not np.allclose(np.asarray(base.compute_loss(key, obs, actions, train=True)),
                           np.asarray(live.compute_loss(key, obs, actions, train=True)))
    noise, serve_obs = _actions(2, seed=8), _observation(2, 1, seed=10)
    base_actions, _ = base.sample_actions(key, serve_obs, num_steps=4, noise=noise, memory=_fresh_memory(base))
    live_actions, _ = live.sample_actions(key, serve_obs, num_steps=4, noise=noise, memory=_fresh_memory(live))
    assert np.abs(np.asarray(base_actions) - np.asarray(live_actions)).max() > 1e-3


def _joint_prefix_pass(model, obs: _model.Observation):
    """The training pass's prefix alone: every history frame at once, block-causal over frames."""
    obs = _model.preprocess_observation(None, obs, train=False, image_keys=(CAMERA,))
    tokens, mask, ar_mask, _frame_len, frames = model.embed_prefix(obs)
    return model.PaliGemma.llm([tokens, None], mask=_pi0.make_attn_mask(mask, ar_mask),
                               positions=jnp.cumsum(mask, axis=1) - 1, ttt_mask=model._ttt_mask(mask, frames))


def _streamed(model, obs: _model.Observation, frames: range, memory: dict) -> dict:
    """Serve calls over `frames` of a training observation, one frame each, the memory carried."""
    for frame in frames:
        frame_obs = _model.Observation(
            images={CAMERA: obs.images[CAMERA][:, frame : frame + 1]}, image_masks=obs.image_masks, state=obs.state,
            tokenized_prompt=obs.tokenized_prompt, tokenized_prompt_mask=obs.tokenized_prompt_mask)
        _, memory = model.sample_actions(jax.random.key(0), frame_obs, num_steps=1, noise=_actions(2), memory=memory)
    return memory


def test_a_served_stream_steps_the_fast_weights_the_training_pass_does(models):
    _base, _ttt_model, live = models
    obs = _observation(2, HIST)
    _outputs, joint_cache, joint_state = _joint_prefix_pass(live, obs)
    memory = _streamed(live, obs, range(HIST), _fresh_memory(live))

    weights, biases = memory[_pi0.TTT_STATE_KEY]
    assert weights.shape == (DEPTH, 2, TTT_WIDTH // _ttt.TTT_HEAD_DIM, _ttt.TTT_HEAD_DIM, _ttt.TTT_HEAD_DIM)
    np.testing.assert_allclose(weights, joint_state[0], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(biases, joint_state[1], rtol=1e-4, atol=1e-5)
    # Padding tokens are never attended to, and their own cached rows differ between the two passes in the
    # base model too.
    valid = np.asarray(memory["memory_prefix_mask"])
    for streamed, joint in zip(memory["memory_kv_cache"], joint_cache, strict=True):
        np.testing.assert_allclose(np.asarray(streamed)[:, valid], np.asarray(joint)[:, valid], rtol=1e-3, atol=1e-4)

    # The weights moved with every frame, and a reset returns them to the learned start.
    one_frame = _streamed(live, obs, range(1), _fresh_memory(live))[_pi0.TTT_STATE_KEY]
    assert np.abs(np.asarray(one_frame[0]) - np.asarray(weights)).max() > 1e-6
    live.reset_memory(memory)
    assert memory[_pi0.TTT_STATE_KEY] is None and memory["memory_kv_cache"] is None


def test_a_dropped_history_frame_is_a_stream_that_starts_later(models):
    """Training's history drop masks the oldest frames; the fast weights then equal a served stream that
    began after them."""
    _base, _ttt_model, live = models
    obs = _observation(2, HIST)
    prefix_obs = _model.preprocess_observation(None, obs, train=False, image_keys=(CAMERA,))
    tokens, mask, ar_mask, frame_len, frames = live.embed_prefix(prefix_obs)
    mask = mask & (jnp.arange(mask.shape[1]) >= frame_len)[None, :]
    _, _, dropped_state = live.PaliGemma.llm(
        [tokens, None], mask=_pi0.make_attn_mask(mask, ar_mask), positions=jnp.cumsum(mask, axis=1) - 1,
        ttt_mask=live._ttt_mask(mask, frames))
    later = _streamed(live, obs, range(1, HIST), _fresh_memory(live))[_pi0.TTT_STATE_KEY]
    for dropped, streamed in zip(dropped_state, later, strict=True):
        np.testing.assert_allclose(dropped, streamed, rtol=1e-4, atol=1e-5)


def test_the_loss_gradient_is_finite_and_reaches_the_ttt_weights(models):
    _base, ttt, live = models
    obs, actions = _observation(2, HIST), _actions(2)

    def grads_of(model):
        def loss(model):
            return jnp.mean(model.compute_loss(jax.random.key(5), obs, actions, train=True))
        flat = traverse_util.flatten_dict(nnx.grad(loss)(model).to_pure_dict(), sep="/")
        return {key: np.asarray(value) for key, value in flat.items() if _gemma.TTT_LAYER_NAME in key}

    at_init, once_live = grads_of(ttt), grads_of(live)
    assert at_init and all(np.isfinite(grad).all() for grad in at_init.values())
    assert np.abs(at_init[f"PaliGemma/llm/layers/{_gemma.TTT_LAYER_NAME}/out_proj/kernel"]).max() > 0, \
        "the zero output projection is what the first step moves"
    for key, grad in once_live.items():
        assert np.isfinite(grad).all() and np.abs(grad).max() > 0, key


def test_freeze_filter_keeps_ttt_weights_trainable():
    def frozen(config):
        model = nnx.eval_shape(config.create, jax.random.key(0))
        return set(nnx.state(model, nnx.All(nnx.Param, config.get_freeze_filter())).flat_state())

    lora = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora"}
    with_ttt = frozen(_pi0_config.Pi0Config(pi05=True, ttt_width=TTT_WIDTH, **lora))
    assert with_ttt == frozen(_pi0_config.Pi0Config(pi05=True, **lora))
    assert not any(_gemma.TTT_LAYER_NAME in str(path) for path in with_ttt)


def test_config_rules():
    with pytest.raises(ValueError, match="multiple of"):
        _config(ttt_width=100)
    with pytest.raises(ValueError, match="vlash_branches"):
        _config(ttt_width=TTT_WIDTH, state_cond=True, vlash_branches=3)
    assert not _config().create(jax.random.key(0)).ttt


def test_jitted_sampler_carries_the_state(models):
    _base, _ttt_model, live = models
    sample = nnx.jit(lambda model, rng, obs, noise, memory: model.sample_actions(
        rng, obs, num_steps=2, noise=noise, memory=memory))
    memory = _fresh_memory(live)
    for step in range(2):
        actions, memory = sample(live, jax.random.key(0), _observation(2, 1, seed=step), _actions(2), memory)
        assert np.isfinite(np.asarray(actions)).all()
    assert memory[_pi0.TTT_STATE_KEY][0].shape[0] == DEPTH


def test_step0_equals_the_base_model_in_bfloat16():
    """The dtype the YAM recipes train and serve in. XLA's CPU backend computes bfloat16 in float32 and, by
    default, drops a rounding step wherever the graph lets it (xla_allow_excess_precision), so a branch that
    adds exact zeros still moves where the base model rounds. With that off the two models agree bit for bit,
    so the comparison runs in a process that turns it off."""
    if EXACT_BFLOAT16_FLAG not in os.environ.get("XLA_FLAGS", ""):
        env = {**os.environ, "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} {EXACT_BFLOAT16_FLAG}".strip()}
        rerun = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", "-q", "-x", f"{__file__}::test_step0_equals_the_base_model_in_bfloat16"],
            env=env, capture_output=True, text=True, check=False)
        assert rerun.returncode == 0, rerun.stdout[-3000:]
        return
    base = pi0_testing.awaken(_config(dtype="bfloat16").create(jax.random.key(0)), jax.random.key(1))
    ttt = _config(dtype="bfloat16", ttt_width=TTT_WIDTH).create(jax.random.key(7))
    ttt = _with_params(ttt, weight_loaders._merge_params(_params(base), _params(ttt),
                                                         missing_regex=_pi0_config.TTT_PARAMS_REGEX))
    obs, actions, key = _observation(2, HIST), _actions(2), jax.random.key(5)
    base_loss, ttt_loss = (model.compute_loss(key, obs, actions, train=True) for model in (base, ttt))
    assert np.array_equal(np.asarray(base_loss), np.asarray(ttt_loss))
    live_loss = _with_live_output(ttt).compute_loss(key, obs, actions, train=True)
    assert np.isfinite(np.asarray(live_loss)).all() and not np.array_equal(np.asarray(live_loss), np.asarray(base_loss))
