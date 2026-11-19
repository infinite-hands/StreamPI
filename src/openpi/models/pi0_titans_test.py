"""Titans' long-term memory (arXiv 2501.00663) on pi0.5: with its gates closed the model is the checkpoint it was
loaded over; the memory training unrolls is the memory serving builds call by call; only a new history clears it.
Dummy gemma variants, float32 unless a test says otherwise, CPU.

A freshly initialised pi0.5 ignores its prefix (pi0_testing.awaken), so the fixtures randomise those kernels first, or
"the memory changes nothing" would hold trivially."""

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
from openpi.models import pi0_testing
import openpi.models.gemma as _gemma
from openpi.policies import agilex_policy
from openpi.policies import policy as _policy
from openpi.shared import normalize as _normalize
from openpi.training import weight_loaders

ACTION_DIM, HORIZON, TOKENS, HIST = 14, 4, 16, 2
CAMERA = "base_0_rgb"
# The memory under test: frames unrolled in training, calls between two writes, cells per side, widths.
FRAMES, STRIDE, GRID, DIM, HIDDEN = 3, 2, 2, 16, 24


def _config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(**{
        "paligemma_variant": "dummy", "action_expert_variant": "dummy", "pi05": True, "action_dim": ACTION_DIM,
        "action_horizon": HORIZON, "hist_horizon": HIST, "max_token_len": TOKENS, "dtype": "float32",
        "image_keys": (CAMERA,), **overrides})


def _titans_config(**overrides) -> _pi0_config.Pi0Config:
    return _config(**{"titans_frames": FRAMES, "titans_stride": STRIDE, "titans_grid": GRID, "titans_dim": DIM,
                      "titans_hidden_dim": HIDDEN, "titans_image_key": CAMERA, "titans_dropout": 0.0, **overrides})


def _params(model) -> dict:
    return nnx.state(model).to_pure_dict()


def _with_params(model, params: dict):
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    return nnx.merge(graphdef, state)


def _with_gates(model, value: float):
    """`model` with every layer's memory-read gate set to `value` (tanh(value) scales the read)."""
    flat = traverse_util.flatten_dict(_params(model), sep="/")
    gates = [key for key in flat if key.endswith("memory_gate")]
    assert gates
    for key in gates:
        flat[key] = jnp.full(flat[key].shape, value, flat[key].dtype)
    return _with_params(model, traverse_util.unflatten_dict(flat, sep="/"))


def _over_base(config: _pi0_config.Pi0Config, base_params: dict, seed: int = 7):
    """A memory model built the way training builds it: fresh weights, then the base checkpoint merged in by
    CheckpointWeightLoader's rule, which lets the memory's weights be missing."""
    fresh = config.create(jax.random.key(seed))
    merged = weight_loaders._merge_params(base_params, _params(fresh), missing_regex=_pi0_config.TITANS_PARAMS_REGEX)
    return _with_params(fresh, merged)


def _pictures(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, size=(count, *_model.IMAGE_RESOLUTION, 3)).astype(np.float32)


def _observation(window: np.ndarray, older: np.ndarray | None = None, older_mask: np.ndarray | None = None, *,
                 seed: int = 0, camera: bool = True) -> _model.Observation:
    """`window` (batch, frames, h, w, 3) is the streaming window, its last frame the current one; `older` (batch,
    memory frames, h, w, 3) the memory's frames, newest first, with `older_mask` saying which of them exist."""
    rng = np.random.default_rng(seed)
    batch = window.shape[0]
    return _model.Observation(
        images={CAMERA: jnp.asarray(window)},
        image_masks={CAMERA: jnp.full((batch,), camera, dtype=bool)},
        state=jnp.asarray(rng.normal(size=(batch, ACTION_DIM)), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray(rng.integers(0, 1000, size=(batch, TOKENS)), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray(np.arange(TOKENS)[None, :] < TOKENS - 3).repeat(batch, 0),
        memory_images=None if older is None else jnp.asarray(older),
        memory_mask=None if older_mask is None else jnp.asarray(older_mask),
    )


def _training_observation(batch: int = 2, seed: int = 0, *, older_seed: int = 50, mask: np.ndarray | None = None):
    window = np.stack([_pictures(HIST, seed + index) for index in range(batch)])
    older = np.stack([_pictures(FRAMES, older_seed + index) for index in range(batch)])
    mask = np.ones((batch, FRAMES), dtype=bool) if mask is None else mask
    return _observation(window, older, mask, seed=seed)


def _serve_observation(picture: np.ndarray, *, seed: int = 0, camera: bool = True) -> _model.Observation:
    return _observation(picture[None, None], seed=seed, camera=camera)


def _fresh_memory() -> dict:
    return dict(memory_tokens=None, memory_kv_cache=None, memory_prefix_mask=None)


def _preprocessed(model, observation: _model.Observation) -> _model.Observation:
    return _model.preprocess_observation(None, observation, train=False, image_keys=model.image_keys)


def _actions(batch: int = 2, seed: int = 4):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(batch, HORIZON, ACTION_DIM)), dtype=jnp.float32)


@pytest.fixture(scope="module")
def models():
    """(base, the memory model at its initialisation over the base checkpoint, the same with its gates open)."""
    base = pi0_testing.awaken(_config().create(jax.random.key(0)), jax.random.key(1))
    closed = _over_base(_titans_config(), _params(base))
    return base, closed, _with_gates(closed, 1.0)


def test_the_memory_weights_exist_and_its_gates_start_closed(models):
    base, closed, _live = models
    flat = traverse_util.flatten_dict(_params(closed), sep="/")
    gates = [key for key in flat if key.endswith("memory_gate")]
    assert [np.asarray(flat[key]).shape for key in gates] == [(_gemma.get_config("dummy").depth,)], "one gate per layer"
    assert not np.asarray(flat[gates[0]]).any()
    assert flat["titans_memory/initial_weights/layer_0"].shape == (DIM, HIDDEN)
    assert flat["titans_memory/initial_weights/layer_1"].shape == (HIDDEN, DIM)
    assert flat["titans_memory/position"].shape == (GRID * GRID, DIM)
    read = [key for key in flat if "/memory_k/" in key or "/memory_v/" in key]
    assert read and all(np.asarray(flat[key]).any() for key in read), "only the gate starts at zero"
    base_flat = traverse_util.flatten_dict(_params(base), sep="/")
    assert not any("titans_" in key or "memory_" in key for key in base_flat)
    assert set(flat) - set(base_flat) == {key for key in flat if "titans_" in key or "memory_" in key}


def test_step0_equals_the_base_model(models):
    """The memory at its initialisation, loaded over a checkpoint without it: that checkpoint's loss and actions."""
    base, closed, _live = models
    observation = _training_observation()
    actions, key = _actions(), jax.random.key(5)
    for train in (True, False):
        base_loss = base.compute_loss(key, observation, actions, train=train)
        closed_loss = closed.compute_loss(key, observation, actions, train=train)
        assert np.array_equal(np.asarray(base_loss), np.asarray(closed_loss)), np.abs(base_loss - closed_loss).max()

    noise = _actions(1, seed=8)
    base_memory, closed_memory = _fresh_memory(), _fresh_memory()
    for call in range(STRIDE + 2):   # past the first write, so the memory is no longer empty
        served = _serve_observation(_pictures(1, 20 + call)[0], seed=call)
        base_actions, base_memory = base.sample_actions(key, served, num_steps=3, noise=noise, memory=base_memory)
        closed_actions, closed_memory = closed.sample_actions(key, served, num_steps=3, noise=noise,
                                                              memory=closed_memory)
        assert np.array_equal(np.asarray(base_actions), np.asarray(closed_actions)), call
    assert bool(closed_memory[_pi0.LONG_MEMORY_KEY]["state"]["written"].all())


def test_a_second_camera_is_encoded_and_the_memory_reads_only_its_own():
    """The recipes this is for show the model three cameras and give the memory one: the encoded tokens are handed to
    the prefix for every camera, and step 0 is still the base model."""
    cameras = (CAMERA, "right_wrist_0_rgb")
    base = pi0_testing.awaken(_config(image_keys=cameras).create(jax.random.key(0)), jax.random.key(1))
    closed = _over_base(_titans_config(image_keys=cameras), _params(base))
    rng = np.random.default_rng(3)
    window = np.stack([_pictures(HIST, 0)])
    older = np.stack([_pictures(FRAMES, 50)])
    observation = _observation(window, older, np.ones((1, FRAMES), dtype=bool))
    observation = dataclasses.replace(
        observation,
        images={**observation.images, cameras[1]: jnp.asarray(rng.uniform(-1, 1, size=(1, HIST, *_model.IMAGE_RESOLUTION, 3)),
                                                              dtype=jnp.float32)},
        image_masks={**observation.image_masks, cameras[1]: jnp.ones((1,), dtype=bool)})
    actions, key = _actions(1), jax.random.key(5)
    base_loss = base.compute_loss(key, observation, actions, train=True)
    closed_loss = closed.compute_loss(key, observation, actions, train=True)
    assert np.array_equal(np.asarray(base_loss), np.asarray(closed_loss))
    served = _observation(window[:, -1:], seed=2)
    served = dataclasses.replace(served, images={**served.images, cameras[1]: observation.images[cameras[1]][:, -1:]},
                                 image_masks={**served.image_masks, cameras[1]: jnp.ones((1,), dtype=bool)})
    memory = _fresh_memory()
    actions_out, memory = _with_gates(closed, 1.0).sample_actions(key, served, num_steps=2, memory=memory)
    assert np.isfinite(np.asarray(actions_out)).all() and bool(memory[_pi0.LONG_MEMORY_KEY]["state"]["written"].all())


def test_step0_equals_the_base_model_in_bfloat16():
    """The same in the dtype the YAM recipes train and serve in."""
    base = pi0_testing.awaken(_config(dtype="bfloat16").create(jax.random.key(0)), jax.random.key(1))
    closed = _over_base(_titans_config(dtype="bfloat16"), _params(base))
    observation = _training_observation()
    actions, key = _actions(), jax.random.key(5)
    base_loss = base.compute_loss(key, observation, actions, train=True)
    closed_loss = closed.compute_loss(key, observation, actions, train=True)
    assert np.array_equal(np.asarray(base_loss), np.asarray(closed_loss)), np.abs(base_loss - closed_loss).max()
    # Open, the read still reaches a bf16 model and leaves it bf16.
    live_loss = _with_gates(closed, 1.0).compute_loss(key, observation, actions, train=True)
    assert np.isfinite(np.asarray(live_loss, dtype=np.float32)).all()
    assert not np.array_equal(np.asarray(base_loss), np.asarray(live_loss))


def test_open_gates_let_what_the_memory_holds_reach_the_loss(models):
    base, _closed, live = models
    actions, key = _actions(), jax.random.key(5)
    observation = _training_observation()
    base_loss = np.asarray(base.compute_loss(key, observation, actions, train=False))
    live_loss = np.asarray(live.compute_loss(key, observation, actions, train=False))
    assert not np.allclose(base_loss, live_loss, atol=1e-6)
    # Other frames in the memory, the same window: another loss.
    other = np.asarray(live.compute_loss(key, _training_observation(older_seed=70), actions, train=False))
    assert not np.allclose(live_loss, other, atol=1e-6)
    # A sample none of whose older frames exist reads nothing, whatever the gates: it is the base model's sample.
    mask = np.ones((2, FRAMES), dtype=bool)
    mask[1] = False
    partly = np.asarray(live.compute_loss(key, _training_observation(mask=mask), actions, train=False))
    assert np.array_equal(partly[1], base_loss[1])
    assert not np.allclose(partly[0], base_loss[0], atol=1e-6)


def test_the_served_memory_is_the_one_training_unrolls(models):
    """Serving reads the memory, then writes the current frame on every STRIDE-th call. At a write call n * STRIDE
    the memory holds the frames of calls 0, STRIDE, ..., (n - 1) * STRIDE: the layout the loader gives a training
    sample (older frames STRIDE calls apart, newest first, absent ones masked). The two banks must agree."""
    _base, _closed, live = models
    calls = FRAMES * STRIDE + 1
    pictures = _pictures(calls, seed=100)
    memory = _fresh_memory()
    live.reset_memory(memory)
    served = []
    for call in range(calls):
        observation = _preprocessed(live, _serve_observation(pictures[call], seed=call))
        bank, memory = live.step_long_memory(observation, memory, live.image_tokens(observation))
        served.append(bank)
    assert int(memory[_pi0.LONG_MEMORY_KEY]["calls"]) == calls
    assert not np.asarray(served[0][1]).any(), "the first call reads an empty memory"

    for writes in range(1, FRAMES + 1):
        call = writes * STRIDE
        older = np.zeros((1, FRAMES, *pictures.shape[1:]), dtype=np.float32)
        mask = np.zeros((1, FRAMES), dtype=bool)
        for slot in range(writes):   # slot 0 is the newest older frame: STRIDE calls back
            older[0, slot] = pictures[call - (slot + 1) * STRIDE]
            mask[0, slot] = True
        window = np.stack([pictures[max(call - 1, 0)], pictures[call]])[None]   # HIST frames, the last one current
        observation = _preprocessed(live, _observation(window, older, mask))
        tokens, counted = live.training_memory_bank(
            jax.random.key(0), observation, live.image_tokens(observation), train=False)
        np.testing.assert_allclose(np.asarray(tokens), np.asarray(served[call][0]), atol=2e-5, err_msg=str(writes))
        assert np.array_equal(np.asarray(counted), np.asarray(served[call][1]))
        assert np.asarray(counted).all()


def _request(step: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "image": {CAMERA: rng.uniform(-1, 1, size=(*_model.IMAGE_RESOLUTION, 3)).astype(np.float32)},
        "image_mask": {CAMERA: np.True_},
        "state": rng.normal(size=ACTION_DIM).astype(np.float32),
        "tokenized_prompt": rng.integers(0, 1000, size=TOKENS).astype(np.int32),
        "tokenized_prompt_mask": np.ones(TOKENS, dtype=bool),
        "step": step,
    }


def _served_policy(model) -> _policy.Policy:
    stats = _normalize.NormStats(mean=np.zeros(ACTION_DIM), std=np.ones(ACTION_DIM),
                                 q01=-np.ones(ACTION_DIM), q99=np.ones(ACTION_DIM))
    outputs = [_transforms.Unnormalize({"state": stats, "actions": stats}, use_quantiles=True),
               agilex_policy.AgilexOutputs()]
    return _policy.Policy(model, output_transforms=outputs, sample_kwargs={"num_steps": 2})


def test_only_a_new_history_clears_the_long_memory(models):
    """Policy.infer empties the streaming KV cache every hist_horizon calls; the long memory must outlive that and
    be cleared only when the step counter is back at 0."""
    _base, _closed, live = models
    policy = _served_policy(live)
    steps = 2 * HIST + 1   # two KV resets inside the run
    for step in range(steps):
        out = policy.infer(_request(step, seed=40 + step))
        assert out["actions"].shape == (HORIZON, 14)
    held = policy.memory[_pi0.LONG_MEMORY_KEY]
    assert int(held["calls"]) == steps
    assert bool(held["state"]["written"].all())
    assert any(np.asarray(layer).any() for layer in held["state"]["deviation"])

    policy.infer(_request(0, seed=99))   # the box reset its history: a new run
    held = policy.memory[_pi0.LONG_MEMORY_KEY]
    assert int(held["calls"]) == 1
    # One write since the reset: exactly what a policy that never served the earlier run holds after the same call.
    fresh = _served_policy(live)
    fresh.infer(_request(0, seed=99))
    for kept, expected in zip(held["state"]["deviation"], fresh.memory[_pi0.LONG_MEMORY_KEY]["state"]["deviation"],
                              strict=True):
        np.testing.assert_allclose(np.asarray(kept), np.asarray(expected), atol=1e-6)


def test_a_camera_that_is_not_there_neither_writes_nor_reads(models):
    _base, _closed, live = models
    memory = _fresh_memory()
    live.reset_memory(memory)
    blind = _preprocessed(live, _serve_observation(_pictures(1, 3)[0], camera=False))
    bank, memory = live.step_long_memory(blind, memory, live.image_tokens(blind))
    assert not np.asarray(bank[1]).any()
    assert not bool(memory[_pi0.LONG_MEMORY_KEY]["state"]["written"].any())
    assert not any(np.asarray(layer).any() for layer in memory[_pi0.LONG_MEMORY_KEY]["state"]["deviation"])
    assert int(memory[_pi0.LONG_MEMORY_KEY]["calls"]) == 1

    # In training a masked camera's older frames are not written either: the sample reads nothing.
    window = np.stack([_pictures(HIST, 0)])
    older = np.stack([_pictures(FRAMES, 50)])
    observation = _preprocessed(live, _observation(window, older, np.ones((1, FRAMES), dtype=bool), camera=False))
    _tokens, counted = live.training_memory_bank(
        jax.random.key(0), observation, live.image_tokens(observation), train=False)
    assert not np.asarray(counted).any()


def test_the_memory_adds_no_input_structure_to_the_served_sampler(models):
    """The serve path jits sample_actions and warms it over every KV-cache shape before the socket opens. The long
    memory must not add a shape of its own, or a live call would stall on a compile: over the same calls a memory
    model traces exactly as often as the base model."""
    base, _closed, live = models
    traces = []
    for model in (base, live):
        graphdef, state = nnx.split(model)
        sample = jax.jit(lambda state, key, observation, memory, graphdef=graphdef: nnx.merge(
            graphdef, state).sample_actions(key, observation, num_steps=2, memory=memory))
        memory = _fresh_memory()
        for step in range(2 * HIST + 1):
            if step % HIST == 0:
                model.reset_memory(memory)   # what Policy.infer does
            _actions_out, memory = sample(state, jax.random.key(step), _serve_observation(_pictures(1, step)[0]),
                                          memory)
        traces.append(sample._cache_size())
    assert traces[0] == traces[1] == HIST, traces


def test_training_cuts_some_memories_short():
    """Training-only: with probability `rate` a sample keeps only its newest n older frames, n uniform in 0..K-1,
    as the first calls of a served run see the memory."""
    valid = jnp.ones((400, 6), dtype=bool).at[:, 5].set(False)   # newest first; the oldest slot does not exist
    assert np.array_equal(np.asarray(_pi0.cut_memory_short(jax.random.key(0), valid, 0.0)), np.asarray(valid))
    cut = np.asarray(_pi0.cut_memory_short(jax.random.key(0), valid, 0.5))
    assert not (cut & ~np.asarray(valid)).any(), "nothing absent comes back"
    kept = cut.sum(axis=1)
    assert all(row[:count].all() and not row[count:].any() for row, count in zip(cut, kept, strict=True)), \
        "what stays is the newest frames"
    untouched = (cut == np.asarray(valid)).all(axis=1)
    assert 0.3 < untouched.mean() < 0.75   # about half are left whole (a cut at n = 5 also leaves the row whole)
    assert set(kept[~untouched]) == set(range(5)), "every shorter length occurs, the empty memory included"


def test_a_memory_model_refuses_a_training_sample_without_older_frames(models):
    _base, closed, _live = models
    window = np.stack([_pictures(HIST, 0)])
    with pytest.raises(ValueError, match="older frames"):
        closed.compute_loss(jax.random.key(0), _observation(window), _actions(1), train=False)
    wrong = _observation(window, np.stack([_pictures(FRAMES + 1, 50)]), np.ones((1, FRAMES + 1), dtype=bool))
    with pytest.raises(ValueError, match="older frames"):
        closed.compute_loss(jax.random.key(0), wrong, _actions(1), train=False)


def test_freeze_filter_keeps_the_memory_trainable():
    """A LoRA recipe freezes the LLM; the memory's read lives inside it and must still train."""
    config = dataclasses.replace(_titans_config(), paligemma_variant="gemma_2b_lora",
                                 action_expert_variant="gemma_300m_lora")
    frozen = nnx.filterlib.to_predicate(config.get_freeze_filter())
    param = nnx.Param(jnp.zeros(()))
    for name in ("memory_k", "memory_v"):
        assert not frozen(("PaliGemma", "llm", "layers", "attn", name, "kernel"), param)
    assert not frozen(("PaliGemma", "llm", "layers", "attn", "memory_gate"), param)
    assert not frozen(("titans_memory", "key_proj"), param)
    assert not frozen(("titans_memory", "initial_weights", "layer_0"), param)
    assert frozen(("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"), param)
    base = dataclasses.replace(config, titans_frames=0)
    assert nnx.filterlib.to_predicate(base.get_freeze_filter())(
        ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"), param)


def test_the_weight_loader_leaves_the_memory_at_its_initial_values(models):
    base, closed, _live = models
    fresh = traverse_util.flatten_dict(_params(_titans_config().create(jax.random.key(7))), sep="/")
    loaded = traverse_util.flatten_dict(_params(closed), sep="/")
    base_flat = traverse_util.flatten_dict(_params(base), sep="/")
    for key, value in loaded.items():
        source = base_flat if key in base_flat else fresh
        assert np.array_equal(np.asarray(value), np.asarray(source[key])), key


def test_config_rules():
    with pytest.raises(ValueError, match="pi0.5"):
        _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", titans_frames=2)
    with pytest.raises(ValueError, match="race or vlash_branches"):
        _titans_config(race=True)
    with pytest.raises(ValueError, match="race or vlash_branches"):
        _titans_config(state_cond=True, vlash_branches=2)
    with pytest.raises(ValueError, match="titans_image_key"):
        _titans_config(titans_image_key="left_wrist_0_rgb")
    with pytest.raises(ValueError, match="positive"):
        _titans_config(titans_stride=0)
    with pytest.raises(ValueError, match="titans_dropout"):
        _titans_config(titans_dropout=1.0)
    off = _config()
    assert off.titans_frames == 0
    observation_spec, _ = off.inputs_spec()
    assert observation_spec.memory_images is None and observation_spec.memory_mask is None
    observation_spec, _ = _titans_config().inputs_spec(batch_size=2)
    assert observation_spec.memory_images.shape == (2, FRAMES, *_model.IMAGE_RESOLUTION, 3)
    assert observation_spec.memory_mask.shape == (2, FRAMES)
