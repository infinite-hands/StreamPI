"""pi0's memory bank (compressed image memory and state history): zero at init, one bank for training and serving, no
gradient into the image encoders."""

# The tests check that the training and serving banks agree, which is only visible through the private helpers that build them.
# ruff: noqa: SLF001

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import pi0_testing
import openpi.models.model as model_module
from openpi.policies import agilex_policy
from openpi.shared import nnx_utils
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms

ACTION_DIM, HORIZON, TOKENS, WINDOW = 14, 4, 16, 2
CAMERA = "right_wrist_0_rgb"
MEMORY = {"memory_frames": 3, "memory_stride": 2, "memory_tokens": 2, "memory_dim": 16}
# hist_horizon 2 and a stride of 2: the slots read the frames 3, 5 and 7 calls back.
AGES = (3, 5, 7)
GATE_PATH = ("PaliGemma", "llm", "layers", "attn", "memory_gate")


def _config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(**{
        "paligemma_variant": "dummy", "action_expert_variant": "dummy", "pi05": True, "action_dim": ACTION_DIM,
        "action_horizon": HORIZON, "max_token_len": TOKENS, "dtype": "float32", "image_keys": (CAMERA,),
        "hist_horizon": WINDOW, **overrides})


def _frame(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, size=(224, 224, 3)).astype(np.float32)


def _observation(window: int, memory=None, mask=None, seed: int = 1, history=None, history_mask=None) -> _model.Observation:
    images = np.stack([_frame(seed * 100 + t) for t in range(window)])[None]
    state = np.random.default_rng(2).normal(size=(1, ACTION_DIM)).astype(np.float32)
    extra = {}
    if memory is not None:
        extra = {"memory_images": jnp.asarray(np.stack(memory)[None]),
                 "memory_mask": jnp.asarray([[True] * len(memory) if mask is None else mask])}
    if history is not None:
        extra |= {"state_history": jnp.asarray(np.stack(history)[None]),
                  "state_history_mask": jnp.asarray([[True] * len(history) if history_mask is None else history_mask])}
    return _model.Observation(
        images={CAMERA: jnp.asarray(images)}, image_masks={CAMERA: jnp.ones((1,), dtype=bool)}, state=jnp.asarray(state),
        tokenized_prompt=jnp.ones((1, TOKENS), dtype=jnp.int32), tokenized_prompt_mask=jnp.ones((1, TOKENS), dtype=bool),
        **extra)


def _create(**overrides):
    return pi0_testing.awaken(_config(**overrides).create(jax.random.key(0)), jax.random.key(7))


def _edit(model, change):
    state = nnx.state(model, nnx.Param)
    pure = state.to_pure_dict()
    change(pure)
    state.replace_by_pure_dict(pure)
    nnx.update(model, state)
    return model


def _open_gate(model, value: float = 0.5):
    def change(pure):
        node = pure
        for key in GATE_PATH[:-1]:
            node = node[key]
        node[GATE_PATH[-1]] = jnp.full_like(node[GATE_PATH[-1]], value)

    return _edit(model, change)


def _flat(model):
    return flax.traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep="/")


@pytest.fixture(scope="module")
def plain():
    return _create()


@pytest.fixture(scope="module")
def with_memory():
    return _create(**MEMORY)


# Three rows of the state at calls 2, 3 and 4 back, of dims 0-3 only.
HISTORY = {"state_history": 3, "state_history_first_age": 2, "state_history_dims": (0, 1, 2, 3)}
HISTORY_AGES = (2, 3, 4)


@pytest.fixture(scope="module")
def with_history():
    return _create(**HISTORY)


def _states(count: int, seed: int = 9) -> list[np.ndarray]:
    return [np.random.default_rng(seed + n).normal(size=ACTION_DIM).astype(np.float32) for n in range(count)]


def _loss(model, observation, seed: int = 0):
    return np.asarray(model.compute_loss(jax.random.key(seed), observation, jnp.zeros((1, HORIZON, ACTION_DIM), jnp.float32)))


def _memory_frames(seed: int = 50):
    return [_frame(seed + slot) for slot in range(MEMORY["memory_frames"])]


def test_the_slots_read_frames_beyond_the_kv_window():
    assert _config(**MEMORY).memory_ages == AGES
    assert _config(hist_horizon=5, memory_frames=16, memory_stride=4).memory_ages[:2] == (8, 12)
    assert _config(hist_horizon=5, memory_frames=16, memory_stride=4).memory_ages[-1] == 68


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"image_keys": ("right_wrist_0_rgb", "left_wrist_0_rgb")}, "one camera"),
     ({"race": True}, "plain pi0.5"),
     ({"memory_dropout": 1.0}, "memory_dropout"),
     ({"memory_tokens": 0}, "must be positive")],
)
def test_the_config_refuses_what_the_memory_is_not_wired_for(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**{**MEMORY, **overrides})


def test_a_model_with_memory_starts_as_the_model_without_it(plain, with_memory):
    shared = {key: value for key, value in _flat(with_memory).items() if "memory" not in key}
    assert shared.keys() == _flat(plain).keys()
    assert all(np.array_equal(np.asarray(value), np.asarray(_flat(plain)[key])) for key, value in shared.items())

    frames = _memory_frames()
    assert np.array_equal(_loss(plain, _observation(WINDOW)), _loss(with_memory, _observation(WINDOW, frames)))


def test_the_loss_follows_the_memory_once_the_gate_opens(plain, with_memory):
    opened = _open_gate(nnx.merge(*nnx.split(with_memory)))
    baseline = _loss(plain, _observation(WINDOW))

    seen = _loss(opened, _observation(WINDOW, _memory_frames(50)))
    other = _loss(opened, _observation(WINDOW, _memory_frames(80)))
    assert not np.allclose(seen, baseline, atol=1e-6)
    assert not np.allclose(seen, other, atol=1e-6)

    # A bank with no valid slot adds nothing, whatever the gate: the first calls of an episode are the base model's.
    hidden = _loss(opened, _observation(WINDOW, _memory_frames(50), mask=[False] * 3))
    assert np.array_equal(hidden, baseline)


def test_no_gradient_reaches_the_encoders_through_the_memory_frames(with_memory):
    opened = _open_gate(nnx.merge(*nnx.split(with_memory)))
    observation = _observation(WINDOW, _memory_frames())
    actions = jnp.zeros((1, HORIZON, ACTION_DIM), jnp.float32)

    def loss_of_memory_images(images):
        return jnp.mean(opened.compute_loss(jax.random.key(0), observation.replace(memory_images=images), actions))

    assert np.all(np.asarray(jax.grad(loss_of_memory_images)(observation.memory_images)) == 0)

    # The memory modules themselves do learn: the pooling and the age embeddings are reached through the open gate.
    graphdef, params, rest = nnx.split(opened, nnx_utils.PathRegex(_pi0_config.MEMORY_PARAMS_REGEX), ...)

    def loss_of_params(trainable):
        return jnp.mean(nnx.merge(graphdef, trainable, rest).compute_loss(jax.random.key(0), observation, actions))

    grads = flax.traverse_util.flatten_dict(jax.grad(loss_of_params)(params).to_pure_dict(), sep="/")
    grads = {key: value for key, value in grads.items() if value is not None}
    assert {"memory_query", "memory_slot_embed", "memory_pool_k/kernel", "memory_pool_norm/scale",
            "PaliGemma/llm/layers/attn/memory_k/kernel", "PaliGemma/llm/layers/attn/memory_gate"} <= grads.keys()
    assert all(np.any(np.asarray(value) != 0) for value in grads.values())


def test_the_memory_drop_hides_whole_banks(monkeypatch, with_memory):
    width = with_memory.memory_pool_k.in_features
    monkeypatch.setattr(type(with_memory), "_encode_images",
                        lambda self, image: jnp.ones((image.shape[0], 4, width), jnp.float32))
    batch = 64
    observation = _model.Observation(
        images={CAMERA: jnp.zeros((batch, 1, 224, 224, 3))}, image_masks={CAMERA: jnp.ones((batch,), dtype=bool)},
        state=jnp.zeros((batch, ACTION_DIM)),
        memory_images=jnp.zeros((batch, 3, 224, 224, 3)), memory_mask=jnp.ones((batch, 3), dtype=bool))

    _, evaluated = with_memory._training_memory_bank(jax.random.key(0), observation, train=False)
    assert bool(evaluated.all())

    _, trained = with_memory._training_memory_bank(jax.random.key(0), observation, train=True)
    kept = np.asarray(trained).reshape(batch, 3, -1).any(axis=(1, 2))
    # Each sample's bank is whole or hidden, and about memory_dropout (0.1) of them are hidden.
    assert (np.asarray(trained).reshape(batch, 3, -1).all(axis=(1, 2)) == kept).all()
    assert 0.8 * batch < kept.sum() < batch


def _served_bank(model, memory, frame):
    observation = _model.preprocess_observation(None, _observation(1), image_keys=(CAMERA,))
    observation = observation.replace(images={CAMERA: jnp.asarray(frame)[None, None]})
    noise = jnp.zeros((1, HORIZON, ACTION_DIM), jnp.float32)
    return model._encode_prefix(observation, noise, memory)[3]


def test_the_serving_ring_is_the_bank_training_builds(with_memory):
    frames = [_frame(1000 + n) for n in range(AGES[-1] + 2)]
    memory = {"memory_tokens": None, "memory_kv_cache": None, "memory_prefix_mask": None}
    for call, frame in enumerate(frames):
        if call % WINDOW == 0:
            with_memory.reset_memory(memory)  # a server's periodic KV reset must not touch the ring
        served, served_mask = _served_bank(with_memory, memory, frame)

        older = [frames[call - age] if call >= age else np.zeros_like(frame) for age in AGES]
        valid = [call >= age for age in AGES]
        trained, trained_mask = with_memory._training_memory_bank(
            None, _model.preprocess_observation(None, _observation(1, older, valid), image_keys=(CAMERA,)), train=False)

        assert np.array_equal(np.asarray(served_mask), np.asarray(trained_mask))
        tokens = MEMORY["memory_tokens"]
        for slot, is_valid in enumerate(valid):
            if is_valid:
                np.testing.assert_allclose(
                    np.asarray(served)[0, slot * tokens:(slot + 1) * tokens], np.asarray(trained)[0, slot * tokens:(slot + 1) * tokens],
                    atol=1e-4, rtol=1e-4)
        assert int(np.asarray(served_mask).sum()) == sum(valid) * tokens

        if call == len(frames) - 1:
            # Not vacuous: the frame one call older than the slot's would give other tokens.
            wrong, _ = with_memory._training_memory_bank(None, _model.preprocess_observation(None, _observation(
                1, [frames[call - age - 1] for age in AGES], valid), image_keys=(CAMERA,)), train=False)
            assert not np.allclose(np.asarray(served)[0, :tokens], np.asarray(wrong)[0, :tokens], atol=1e-3)

    assert memory["long_frames"].shape[:2] == (1, AGES[-1])
    with_memory.reset_long_memory(memory)
    assert memory["long_frames"] is None
    assert not np.asarray(_served_bank(with_memory, memory, frames[0])[1]).any()


def test_a_served_chunk_follows_the_bank_once_the_gate_opens(with_memory):
    frames = [_frame(2000 + n) for n in range(AGES[0] + 1)]
    noise = jnp.asarray(np.random.default_rng(3).normal(size=(1, HORIZON, ACTION_DIM)).astype(np.float32))

    def serve(model, *, hide_bank: bool):
        memory = {"memory_tokens": None, "memory_kv_cache": None, "memory_prefix_mask": None}
        for call, frame in enumerate(frames):
            observation = _observation(1).replace(images={CAMERA: jnp.asarray(frame)[None, None]})
            if call == len(frames) - 1 and hide_bank:
                memory["long_valid"] = jnp.zeros_like(memory["long_valid"])
            actions, memory = model.sample_actions(jax.random.key(0), observation, noise=noise, memory=memory)
        return np.asarray(actions)

    closed, opened = with_memory, _open_gate(nnx.merge(*nnx.split(with_memory)))
    assert np.array_equal(serve(closed, hide_bank=False), serve(closed, hide_bank=True))
    assert not np.allclose(serve(opened, hide_bank=False), serve(opened, hide_bank=True), atol=1e-6)
    assert np.array_equal(serve(opened, hide_bank=True), serve(closed, hide_bank=True))


def test_candidates_read_the_same_bank_as_a_single_sample(with_memory):
    opened = _open_gate(nnx.merge(*nnx.split(with_memory)))
    frames = [_frame(3000 + n) for n in range(AGES[0] + 1)]
    noise = jnp.asarray(np.random.default_rng(4).normal(size=(1, 2, HORIZON, ACTION_DIM)).astype(np.float32))

    def warm():
        memory = {"memory_tokens": None, "memory_kv_cache": None, "memory_prefix_mask": None}
        for frame in frames[:-1]:
            _, memory = opened.sample_actions(
                jax.random.key(0), _observation(1).replace(images={CAMERA: jnp.asarray(frame)[None, None]}), noise=noise[:, 0], memory=memory)
        return memory

    last = _observation(1).replace(images={CAMERA: jnp.asarray(frames[-1])[None, None]})
    candidates, _ = opened.sample_action_candidates(jax.random.key(0), last, num_candidates=2, noise=noise, memory=warm())
    for index in range(2):
        single, _ = opened.sample_actions(jax.random.key(0), last, noise=noise[:, index], memory=warm())
        np.testing.assert_allclose(np.asarray(candidates)[:, index], np.asarray(single), atol=1e-5)


def test_the_observation_carries_the_memory_frames():
    observation = _observation(1, _memory_frames(), mask=[True, False, True])
    again = _model.Observation.from_dict(observation.to_dict())
    assert np.array_equal(np.asarray(again.memory_images), np.asarray(observation.memory_images))
    assert np.array_equal(np.asarray(again.memory_mask), [[True, False, True]])

    raw = observation.to_dict()
    raw["memory_image"] = np.full((1, 3, 8, 8, 3), 255, dtype=np.uint8)
    assert np.all(np.asarray(_model.Observation.from_dict(raw).memory_images) == 1.0)

    assert _model.Observation.from_dict(_observation(1).to_dict()).memory_images is None


def test_preprocessing_resizes_and_augments_the_memory_frames_without_moving_the_windows_draws():
    small = [np.random.default_rng(s).uniform(-1, 1, size=(100, 100, 3)).astype(np.float32) for s in range(3)]
    observation = _observation(WINDOW, small)
    kwargs = {"image_keys": (CAMERA,), "image_resolution": (224, 224)}

    evaluated = _model.preprocess_observation(None, observation, **kwargs)
    assert evaluated.memory_images.shape == (1, 3, 224, 224, 3)
    assert np.array_equal(np.asarray(evaluated.memory_mask), np.asarray(observation.memory_mask))

    rng = jax.random.key(0)
    trained = _model.preprocess_observation(rng, observation, train=True, wrist_augmentation=True, **kwargs)
    assert not np.allclose(np.asarray(trained.memory_images), np.asarray(evaluated.memory_images), atol=1e-3)

    without = _model.preprocess_observation(rng, _observation(WINDOW), train=True, wrist_augmentation=True, **kwargs)
    assert np.array_equal(np.asarray(trained.images[CAMERA]), np.asarray(without.images[CAMERA]))


def test_the_memory_weights_train_even_where_the_language_model_is_frozen():
    def trainable(model_config) -> set[str]:
        model = nnx.eval_shape(lambda: model_config.create(jax.random.key(0)))
        state = nnx.state(model, nnx.All(nnx.Param, nnx.Not(model_config.get_freeze_filter())))
        names = ("/".join(str(getattr(part, "key", getattr(part, "name", part))) for part in path)
                 for path, _ in jax.tree_util.tree_flatten_with_path(state)[0])
        return {name.removesuffix("/value") for name in names}

    lora = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora", "pi05": True,
            "image_keys": (CAMERA,), "hist_horizon": WINDOW}
    base = trainable(_pi0_config.Pi0Config(**lora))
    grown = trainable(_pi0_config.Pi0Config(**lora, **MEMORY))
    added = grown - base
    assert {"PaliGemma/llm/layers/attn/memory_gate", "PaliGemma/llm/layers/attn/memory_k/kernel",
            "PaliGemma/llm/layers/attn/memory_v/kernel", "memory_query", "memory_slot_embed"} <= added
    assert base <= grown
    assert all("memory" in name for name in added)


def test_the_loader_leaves_the_memory_weights_at_their_initialisation(monkeypatch):
    saved = {"PaliGemma": {"img": {"w": np.ones(3, np.float32)}}}
    monkeypatch.setattr(model_module, "restore_params", lambda *args, **kwargs: saved)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path, **kwargs: path)
    reference = {"PaliGemma": {"img": {"w": jax.ShapeDtypeStruct((3,), np.float32)},
                               "llm": {"layers": {"attn": {"memory_gate": jax.ShapeDtypeStruct((2,), np.float32)}}}},
                 "memory_query": jax.ShapeDtypeStruct((2, 16), np.float32)}

    merged = weight_loaders.CheckpointWeightLoader("ignored").load(reference)
    assert merged["memory_query"] is reference["memory_query"]
    assert merged["PaliGemma"]["llm"]["layers"]["attn"]["memory_gate"] is reference["PaliGemma"]["llm"]["layers"]["attn"]["memory_gate"]
    assert np.array_equal(merged["PaliGemma"]["img"]["w"], saved["PaliGemma"]["img"]["w"])


def test_a_batch_from_the_transform_chain_reaches_the_loss_through_the_bank(with_memory):
    rng = np.random.default_rng(5)
    sample = {
        "images": {"cam_right_wrist": rng.uniform(size=(WINDOW, 3, 224, 224)).astype(np.float32)},
        "state": np.zeros(ACTION_DIM, np.float32), "actions": np.zeros((HORIZON, ACTION_DIM), np.float32),
        "memory_image": rng.uniform(size=(3, 3, 224, 224)).astype(np.float32),
        "memory_mask": np.asarray([True, True, False]),
    }
    data = transforms.ResizeImages(224, 224)(agilex_policy.AgilexInputs()(sample))
    data["tokenized_prompt"], data["tokenized_prompt_mask"] = np.ones(TOKENS, np.int32), np.ones(TOKENS, bool)
    observation = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], data))
    assert observation.memory_images.shape == (1, 3, 224, 224, 3)
    assert observation.memory_images.dtype == np.float32
    assert np.asarray(observation.memory_mask).tolist() == [[True, True, False]]

    opened = _open_gate(nnx.merge(*nnx.split(with_memory)))
    loss = _loss(opened, observation)
    assert np.isfinite(loss).all()
    hidden = observation.replace(memory_mask=jnp.zeros_like(observation.memory_mask))
    assert not np.allclose(loss, _loss(opened, hidden), atol=1e-6)


def test_the_state_history_rows_sit_first_age_calls_back():
    config = _config(**HISTORY)
    assert config.state_history_ages == HISTORY_AGES
    assert config.uses_memory_bank
    assert not _config().uses_memory_bank


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"state_history_dims": ()}, "state_history_dims"),
     ({"state_history_dims": (0, ACTION_DIM)}, "state_history_dims"),
     ({"state_history_first_age": 0}, "first_age"),
     ({"state_history_dropout": 1.0}, "state_history_dropout"),
     ({"race": True}, "plain pi0.5")],
)
def test_the_config_refuses_a_state_history_it_cannot_serve(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**{**HISTORY, **overrides})


def test_a_model_with_a_state_history_starts_as_the_model_without_it(plain, with_history):
    shared = {key: value for key, value in _flat(with_history).items() if "memory" not in key}
    assert shared.keys() == _flat(plain).keys()
    assert all(np.array_equal(np.asarray(value), np.asarray(_flat(plain)[key])) for key, value in shared.items())

    rows = _states(3)
    assert np.array_equal(_loss(plain, _observation(WINDOW)), _loss(with_history, _observation(WINDOW, history=rows)))


def test_the_loss_follows_the_chosen_state_dims_once_the_gate_opens(plain, with_history):
    opened = _open_gate(nnx.merge(*nnx.split(with_history)))
    baseline = _loss(plain, _observation(WINDOW))
    rows = _states(3)

    seen = _loss(opened, _observation(WINDOW, history=rows))
    assert not np.allclose(seen, baseline, atol=1e-6)
    assert not np.allclose(seen, _loss(opened, _observation(WINDOW, history=_states(3, seed=40))), atol=1e-6)

    # Dims the history does not name are not read: change one and nothing moves.
    other = [row.copy() for row in rows]
    for row in other:
        row[10] += 5.0
    assert np.array_equal(seen, _loss(opened, _observation(WINDOW, history=other)))

    hidden = _loss(opened, _observation(WINDOW, history=rows, history_mask=[False] * 3))
    assert np.array_equal(hidden, baseline)


def test_the_serving_state_ring_is_the_bank_training_builds(with_history):
    states = _states(HISTORY_AGES[-1] + 2)
    memory = {}
    for call, state in enumerate(states):
        observation = _observation(1).replace(state=jnp.asarray(state)[None])
        served, served_mask = with_history._serving_memory_bank(observation, {}, memory)

        older = [states[call - age] if call >= age else np.zeros_like(state) for age in HISTORY_AGES]
        valid = [call >= age for age in HISTORY_AGES]
        trained, trained_mask = with_history._training_memory_bank(
            None, _observation(1, history=older, history_mask=valid), train=False)

        assert np.array_equal(np.asarray(served_mask), np.asarray(trained_mask))
        assert np.asarray(served_mask).tolist() == [valid]
        for row, is_valid in enumerate(valid):
            if is_valid:
                np.testing.assert_allclose(np.asarray(served)[0, row], np.asarray(trained)[0, row], atol=1e-6)

        if call == len(states) - 1:
            # Not vacuous: one call older than each row's age would give other rows.
            wrong, _ = with_history._training_memory_bank(
                None, _observation(1, history=[states[call - age - 1] for age in HISTORY_AGES]), train=False)
            assert not np.allclose(np.asarray(served), np.asarray(wrong), atol=1e-3)

    assert memory["long_states"].shape[:2] == (1, HISTORY_AGES[-1])
    with_history.reset_long_memory(memory)
    assert all(memory[key] is None for key in ("long_frames", "long_valid", "long_states", "long_states_valid"))


def test_the_history_hides_whole_per_sample_in_training(with_history):
    batch = 64
    observation = _model.Observation(
        images={CAMERA: jnp.zeros((batch, 1, 224, 224, 3))}, image_masks={CAMERA: jnp.ones((batch,), dtype=bool)},
        state=jnp.zeros((batch, ACTION_DIM)),
        state_history=jnp.ones((batch, 3, ACTION_DIM)), state_history_mask=jnp.ones((batch, 3), dtype=bool))
    _, evaluated = with_history._training_memory_bank(jax.random.key(0), observation, train=False)
    assert bool(evaluated.all())

    _, trained = with_history._training_memory_bank(jax.random.key(0), observation, train=True)
    kept = np.asarray(trained).any(axis=1)
    assert (np.asarray(trained).all(axis=1) == kept).all()
    # state_history_dropout is 0.5.
    assert 20 < kept.sum() < 44


def test_an_image_memory_and_a_state_history_share_one_bank(monkeypatch):
    model = _create(**MEMORY, **HISTORY)
    width = model.memory_pool_k.in_features
    projection = np.random.default_rng(0).normal(size=(3, width)).astype(np.float32)

    def cheap_encoder(self, image):
        # A few fixed pixels of each frame, projected to the token width: distinct frames give distinct tokens.
        pixels = image[:, ::56, ::56, :].reshape(image.shape[0], -1, 3)
        return pixels @ projection

    monkeypatch.setattr(type(model), "_encode_images", cheap_encoder)
    frames, states = [_frame(4000 + n) for n in range(AGES[-1] + 1)], _states(AGES[-1] + 1)
    memory = {}
    for call, (frame, state) in enumerate(zip(frames, states, strict=True)):
        observation = _model.preprocess_observation(None, _observation(1), image_keys=(CAMERA,)).replace(
            images={CAMERA: jnp.asarray(frame)[None, None]}, state=jnp.asarray(state)[None])
        served, served_mask = model._serving_memory_bank(observation, model._frame_tokens(observation), memory)

        older = [frames[call - age] if call >= age else np.zeros_like(frame) for age in AGES]
        valid = [call >= age for age in AGES]
        rows = [states[call - age] if call >= age else np.zeros_like(states[0]) for age in HISTORY_AGES]
        rows_valid = [call >= age for age in HISTORY_AGES]
        trained, trained_mask = model._training_memory_bank(None, _model.preprocess_observation(None, _observation(
            1, older, valid, history=rows, history_mask=rows_valid), image_keys=(CAMERA,)), train=False)

        images = MEMORY["memory_frames"] * MEMORY["memory_tokens"]
        assert served.shape[1] == images + HISTORY["state_history"] == 9
        assert np.array_equal(np.asarray(served_mask), np.asarray(trained_mask))
        valid_rows = np.asarray(trained_mask)[0]
        np.testing.assert_allclose(np.asarray(served)[0][valid_rows], np.asarray(trained)[0][valid_rows], atol=1e-5)
    assert np.asarray(served_mask).all()


def test_the_state_encoder_trains_under_a_frozen_language_model():
    def trainable(model_config) -> set[str]:
        model = nnx.eval_shape(lambda: model_config.create(jax.random.key(0)))
        state = nnx.state(model, nnx.All(nnx.Param, nnx.Not(model_config.get_freeze_filter())))
        names = ("/".join(str(getattr(part, "key", getattr(part, "name", part))) for part in path)
                 for path, _ in jax.tree_util.tree_flatten_with_path(state)[0])
        return {name.removesuffix("/value") for name in names}

    lora = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora", "pi05": True,
            "image_keys": (CAMERA,), "hist_horizon": WINDOW}
    added = trainable(_pi0_config.Pi0Config(**lora, **HISTORY)) - trainable(_pi0_config.Pi0Config(**lora))
    assert {"memory_state_in/kernel", "memory_state_out/kernel", "memory_state_age", "PaliGemma/llm/layers/attn/memory_gate",
            "PaliGemma/llm/layers/attn/memory_k/kernel"} <= added
    assert all("memory" in name for name in added)
    # A state history alone has no image pooling.
    assert not any(name.startswith(("memory_query", "memory_pool", "memory_slot")) for name in added)


def test_the_observation_carries_the_state_history():
    rows = _states(3)
    observation = _observation(1, history=rows, history_mask=[True, False, True])
    again = _model.Observation.from_dict(observation.to_dict())
    assert np.array_equal(np.asarray(again.state_history), np.asarray(observation.state_history))
    assert np.asarray(again.state_history_mask).tolist() == [[True, False, True]]
    preprocessed = _model.preprocess_observation(None, observation, image_keys=(CAMERA,))
    assert np.array_equal(np.asarray(preprocessed.state_history), np.asarray(observation.state_history))
    assert _model.Observation.from_dict(_observation(1).to_dict()).state_history is None
