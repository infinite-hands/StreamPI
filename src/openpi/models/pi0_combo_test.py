"""pi0 with every opt-in single-camera feature at once (the `_xr128_px336_dino_mem_hist` recipe's model): a non-stock resolution,
the DINOv2 branch, the image memory and the state history on one wrist camera. The sibling tests (pi0_dino_test,
pi0_memory_test, pi0_options_test) prove each feature alone; these prove they compose, and that the loader's growth steps
turn a plain checkpoint into the combined model's parameters. Dummy gemma and DINOv2 variants (the SigLIP is the real one), CPU."""

import gc

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import dinov2
from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import pi0_testing
import openpi.models.model as model_module
from openpi.shared import nnx_utils
import openpi.training.config as config
import openpi.training.weight_loaders as weight_loaders

ACTION_DIM, HORIZON, TOKENS, WINDOW = 14, 4, 16, 2
# 18 x 18 patches: not SigLIP's stock 16 x 16, and a multiple of its 14-pixel patch.
SIDE = 252
CAMERA = "left_wrist_0_rgb"
COMBO = {
    "dino_variant": "dummy",
    "memory_frames": 3, "memory_stride": 2, "memory_tokens": 2, "memory_dim": 16,
    # Three rows of the left arm's dims (0-6) of the state, 2, 3 and 4 calls back.
    "state_history": 3, "state_history_first_age": 2, "state_history_dims": tuple(range(7)),
}
# hist_horizon 2 and a stride of 2: the image slots read the frames 3, 5 and 7 calls back.
MEMORY_AGES, HISTORY_AGES = (3, 5, 7), (2, 3, 4)
GATE_PATH = ("PaliGemma", "llm", "layers", "attn", "memory_gate")
CALLS = 9
OPT_IN_PREFIXES = ("dino/", "dino_proj/", "memory")


def _config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(**{
        "paligemma_variant": "dummy", "action_expert_variant": "dummy", "pi05": True, "action_dim": ACTION_DIM,
        "action_horizon": HORIZON, "max_token_len": TOKENS, "dtype": "float32", "image_keys": (CAMERA,),
        "hist_horizon": WINDOW, "image_resolution": (SIDE, SIDE), **overrides})


def _uniform(seed: int, *shape: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, size=shape).astype(np.float32)


def _training_observation(side: int = SIDE, seed: int = 1) -> _model.Observation:
    """What the data loader hands the model: the window's frames, the memory frames and the state history (all rows valid)."""
    return _model.Observation(
        images={CAMERA: jnp.asarray(_uniform(seed, 1, WINDOW, side, side, 3))},
        image_masks={CAMERA: jnp.ones((1,), dtype=bool)},
        state=jnp.asarray(np.random.default_rng(2).normal(size=(1, ACTION_DIM)).astype(np.float32)),
        tokenized_prompt=jnp.ones((1, TOKENS), dtype=jnp.int32), tokenized_prompt_mask=jnp.ones((1, TOKENS), dtype=bool),
        memory_images=jnp.asarray(_uniform(seed + 10, 1, COMBO["memory_frames"], side, side, 3)),
        memory_mask=jnp.ones((1, COMBO["memory_frames"]), dtype=bool),
        state_history=jnp.asarray(np.random.default_rng(seed + 20).normal(size=(1, COMBO["state_history"], ACTION_DIM)).astype(np.float32)),
        state_history_mask=jnp.ones((1, COMBO["state_history"]), dtype=bool))


def _bare(observation: _model.Observation) -> _model.Observation:
    return observation.replace(memory_images=None, memory_mask=None, state_history=None, state_history_mask=None)


def _loss(model, observation, *, train: bool = False) -> np.ndarray:
    return np.asarray(model.compute_loss(jax.random.key(0), observation, jnp.zeros((1, HORIZON, ACTION_DIM), jnp.float32), train=train))


def _serving_observation(call: int) -> _model.Observation:
    """One call of a served episode: one new frame and the state, as the client sends them."""
    return _model.Observation(
        images={CAMERA: jnp.asarray(_uniform(5000 + call, 1, 1, SIDE, SIDE, 3))}, image_masks={CAMERA: jnp.ones((1,), dtype=bool)},
        state=jnp.asarray(np.random.default_rng(6000 + call).normal(size=(1, ACTION_DIM)).astype(np.float32)),
        tokenized_prompt=jnp.ones((1, TOKENS), dtype=jnp.int32), tokenized_prompt_mask=jnp.ones((1, TOKENS), dtype=bool))


NOISE = jnp.asarray(np.random.default_rng(3).normal(size=(1, HORIZON, ACTION_DIM)).astype(np.float32))


def _serve(model, memory: dict, steps=None):
    """Serves the calls `steps` the way Policy.infer does: the KV window is reset every hist_horizon calls and the compressed
    memory only when the step counter is back at 0. Returns the chunks and, after each call, the ring fill counts."""
    steps = range(CALLS) if steps is None else steps
    chunks, fills = [], []
    for step in steps:
        if step % model.hist_horizon == 0:
            model.reset_memory(memory)
            if step == 0:
                model.reset_long_memory(memory)
        actions, _ = model.sample_actions(jax.random.key(0), _serving_observation(step), noise=NOISE, memory=memory)
        chunks.append(np.asarray(actions))
        fills.append(tuple(int(np.asarray(memory[key]).sum()) if memory.get(key) is not None else None
                           for key in ("long_valid", "long_states_valid")))
    return chunks, fills


def _fresh_memory() -> dict:
    return {"memory_tokens": None, "memory_kv_cache": None, "memory_prefix_mask": None}


def _create(**overrides):
    return pi0_testing.awaken(_config(**overrides).create(jax.random.key(0)), jax.random.key(7))


def _copy(model):
    return nnx.merge(*nnx.split(model))


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


def _move_dino_proj(model):
    def change(pure):
        pure["dino_proj"]["kernel"] = 0.5 * jax.random.normal(jax.random.key(3), pure["dino_proj"]["kernel"].shape)

    return _edit(model, change)


def _flat(model) -> dict:
    return flax.traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep="/")


def _checksum(value) -> tuple:
    array = np.asarray(value, np.float64)
    return array.shape, float(array.sum()), float(np.abs(array).sum())


@pytest.fixture(scope="module")
def plain_reference():
    """What the same model without the opt-in parts computes, kept as numbers: two SigLIP-bearing models never coexist."""
    model = _create()
    chunks, _ = _serve(model, _fresh_memory())
    reference = {"loss": _loss(model, _bare(_training_observation())), "chunks": chunks,
                 "checksums": {key: _checksum(value) for key, value in _flat(model).items()}}
    del model
    gc.collect()
    jax.clear_caches()
    return reference


@pytest.fixture(scope="module")
def combo(plain_reference):
    return _create(**COMBO)


def test_the_combo_model_builds_with_every_option_and_reads_one_camera_at_the_non_stock_size(combo):
    assert (combo.image_keys, combo.image_resolution) == ((CAMERA,), (SIDE, SIDE))
    assert combo.dino is not None
    assert (combo.memory_frames, combo.state_history, combo.uses_memory_bank) == (3, 3, True)
    assert (combo.memory_ages, combo.state_history_ages) == (MEMORY_AGES, HISTORY_AGES)

    observation, _ = _config(**COMBO).inputs_spec(batch_size=1)
    assert observation.images[CAMERA].shape == (1, SIDE, SIDE, 3)
    assert observation.memory_images.shape == (1, 3, SIDE, SIDE, 3)
    assert observation.state_history.shape == (1, 3, ACTION_DIM)

    # 18 x 18 = 324 image tokens per frame, and the DINOv2 branch adds no tokens of its own.
    prefix = combo.embed_prefix(_model.preprocess_observation(None, _bare(_training_observation()), image_keys=(CAMERA,),
                                                              image_resolution=(SIDE, SIDE)))
    assert (prefix[4], prefix[3]) == (WINDOW, 324 + TOKENS)


def test_the_loader_supplied_inputs_give_a_finite_loss_in_evaluation_and_training(combo):
    observation = _training_observation()
    evaluated = _loss(combo, observation)
    assert evaluated.shape == (1, HORIZON)
    assert np.isfinite(evaluated).all()

    # Training mode runs the augmentations, the memory and state-history dropouts and the frame dropout.
    trained = _loss(combo, observation, train=True)
    assert np.isfinite(trained).all()

    # The data transforms resize to the model's size, but the model also resizes a 224 px window and memory by itself.
    small = _training_observation(side=224)
    assert small.memory_images.shape[2:4] == (224, 224)
    assert np.isfinite(_loss(combo, small)).all()


def test_every_new_weight_gets_a_gradient_through_the_combined_loss(combo):
    opened = _move_dino_proj(_open_gate(_copy(combo)))
    observation = _training_observation()
    actions = jnp.zeros((1, HORIZON, ACTION_DIM), jnp.float32)
    regex = f"{_pi0_config.MEMORY_PARAMS_REGEX}|.*dino_proj.*"
    graphdef, params, rest = nnx.split(opened, nnx_utils.PathRegex(regex), ...)

    def loss_of_params(trainable):
        return jnp.mean(nnx.merge(graphdef, trainable, rest).compute_loss(jax.random.key(0), observation, actions))

    grads = flax.traverse_util.flatten_dict(jax.grad(loss_of_params)(params).to_pure_dict(), sep="/")
    grads = {key: value for key, value in grads.items() if value is not None}
    assert {"dino_proj/kernel", "memory_query", "memory_slot_embed", "memory_pool_k/kernel", "memory_state_in/kernel",
            "memory_state_out/kernel", "memory_state_age", "PaliGemma/llm/layers/attn/memory_k/kernel",
            "PaliGemma/llm/layers/attn/memory_gate"} <= grads.keys()
    assert all(np.isfinite(np.asarray(value)).all() for value in grads.values())
    assert all(np.any(np.asarray(value) != 0) for value in grads.values())


def test_with_the_gate_closed_and_dino_proj_zero_the_combo_is_the_model_without_the_new_parts(combo, plain_reference):
    flat = _flat(combo)
    shared = {key: value for key, value in flat.items() if not key.startswith(OPT_IN_PREFIXES) and "memory" not in key}
    assert shared.keys() == plain_reference["checksums"].keys()
    assert all(_checksum(value) == plain_reference["checksums"][key] for key, value in shared.items())
    # The zero-initialised parts really are zero: the closed gate and the projection.
    assert not np.any(np.asarray(flat["PaliGemma/llm/layers/attn/memory_gate"]))
    assert not np.any(np.asarray(flat["dino_proj/kernel"]))

    np.testing.assert_allclose(_loss(combo, _training_observation()), plain_reference["loss"], atol=1e-5)

    # Served over a whole episode with the periodic KV reset, the chunks are the plain model's, call for call.
    chunks, _ = _serve(combo, _fresh_memory())
    assert len(chunks) == CALLS
    for call, (chunk, expected) in enumerate(zip(chunks, plain_reference["chunks"], strict=True)):
        assert np.isfinite(chunk).all(), call
        np.testing.assert_allclose(chunk, expected, atol=1e-5, err_msg=f"call {call}")


def test_a_served_episode_advances_the_rings_and_the_chunks_follow_the_gate_and_the_projection_once_they_are_valid(combo):
    baseline, fills = _serve(combo, memory := _fresh_memory())
    assert all(np.isfinite(chunk).all() for chunk in baseline)
    # The rings fill one entry per call up to their length, through the periodic resets (steps 2, 4, 6 and 8).
    assert fills == [(min(call + 1, MEMORY_AGES[-1]), min(call + 1, HISTORY_AGES[-1])) for call in range(CALLS)]
    assert memory["long_frames"].shape == (1, MEMORY_AGES[-1], COMBO["memory_tokens"], COMBO["memory_dim"])
    assert memory["long_states"].shape == (1, HISTORY_AGES[-1], len(COMBO["state_history_dims"]))

    # A new episode (the step counter back at 0) forgets the rings and serves its first call as the first call was.
    again, restart_fills = _serve(combo, memory, steps=[0])
    assert restart_fills == [(1, 1)]
    np.testing.assert_allclose(again[0], baseline[0], atol=1e-6)

    # Gate open, projection still zero: the state rows (2-4 calls back) are valid from call 2 and the image slots from call 3.
    gated, _ = _serve(_open_gate(_copy(combo)), _fresh_memory())
    for call in range(CALLS):
        same = np.allclose(gated[call], baseline[call], atol=1e-6)
        assert same == (call < HISTORY_AGES[0]), f"call {call}"
    assert np.isfinite(gated[-1]).all()

    # Projection moved with the gate still closed: the DINOv2 tokens change every call, from the first.
    moved, _ = _serve(_move_dino_proj(_copy(combo)), _fresh_memory())
    assert all(not np.allclose(chunk, base, atol=1e-6) for chunk, base in zip(moved, baseline, strict=True))

    # Both: the chunks differ from either alone once the rings are valid, and the oldest image slot has filled by the last call.
    both, _ = _serve(_move_dino_proj(_open_gate(_copy(combo))), _fresh_memory())
    assert all(np.isfinite(chunk).all() for chunk in both)
    for call in range(HISTORY_AGES[0], CALLS):
        assert not np.allclose(both[call], moved[call], atol=1e-6), call
        assert not np.allclose(both[call], gated[call], atol=1e-6), call


def _param_shapes(model_config) -> dict:
    return flax.traverse_util.flatten_dict(
        jax.eval_shape(lambda rng: nnx.state(model_config.create(rng), nnx.Param).to_pure_dict(), jax.random.key(0)), sep="/")


def test_the_loader_turns_a_plain_left_i15_checkpoint_into_the_combined_models_parameters(monkeypatch):
    base, twin = config.get_config("pi05_yam_stream5_i15_bagging_left_real"), config.get_config(
        "pi05_yam_stream5_i15_bagging_left_real_xr128_px336_dino_mem_hist")
    small, large = _param_shapes(base.model), _param_shapes(twin.model)
    assert small.keys() < large.keys()
    new = large.keys() - small.keys()
    assert new
    assert all(key.startswith(OPT_IN_PREFIXES) or "memory" in key for key in new)
    assert {key for key in small if small[key].shape != large[key].shape} == {
        key for key in small if ("lora" in key and "_1/" in key) or key.endswith("pos_embedding")}

    # A checkpoint of the plain recipe, zero-copy except where a value is checked: the adapters and the position embedding.
    def stand_in(key, leaf):
        value = np.ones((), leaf.dtype) if "lora" in key or key.endswith("pos_embedding") else np.zeros((), leaf.dtype)
        return np.broadcast_to(value, leaf.shape)

    saved = flax.traverse_util.unflatten_dict({key: stand_in(key, leaf) for key, leaf in small.items()}, sep="/")
    sources = []

    def fake_hugging_face(source, dino_config):
        sources.append(source)
        return flax.traverse_util.unflatten_dict(
            {key.removeprefix("dino/"): np.broadcast_to(np.full((), 7.0, leaf.dtype), leaf.shape)
             for key, leaf in large.items() if key.startswith("dino/")}, sep="/")

    monkeypatch.setattr(model_module, "restore_params", lambda *args, **kwargs: saved)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path, **kwargs: path)
    monkeypatch.setattr(dinov2, "load_hf_params", fake_hugging_face)

    reference = flax.traverse_util.unflatten_dict(large, sep="/")
    loaded = flax.traverse_util.flatten_dict(twin.weight_loader.load(reference), sep="/")

    assert loaded.keys() == large.keys()
    # A bias-free layer's bias is an empty (None) entry in the parameter tree, on both sides.
    assert {key for key, leaf in large.items() if leaf is None} == {key for key, leaf in loaded.items() if leaf is None}
    arrays = [key for key, leaf in large.items() if leaf is not None]
    assert all(tuple(loaded[key].shape) == tuple(large[key].shape) for key in arrays)
    assert all(np.dtype(loaded[key].dtype) == np.dtype(large[key].dtype) for key in arrays)
    assert sources == [dinov2.HF_REPOS["base"]]

    # DINOv2 comes from Hugging Face; the projection and the memory modules are left at the model's own initialisation.
    assert all(np.all(np.asarray(loaded[key]) == 7.0) for key in large if key.startswith("dino/") and loaded[key].size < 10_000)
    assert all(loaded[key] is large[key] for key in new if not key.startswith("dino/"))
    # Everything else the checkpoint had is kept: untouched weights as they were, the grown ones containing the old values.
    saved_flat = flax.traverse_util.flatten_dict(saved, sep="/")
    assert all(loaded[key] is saved_flat[key] for key in small if small[key].shape == large[key].shape)
    expert_lora_a = next(key for key in small if key.endswith("gating_einsum_lora_a") and "_1/" in key)
    old_rank, new_rank = small[expert_lora_a].shape[-1], large[expert_lora_a].shape[-1]
    assert (old_rank, new_rank) == (32, 128)
    assert np.all(np.asarray(loaded[expert_lora_a])[..., :old_rank] == 1.0)
    expert_lora_b = expert_lora_a.replace("lora_a", "lora_b")
    assert np.all(np.asarray(loaded[expert_lora_b])[..., :old_rank, :] == 1.0)
    # The new ranks add nothing until they train: lora_b's new rows are zero.
    assert not np.any(np.asarray(loaded[expert_lora_b])[..., old_rank:, :])
    posemb = next(key for key in small if key.endswith("pos_embedding"))
    assert tuple(loaded[posemb].shape) == (1, 576, 1152)
    np.testing.assert_allclose(np.asarray(loaded[posemb]), 1.0, atol=1e-4)
