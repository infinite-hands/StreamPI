"""pi0's DINOv2 branch: zero at init, frozen backbone, trained projection, and weights merged from Hugging Face."""

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
import openpi.training.config as config
import openpi.training.weight_loaders as weight_loaders

ACTION_DIM, HORIZON, TOKENS = 14, 4, 16
CAMERA = "right_wrist_0_rgb"


def _config(**overrides) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=ACTION_DIM, action_horizon=HORIZON,
        hist_horizon=1, max_token_len=TOKENS, dtype="float32", image_keys=(CAMERA,), **overrides)


def _observation(seed: int = 1) -> _model.Observation:
    with jax.ensure_compile_time_eval():
        image = jnp.asarray(np.random.default_rng(seed).uniform(-1, 1, size=(1, 1, 224, 224, 3)), dtype=jnp.float32)
        state = jnp.asarray(np.random.default_rng(2).normal(size=(1, ACTION_DIM)), dtype=jnp.float32)
    return _model.Observation(
        images={CAMERA: image}, image_masks={CAMERA: jnp.ones((1,), dtype=bool)}, state=state,
        tokenized_prompt=jnp.ones((1, TOKENS), dtype=jnp.int32), tokenized_prompt_mask=jnp.ones((1, TOKENS), dtype=bool))


def _create(**overrides):
    return pi0_testing.awaken(_config(**overrides).create(jax.random.key(0)), jax.random.key(7))


def _flat(model_or_state):
    return flax.traverse_util.flatten_dict(nnx.state(model_or_state, nnx.Param).to_pure_dict(), sep="/")


def test_a_model_with_a_dino_branch_starts_as_the_model_without_it():
    plain, with_dino = _create(), _create(dino_variant="dummy")
    shared = {key: value for key, value in _flat(with_dino).items() if not key.startswith(("dino/", "dino_proj/"))}
    assert shared.keys() == _flat(plain).keys()
    assert all(np.array_equal(np.asarray(value), np.asarray(_flat(plain)[key])) for key, value in shared.items())

    actions = jnp.zeros((1, HORIZON, ACTION_DIM), dtype=jnp.float32)
    key = jax.random.key(0)
    assert np.array_equal(np.asarray(plain.compute_loss(key, _observation(), actions)),
                          np.asarray(with_dino.compute_loss(key, _observation(), actions)))

    # The equality above can fail: give the projection weights and the loss moves, and now it follows the image.
    state = nnx.state(with_dino, nnx.Param)
    pure = state.to_pure_dict()
    pure["dino_proj"]["kernel"] = 0.5 * jax.random.normal(jax.random.key(3), pure["dino_proj"]["kernel"].shape)
    state.replace_by_pure_dict(pure)
    nnx.update(with_dino, state)
    moved = with_dino.compute_loss(key, _observation(), actions)
    assert not np.allclose(np.asarray(moved), np.asarray(plain.compute_loss(key, _observation(), actions)), atol=1e-6)
    assert not np.allclose(np.asarray(moved), np.asarray(with_dino.compute_loss(key, _observation(seed=99), actions)), atol=1e-6)


def test_the_dino_branch_adds_no_prefix_tokens():
    plain, with_dino = _create(), _create(dino_variant="dummy")
    observation = _model.preprocess_observation(None, _observation(), image_keys=(CAMERA,))
    assert plain.embed_prefix(observation)[0].shape == with_dino.embed_prefix(observation)[0].shape


def _trainable(model_config) -> set[str]:
    model = nnx.eval_shape(lambda: model_config.create(jax.random.key(0)))
    state = nnx.state(model, nnx.All(nnx.Param, nnx.Not(model_config.get_freeze_filter())))
    return {"/".join(str(getattr(part, "key", getattr(part, "name", part))) for part in path)
            for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]}


def test_the_backbone_is_frozen_the_projection_trains_and_nothing_else_changes():
    plain, frozen = _trainable(_config()), _trainable(_config(dino_variant="dummy"))
    assert any(name.startswith("dino_proj/") for name in frozen)
    assert not any(name.startswith("dino/") for name in frozen)
    assert {name for name in frozen if not name.startswith("dino_proj/")} == plain

    trained = _trainable(_config(dino_variant="dummy", dino_trainable=True))
    assert any(name.startswith("dino/") for name in trained)


def test_the_models_dino_parameters_are_the_modules_parameter_tree():
    model_config = _config(dino_variant="dummy")
    module = dinov2.Dinov2(dinov2.get_config("dummy"), dtype_mm="float32")
    expected = flax.traverse_util.flatten_dict(
        jax.eval_shape(module.init, jax.random.key(0), jax.ShapeDtypeStruct((1, 224, 224, 3), jnp.float32))["params"], sep="/")
    shapes = jax.eval_shape(lambda rng: nnx.state(model_config.create(rng), nnx.Param).to_pure_dict(), jax.random.key(0))
    actual = {key.removeprefix("dino/"): leaf for key, leaf in flax.traverse_util.flatten_dict(shapes, sep="/").items()
              if key.startswith("dino/")}
    assert actual.keys() == expected.keys()
    assert all(actual[key].shape == expected[key].shape for key in expected)


@pytest.fixture
def checkpoint(monkeypatch):
    saved = {"PaliGemma": {"img": {"w": np.ones(3, np.float32)}}}
    monkeypatch.setattr(model_module, "restore_params", lambda *args, **kwargs: saved)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path, **kwargs: path)
    return saved


def _reference():
    return {"PaliGemma": {"img": {"w": jax.ShapeDtypeStruct((3,), np.float32)}},
            "dino": {"marker": jax.ShapeDtypeStruct((2,), np.float32)},
            "dino_proj": {"kernel": jax.ShapeDtypeStruct((32, 64), np.float32)}}


def test_the_loader_adds_hugging_face_weights_only_when_the_checkpoint_has_no_branch(checkpoint, monkeypatch):
    calls = []
    monkeypatch.setattr(dinov2, "load_hf_params", lambda source, cfg: calls.append(source) or {"marker": np.full(2, 7.0, np.float32)})

    loaded = weight_loaders.CheckpointWeightLoader("ignored", dino_variant="dummy", dino_weights="/models/dino").load(_reference())
    assert calls == ["/models/dino"]
    assert np.array_equal(loaded["dino"]["marker"], np.full(2, 7.0))
    assert isinstance(loaded["dino_proj"]["kernel"], jax.ShapeDtypeStruct)

    checkpoint["dino"] = {"marker": np.full(2, 3.0, np.float32)}
    kept = weight_loaders.CheckpointWeightLoader("ignored", dino_variant="dummy").load(_reference())
    assert calls == ["/models/dino"]
    assert np.array_equal(kept["dino"]["marker"], np.full(2, 3.0))


def test_the_dino_twin_is_the_base_recipe_plus_a_frozen_dinov2_base():
    base = config.get_config("pi05_yam_stream5_i20_bagging_right_real")
    twin = config.get_config("pi05_yam_stream5_i20_bagging_right_real_dino")
    assert (twin.model.dino_variant, twin.weight_loader.dino_variant) == ("base", "base")
    assert (base.model.dino_variant, base.weight_loader.dino_variant) == (None, None)

    shapes = jax.eval_shape(lambda rng: nnx.state(twin.model.create(rng), nnx.Param).to_pure_dict(), jax.random.key(0))
    flat = flax.traverse_util.flatten_dict(shapes, sep="/")
    assert flat["dino/layer_11/q/kernel"].shape == (768, 768)
    assert flat["dino_proj/kernel"].shape == (768, 2048)
    backbone = sum(int(np.prod(leaf.shape)) for key, leaf in flat.items() if key.startswith("dino/"))
    assert 85e6 < backbone < 88e6
