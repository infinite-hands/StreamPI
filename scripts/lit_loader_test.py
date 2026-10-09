"""Weight loading for LIT: the opt-in missing regex, the stage-1 -> stage-2 handoff and its fingerprint, and the
refusal to load a LIT checkpoint into a stock (lit="off") model, at both layers that load params: `Pi0Config.load` (what
serving uses) and the trainer's `CheckpointWeightLoader`.

Parameter trees are the tiny dummy variants' (lit_test_utils) with random values, saved and restored through orbax like
a real checkpoint; the image tower is the stub encoder (one Dense layer) because only the tree matters here.
"""
# ruff: noqa: SLF001

import functools
import os
import re

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import lit_test_utils as _utils
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding
from openpi.training import weight_loaders

from . import train
from . import lit_train_test as _train_test  # noqa: F401  (shares the batch and config helpers)


@pytest.fixture
def stub():
    with _utils.stub_image_encoder():
        yield


no_shared_compile_cache = _train_test.no_shared_compile_cache  # train.main's cache (see lit_train_test.py)


def _model_config(stage):
    return _utils.make_tiny_config(lit=stage, hist_horizon=1, dtype="float32")


def _shapes(stage) -> dict:
    """The pure dict of ShapeDtypeStruct params of a stage's tiny model: what the trainer hands a weight loader."""
    config = _model_config(stage)
    return nnx.state(nnx.eval_shape(lambda: config.create(jax.random.key(0))), nnx.Param).to_pure_dict()


def _checkpoint(tmp_path, stage, seed=0, name=None) -> tuple[str, dict]:
    """(params path, the params) of a random tiny checkpoint of a stage, written through orbax as the trainer does."""
    rng = np.random.default_rng(seed)
    params = jax.tree.map(lambda s: rng.normal(size=s.shape).astype(s.dtype), _shapes(stage))
    path = tmp_path / (name or f"{stage}_{seed}") / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": params})
    return str(path), params


def _flat(tree) -> dict:
    return traverse_util.flatten_dict(tree, sep="/")


# direction -> does it load? (checkpoint stage, target stage)
_DIRECTIONS = [
    ("stage2", "off", False),
    ("stage1", "off", False),
    ("off", "stage2", False),
    ("stage1", "stage2", False),  # without the opt-in regex
    ("stage2", "stage1", False),
    ("off", "off", True),
    ("stage1", "stage1", True),
    ("stage2", "stage2", True),
]


@pytest.mark.parametrize(("source", "target", "loads"), _DIRECTIONS)
def test_model_load_refuses_a_lit_checkpoint_in_a_stock_model_and_every_other_mismatch(
    stub, tmp_path, source, target, loads
):
    """Pi0Config.load (serving): drops extras and used to drop a LIT checkpoint's lit_* leaves without a word."""
    _, params = _checkpoint(tmp_path, source)
    config = _model_config(target)
    if loads:
        assert config.load(params).lit == target
        return
    with pytest.raises(ValueError) as error:
        config.load(params)
    if target == "off":
        assert "LIT" in str(error.value) and "lit_" in str(error.value)
        assert "silently" in str(error.value)


@pytest.mark.parametrize(("source", "target", "loads"), _DIRECTIONS)
def test_the_trainer_loader_refuses_a_lit_checkpoint_in_a_stock_model_and_every_other_mismatch(
    stub, tmp_path, source, target, loads
):
    """The trainer path: CheckpointWeightLoader + the pytree equality train.py demands."""
    path, _ = _checkpoint(tmp_path, source)
    loader = weight_loaders.CheckpointWeightLoader(path)
    shapes = _shapes(target)
    if loads:
        loaded = train._load_weights_and_validate(loader, shapes)
        assert sorted(_flat(loaded)) == sorted(_flat(shapes))
        return
    with pytest.raises(ValueError) as error:
        train._load_weights_and_validate(loader, shapes)
    if target == "off":
        assert "LIT" in str(error.value) and "lit_" in str(error.value)


def test_the_opt_in_regex_lets_a_stage2_config_take_a_checkpoint_without_its_lit_modules(stub, tmp_path):
    shapes = _shapes("stage2")
    path, params = _checkpoint(tmp_path, "off")  # a pi05_base-like stock tree
    loader = weight_loaders.CheckpointWeightLoader(path, missing_regex=weight_loaders.LIT_MISSING_REGEX)
    loaded = _flat(train._load_weights_and_validate(loader, shapes))
    assert set(loaded) == {k for k in _flat(shapes) if not k.startswith("lit_")}
    for key, value in loaded.items():
        np.testing.assert_array_equal(value, _flat(params)[key])
    # the default stays LoRA-only
    assert weight_loaders.CheckpointWeightLoader("x").missing_regex == ".*lora.*"
    assert weight_loaders.CheckpointWeightLoader("x") == weight_loaders.CheckpointWeightLoader("x", ".*lora.*")


def test_the_regex_loader_refuses_a_stage1_checkpoint_unless_the_target_is_stage1(stub, tmp_path):
    """litlite takes --weight-loader.params-path: a stage-1 checkpoint handed to it would lose its goal encoder
    silently (the handoff loader's fingerprint does not run on the generic loader)."""
    path, _ = _checkpoint(tmp_path, "stage1")
    for regex in (weight_loaders.LIT_MISSING_REGEX, ".*"):
        loader = weight_loaders.CheckpointWeightLoader(path, missing_regex=regex)
        with pytest.raises(ValueError, match="stage-1 one.*LitStage1WeightLoader"):
            loader.load(_shapes("stage2"))
    loader = weight_loaders.CheckpointWeightLoader(path, missing_regex=weight_loaders.LIT_MISSING_REGEX)
    assert "lit_goal_encoder" in loader.load(_shapes("stage1")), "a stage-1 target keeps its checkpoint's encoder"
    stage2_path, _ = _checkpoint(tmp_path, "stage2")
    stage2_loader = weight_loaders.CheckpointWeightLoader(stage2_path, missing_regex=weight_loaders.LIT_MISSING_REGEX)
    assert "lit_aggregator" in stage2_loader.load(_shapes("stage2"))


def test_only_a_regex_that_admits_lit_modules_gets_the_stage1_refusal(stub, tmp_path):
    """The default loader keeps its behaviour (its stage-1 -> stage-2 refusal is the trainer's pytree check)."""
    assert weight_loaders._admits_lit_modules(weight_loaders.LIT_MISSING_REGEX)
    assert weight_loaders._admits_lit_modules(".*")
    assert not weight_loaders._admits_lit_modules(weight_loaders.CheckpointWeightLoader("x").missing_regex)
    path, _ = _checkpoint(tmp_path, "stage1")
    with pytest.raises(ValueError, match="PyTrees have different structure"):
        train._load_weights_and_validate(weight_loaders.CheckpointWeightLoader(path), _shapes("stage2"))


def test_the_lit_missing_regex_is_anchored():
    pattern = re.compile(weight_loaders.LIT_MISSING_REGEX)
    assert pattern.fullmatch("lit_aggregator/groups/to_key/kernel")
    assert pattern.fullmatch("lit_pose_decoder/fc1/bias")
    assert pattern.fullmatch("PaliGemma/llm/layers/attn/q_einsum/lora_a")
    # a path that merely contains "lit_" is not a LIT module: an unanchored `.*lit_.*` would tolerate its absence
    assert not pattern.fullmatch("PaliGemma/llm/layers/split_heads/kernel")
    assert re.fullmatch(".*lit_.*", "PaliGemma/llm/layers/split_heads/kernel")


def test_no_stock_parameter_path_matches_the_lit_missing_regex():
    """At real dimensions (abstract): every leaf of the stock model is either LoRA or required."""
    config = pi0_config.Pi0Config(pi05=True, action_horizon=30, hist_horizon=5, max_token_len=200)
    stock = nnx.state(nnx.eval_shape(lambda: config.create(jax.random.key(0))), nnx.Param).flat_state()
    pattern = re.compile(weight_loaders.LIT_MISSING_REGEX)
    assert not [p for p in ("/".join(map(str, path)) for path in stock) if pattern.fullmatch(p)]
    lora = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", max_token_len=200
    )
    lora_paths = [
        "/".join(map(str, path))
        for path in nnx.state(nnx.eval_shape(lambda: lora.create(jax.random.key(0))), nnx.Param).flat_state()
    ]
    matching = [p for p in lora_paths if pattern.fullmatch(p)]
    assert matching and all("lora" in p for p in matching)


# ---- the stage-1 handoff ----


def test_the_handoff_loader_keeps_every_backbone_and_expert_leaf_and_drops_the_goal_encoder(stub, tmp_path):
    path, params = _checkpoint(tmp_path, "stage1")
    shapes = _shapes("stage2")
    loaded = _flat(train._load_weights_and_validate(weight_loaders.LitStage1WeightLoader(path), shapes))
    checkpoint = _flat(params)
    assert set(loaded) == {k for k in checkpoint if not k.startswith("lit_")} == {
        k for k in _flat(shapes) if not k.startswith("lit_")
    }
    assert any(k.startswith("lit_goal_encoder/") for k in checkpoint)
    assert not any(k.startswith("lit_") for k in loaded), "the stage-2 lit_* modules start fresh; no goal encoder"
    for key, value in loaded.items():
        np.testing.assert_array_equal(value, checkpoint[key], err_msg=key)


def test_the_handoff_loader_without_a_path_says_what_to_pass(stub):
    with pytest.raises(ValueError, match="--weight-loader.params-path"):
        weight_loaders.LitStage1WeightLoader().load(_shapes("stage2"))


@pytest.mark.parametrize(
    ("source", "match"),
    [("off", "not a LIT stage-1 checkpoint"), ("stage2", "not a LIT stage-1 checkpoint")],
)
def test_the_handoff_loader_refuses_a_checkpoint_that_is_not_stage1(stub, tmp_path, source, match):
    path, _ = _checkpoint(tmp_path, source)
    with pytest.raises(ValueError, match=match):
        weight_loaders.LitStage1WeightLoader(path).load(_shapes("stage2"))


@pytest.mark.parametrize("target", ["off", "stage1"])
def test_the_handoff_loader_refuses_a_target_that_is_not_stage2(stub, tmp_path, target):
    path, _ = _checkpoint(tmp_path, "stage1")
    with pytest.raises(ValueError, match="lit='stage2'"):
        weight_loaders.LitStage1WeightLoader(path).load(_shapes(target))


def _save_flat(tmp_path, name, flat) -> str:
    path = tmp_path / name / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": traverse_util.unflatten_dict(flat, sep="/")})
    return str(path)


@pytest.mark.parametrize(
    "extra",
    [
        "PaliGemma/llm/layers/attn/some_extra_module/kernel",  # a module the model does not have
        "PaliGemma/llm/layers/attn/q_einsum/lora_a",  # LoRA leaves of a LoRA checkpoint, in a full fine-tune model
        "race_embedding",  # a foreign top-level module, e.g. from a merged branch
    ],
)
def test_the_handoff_loader_refuses_a_checkpoint_leaf_the_model_does_not_have(stub, tmp_path, extra):
    """The merge keeps only the model's leaves and its self-check iterates the model's leaves, so a leaf the checkpoint
    has and the model lacks used to vanish without a word; only the stage-1 goal encoder is meant to."""
    _, params = _checkpoint(tmp_path, "stage1")
    shapes = _shapes("stage2")
    honest = weight_loaders.LitStage1WeightLoader(_save_flat(tmp_path, "honest", _flat(params)))
    assert honest.load(shapes)
    path = _save_flat(tmp_path, "extra", {**_flat(params), extra: np.zeros((3, 3), np.float32)})
    with pytest.raises(ValueError, match=r"has 1 leaves the stage-2 model does not have.*dropped silently") as error:
        weight_loaders.LitStage1WeightLoader(path).load(shapes)
    assert extra in str(error.value)


def test_a_params_path_one_level_off_says_it_must_end_in_step_params(stub, tmp_path):
    """The step directory and the experiment directory made orbax raise a FileNotFoundError about `_METADATA` after the
    container had started; a missing path gets the same hint. Both loaders."""
    path, _ = _checkpoint(tmp_path, "stage1", name="exp/7")
    step_dir, experiment = os.path.dirname(path), os.path.dirname(os.path.dirname(path))
    nowhere = os.path.join(experiment, "8", "params")
    for loader_for in (
        weight_loaders.LitStage1WeightLoader,
        lambda p: weight_loaders.CheckpointWeightLoader(p, missing_regex=weight_loaders.LIT_MISSING_REGEX),
    ):
        for wrong in (step_dir, experiment, nowhere):
            with pytest.raises(FileNotFoundError, match=r"must end in <step>/params") as error:
                loader_for(wrong).load(_shapes("stage2"))
            assert wrong in str(error.value)
    assert weight_loaders.LitStage1WeightLoader(path).load(_shapes("stage2"))


def _expert_and_backbone_keys(tree):
    keys = [k for k in _flat(tree) if k.startswith("PaliGemma/llm/")]
    return next(k for k in keys if "_1" in k), next(k for k in keys if "_1" not in k)


def test_the_fingerprint_catches_a_merge_that_does_not_hand_off_what_it_should(stub, tmp_path, monkeypatch):
    """The loader checks its result instead of trusting the merge: corrupt the merge four ways, each must raise."""
    path, params = _checkpoint(tmp_path, "stage1")
    shapes = _shapes("stage2")
    loader = weight_loaders.LitStage1WeightLoader(path)
    assert loader.load(shapes)  # the honest merge passes
    expert, backbone = _expert_and_backbone_keys(shapes)
    real_merge = weight_loaders._merge_params

    def corrupted(change):
        def merge(loaded, reference, *, missing_regex):
            flat = _flat(real_merge(loaded, reference, missing_regex=missing_regex))
            change(flat)
            return traverse_util.unflatten_dict(flat, sep="/")

        return merge

    def perturb(flat):
        flat[backbone] = flat[backbone] + 1.0

    def fresh_expert(flat):
        flat[expert] = _flat(shapes)[expert]  # the expert left at its own initialisation

    def loaded_lit(flat):
        key = next(k for k in _flat(shapes) if k.startswith("lit_aggregator/"))
        flat[key] = np.zeros(_flat(shapes)[key].shape, np.float32)

    def goal_encoder_survives(flat):
        flat["lit_goal_encoder/fc1/kernel"] = np.zeros((2, 2), np.float32)

    for change, match in [
        (perturb, "not the stage-1 checkpoint's value"),
        (fresh_expert, "not the stage-1 checkpoint's value"),
        (loaded_lit, "must start fresh"),
        (goal_encoder_survives, "survived"),
    ]:
        monkeypatch.setattr(weight_loaders, "_merge_params", corrupted(change))
        with pytest.raises(ValueError, match=match):
            loader.load(shapes)
    monkeypatch.setattr(weight_loaders, "_merge_params", real_merge)

    # a checkpoint that lacks a backbone leaf: neither loaded nor LoRA, so the handoff is incomplete (not left to the
    # trainer's later structure error)
    incomplete = {k: v for k, v in _flat(params).items() if k != backbone}
    with pytest.raises(ValueError, match="incomplete"):
        weight_loaders.LitStage1WeightLoader(_save_flat(tmp_path, "short", incomplete)).load(shapes)


# ---- end to end through the trainer: a real stage-1 run, then a stage-2 train state initialised from it ----


def test_a_stage1_run_hands_off_to_a_stage2_train_state(stub, tmp_path, monkeypatch):
    stage1 = _train_test._main_config(tmp_path, "stage1", num_train_steps=2)
    batch = _train_test._batch(stage1.model)
    monkeypatch.setattr(_data_loader, "create_data_loader", lambda *a, **k: _train_test._Loader(batch))
    monkeypatch.setattr(train, "init_logging", lambda: None)
    train.main(stage1)
    saved = stage1.checkpoint_dir / "1" / "params"
    assert saved.exists()
    checkpoint = _flat(_model.restore_params(saved, restore_type=np.ndarray))
    assert any(k.startswith("lit_goal_encoder/") for k in checkpoint)

    stage2_model = _train_test._model_config("stage2")
    stage2 = _train_test._train_config(
        stage2_model, ema_decay=0.99, weight_loader=weight_loaders.LitStage1WeightLoader(str(saved))
    )
    init_rng = jax.random.key(7)
    state, _ = train.init_train_state(stage2, init_rng, sharding.make_mesh(1), resume=False)
    params = _train_test._leaves(state.params)
    assert not any(k.startswith("lit_goal_encoder/") for k in params)

    # the stage-2 train state is the model's own initialisation (same rng as init_train_state's) with the stage-1
    # checkpoint's backbone and expert in it
    fresh = _train_test._leaves(nnx.state(stage2_model.create(jax.random.split(init_rng)[1]), nnx.Param))
    assert sorted(params) == sorted(fresh)
    loaded = fresh_lit = 0
    for key, value in params.items():
        if key.startswith("lit_"):
            np.testing.assert_array_equal(value, fresh[key], err_msg=f"{key} is not the fresh initialisation")
            fresh_lit += 1
        else:
            np.testing.assert_array_equal(value, checkpoint[key].astype(value.dtype), err_msg=key)
            loaded += 1
    assert loaded and fresh_lit
    for key, value in _train_test._leaves(state.ema_params).items():
        np.testing.assert_array_equal(value, params[key], err_msg=f"ema {key}")

    # and it trains: one stage-2 step is finite, with a real pose loss
    step = jax.jit(functools.partial(train.train_step, stage2))
    new_state, info = step(jax.random.key(0), state, _train_test._batch(stage2_model))
    assert np.isfinite(float(info["loss"])) and float(info["pose_loss"]) > 0.0
    assert int(new_state.step) == 1
