"""The LIT part of the trainer: the loss with its pose term, the logged parts and per-module gradient norms, the stage-2
guard, the EMA of frozen leaves, the freeze-filter check, the Step line, and the checkpoint round trip with resume.

Every model here is a tiny dummy variant (lit_test_utils) with the stub image encoder (`stub_image_encoder`: one Dense
layer instead of SigLIP So400m). These tests are about the optimizer, the pytrees and the logging, not the image tower,
and a train step with gradients through the real SigLIP does not fit this CPU next to another JAX process. History
length 1 keeps the prefix at 776 tokens. Where a LIT leaf must get a gradient the zero-initialised leaves are given
noise first (a freshly initialised pi0.5 is prefix-blind: its adaRMS gates start at zero).
"""
# ruff: noqa: SLF001

import dataclasses
import functools
import inspect
import logging
import os
import pathlib
import re

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import lit_test_utils as _utils
from openpi.models import model as _model
from openpi.models import pi0
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import lit_train as _lit_train
from openpi.training import optimizer as _optimizer
from openpi.training import sharding
from openpi.training import utils as training_utils

from . import train, train_multi_node

SCRIPTS = (train, train_multi_node)
LR = _optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=2e-3, decay_steps=20, decay_lr=1e-3)


@pytest.fixture(scope="module", autouse=True)
def _stub_image_encoder():
    with _utils.stub_image_encoder():
        yield


@pytest.fixture(autouse=True)
def no_shared_compile_cache(monkeypatch):
    """train.main points JAX's persistent compilation cache at ~/.cache/jax. On the Mac these tests were written on,
    that directory (shared with other JAX processes) handed back executables that computed something else for the same
    program (a different step-0 loss on each run, then a segfault) until it was bypassed, so the tests keep the cache
    off: main's update of that one option is ignored."""
    real_update = jax.config.update

    def update(name, value):
        if name != "jax_compilation_cache_dir":
            real_update(name, value)

    monkeypatch.setattr(jax.config, "update", update)


def _model_config(lit="off", *, lora=False, dtype="bfloat16", **overrides):
    variants = {"paligemma_variant": "dummy", "action_expert_variant": "dummy"}
    return _utils.make_tiny_config(lit=lit, dtype=dtype, hist_horizon=1, **variants, **overrides)


def _train_config(model, *, ema_decay: float | None = 0.99, freeze=None, **overrides) -> _config.TrainConfig:
    return _config.TrainConfig(
        name="lit_tiny",
        exp_name="lit_tiny",
        model=model,
        data=_config.FakeDataConfig(),
        freeze_filter=model.get_freeze_filter() if freeze is None else freeze,
        ema_decay=ema_decay,
        lr_schedule=LR,
        batch_size=_utils.BATCH,
        num_workers=0,
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
        fsdp_devices=1,
        wandb_enabled=False,
        **overrides,
    )


def _batch(model, seed=0, goal_mask=(True, True)):
    fields = {}
    if model.lit != "off":
        goal = np.zeros((_utils.BATCH, model.action_dim), np.float32)
        goal[:, : _utils.ROBOT_DIMS] = np.random.default_rng(seed + 10).normal(size=(_utils.BATCH, _utils.ROBOT_DIMS))
        fields = {"lit_goal": jnp.asarray(goal), "lit_goal_mask": jnp.asarray(goal_mask, jnp.bool_)}
    return _utils.make_observation(seed, config=model, **fields), _utils.make_actions(seed + 1, config=model)


def _randomize(state: training_utils.TrainState, seed=0) -> training_utils.TrainState:
    """`state` with every all-zero floating leaf of the params given noise (the EMA follows; rebuild the optimizer)."""
    model = nnx.merge(state.model_def, state.params)
    params = nnx.state(_utils.randomize_zero_init(model, seed))
    return dataclasses.replace(
        state,
        params=params,
        ema_params=None if state.ema_params is None else params,
    )


def _init_state(config, *, randomize=True, seed=0):
    mesh = sharding.make_mesh(1)
    state, state_sharding = train.init_train_state(config, jax.random.key(seed), mesh, resume=False)
    if randomize:
        state = _randomize(state, seed)
        tx = state.tx
        state = dataclasses.replace(state, opt_state=tx.init(state.params.filter(config.trainable_filter)))
    return state, state_sharding, mesh


@functools.cache
def _step_fn(script_name, config):
    script = {s.__name__.rsplit(".", 1)[-1]: s for s in SCRIPTS}[script_name]
    return jax.jit(functools.partial(script.train_step, config))


def _leaves(state: nnx.State) -> dict:
    return {"/".join(map(str, p)): np.asarray(v.value) for p, v in state.flat_state().items()}


def _equal_trees(a, b, what=""):
    fa, fb = _leaves(a), _leaves(b)
    assert sorted(fa) == sorted(fb), what
    for path in fa:
        np.testing.assert_array_equal(fa[path], fb[path], err_msg=f"{what} {path}")


# ---- the stock step: lit=off must not change ----


def _pin_train_step(config, rng, state, batch):
    """scripts/train.py's train_step as it is at the pin 96d891f2, verbatim, but for the module prefix."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _run(fn, state, batches, rng=0):
    infos = []
    for batch in batches:
        state, info = fn(jax.random.key(rng), state, batch)
        infos.append({k: np.asarray(v) for k, v in info.items()})
    return state, infos


@pytest.mark.parametrize("script", [s.__name__.rsplit(".", 1)[-1] for s in SCRIPTS])
def test_the_stock_step_is_bit_identical_to_the_pins_with_ema(script):
    """lit=off, full fine-tune, EMA on (what the *_full recipes run): the new train_step is the pin's, bit for bit:
    loss, grad and param norms, params, EMA params and optimizer state after three steps."""
    config = _train_config(_model_config(), ema_decay=0.99)
    state, _, _ = _init_state(config)
    batches = [_batch(config.model, seed) for seed in range(3)]
    new_state, new_info = _run(_step_fn(script, config), state, batches)
    old_state, old_info = _run(jax.jit(functools.partial(_pin_train_step, config)), state, batches)
    assert sorted(new_info[0]) == sorted(old_info[0]) == ["grad_norm", "loss", "param_norm"]
    for new, old in zip(new_info, old_info, strict=True):
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
    _equal_trees(new_state.params, old_state.params, "params")
    _equal_trees(new_state.ema_params, old_state.ema_params, "ema")
    for a, b in zip(jax.tree.leaves(new_state.opt_state), jax.tree.leaves(old_state.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert int(new_state.step) == 3


def test_the_stock_lora_step_is_bit_identical_to_the_pins_without_ema():
    variants = {"paligemma_variant": "dummy", "action_expert_variant": "dummy"}
    model = _utils.make_tiny_config(dtype="bfloat16", hist_horizon=1, **variants)
    # a LoRA config freezes the base weights and keeps no EMA (its recipes set ema_decay=None)
    freeze = nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
    config = _train_config(model, ema_decay=None, freeze=freeze)
    state, _, _ = _init_state(config)
    batches = [_batch(model, seed) for seed in range(2)]
    new_state, new_info = _run(_step_fn("train", config), state, batches)
    old_state, old_info = _run(jax.jit(functools.partial(_pin_train_step, config)), state, batches)
    for new, old in zip(new_info, old_info, strict=True):
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
    _equal_trees(new_state.params, old_state.params, "params")
    assert new_state.ema_params is None


def test_the_two_scripts_share_one_train_step_and_both_write_the_step_line_to_the_log():
    assert inspect.getsource(train.train_step) == inspect.getsource(train_multi_node.train_step)
    for script in SCRIPTS:
        source = inspect.getsource(script.main)
        assert 'pbar.write(f"Step {step}: {info_str}")' in source
        assert 'logging.info(f"Step {step}: {info_str}")' in source


# ---- the LIT loss: the pose term once, the parts that are logged ----


class _Stub:
    """Just what loss_with_parts reads of a model."""

    def __init__(self, lit, aux, *, weight=0.3, action=2.0):
        self.lit, self.lit_pose_weight, self._aux, self._action = lit, weight, aux, action

    def compute_loss_and_aux(self, rng, observation, actions, *, train):
        assert train
        return jnp.full((2, 4), self._action), self._aux


def test_the_pose_loss_enters_the_total_once_with_its_weight_and_stage1_has_none():
    aux = {"pose_loss": jnp.float32(10.0), "pose_copy_baseline": jnp.float32(1.0)}
    total, parts = _lit_train.loss_with_parts(_Stub("stage2", aux, weight=0.3), None, None, None)
    assert float(total) == pytest.approx(2.0 + 0.3 * 10.0, rel=1e-6)
    assert float(parts["action_loss"]) == 2.0
    assert sorted(parts) == ["action_loss", "pose_copy_baseline", "pose_loss"]
    assert float(parts["pose_loss"]) == 10.0, "the logged pose loss is the unweighted one"
    twice, _ = _lit_train.loss_with_parts(_Stub("stage2", aux, weight=0.6), None, None, None)
    assert float(twice) == pytest.approx(2.0 + 0.6 * 10.0, rel=1e-6)
    zero, _ = _lit_train.loss_with_parts(_Stub("stage2", aux, weight=0.0), None, None, None)
    assert float(zero) == 2.0
    # stage 1 has no pose loss: the total is the action loss, the baseline is only logged
    stage1 = _Stub("stage1", {"pose_copy_baseline": jnp.float32(1.0)})
    total, parts = _lit_train.loss_with_parts(stage1, None, None, None)
    assert float(total) == 2.0
    assert sorted(parts) == ["action_loss", "pose_copy_baseline"]


def test_a_stage2_model_without_a_pose_loss_is_refused():
    with pytest.raises(ValueError, match="pose"):
        _lit_train.loss_with_parts(_Stub("stage2", {"pose_copy_baseline": jnp.float32(1.0)}), None, None, None)
    with pytest.raises(ValueError, match="pose"):
        _lit_train.loss_with_parts(_Stub("stage2", {}), None, None, None)


@functools.cache
def _lit_step(stage, script="train"):
    config = _train_config(_model_config(stage), ema_decay=0.99 if stage == "stage2" else None)
    state, _, _ = _init_state(config)
    fn = _step_fn(script, config)
    new_state, info = fn(jax.random.key(0), state, _batch(config.model))
    return config, state, new_state, {k: np.asarray(v) for k, v in info.items()}


_EXPECTED_KEYS = {
    "stage2": {
        "loss", "grad_norm", "param_norm", "action_loss", "pose_loss", "pose_copy_baseline",
        "grad_norm_backbone", "grad_norm_expert", "grad_norm_lit_aggregator", "grad_norm_lit_pose_decoder",
    },  # fmt: skip
    "stage1": {
        "loss", "grad_norm", "param_norm", "action_loss", "pose_copy_baseline",
        "grad_norm_expert", "grad_norm_lit_goal_encoder",
    },  # fmt: skip
}


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_the_train_step_logs_the_parts_and_the_module_gradient_norms(stage):
    config, _, _, info = _lit_step(stage)
    assert set(info) == _EXPECTED_KEYS[stage]
    assert all(np.isfinite(v) and v.shape == () for v in info.values())
    for key, value in info.items():
        if key.startswith("grad_norm_"):
            assert value > 0.0, key
    if stage == "stage2":
        weight = config.model.lit_pose_weight
        assert info["loss"] == pytest.approx(info["action_loss"] + weight * info["pose_loss"], rel=1e-5)
        assert info["pose_loss"] > 0.0 and info["pose_copy_baseline"] > 0.0
    else:
        assert info["loss"] == pytest.approx(info["action_loss"], rel=1e-6)


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_the_module_norms_partition_the_gradient(stage):
    """Every trainable leaf is in exactly one module: the module norms recompose the global gradient norm."""
    config, _, _, info = _lit_step(stage)
    modules = [v for k, v in info.items() if k.startswith("grad_norm_")]
    assert float(np.sqrt(np.sum(np.square(modules)))) == pytest.approx(float(info["grad_norm"]), rel=1e-5)


def test_module_of_every_parameter_path_of_every_stage():
    for stage in ("off", "stage1", "stage2"):
        model = nnx.eval_shape(lambda stage=stage: _model_config(stage).create(jax.random.key(0)))
        grads = nnx.state(model, nnx.Param)
        norms = _lit_train.module_grad_norms(jax.tree.map(lambda s: jnp.zeros(s.shape, jnp.float32), grads))
        expected = {"off": {"backbone", "expert"}, "stage1": {"backbone", "expert", "lit_goal_encoder"}}.get(
            stage, {"backbone", "expert", "lit_aggregator", "lit_pose_decoder"}
        )
        assert set(norms) == {f"grad_norm_{name}" for name in expected}, stage
    # the expert is the *_1 llm leaves and the action projections; SigLIP and the 2B language model are the backbone
    assert _lit_train._MODULES[3][1].fullmatch("PaliGemma/llm/layers/attn/q_einsum_1/w")
    assert _lit_train._MODULES[3][1].fullmatch("action_out_proj/kernel")
    assert not _lit_train._MODULES[3][1].fullmatch("PaliGemma/llm/layers/attn/q_einsum/w")
    assert not _lit_train._MODULES[3][1].fullmatch("PaliGemma/img/head/kernel")


def test_the_total_gradient_is_the_action_gradient_plus_the_weighted_pose_gradient():
    """The weighted pose loss is in the gradient exactly once: grad(total) = grad(action) + weight * grad(pose), in
    float32 (in bfloat16 two backward passes round differently)."""
    config = _train_config(_model_config("stage2", dtype="float32"), ema_decay=None)
    state, _, _ = _init_state(config)
    observation, actions = _batch(config.model)
    diff = nnx.DiffState(0, config.trainable_filter)
    rng = jax.random.key(3)

    @jax.jit
    def three_gradients(params):
        model = nnx.merge(state.model_def, params)
        model.train()

        def aux_of(m):
            chunked, aux = m.compute_loss_and_aux(rng, observation, actions, train=True)
            return jnp.mean(chunked), aux["pose_loss"]

        action = nnx.grad(lambda m: aux_of(m)[0], argnums=diff)(model)
        pose = nnx.grad(lambda m: aux_of(m)[1], argnums=diff)(model)
        total = nnx.grad(lambda m: _lit_train.loss_with_parts(m, rng, observation, actions)[0], argnums=diff)(model)
        return action, pose, total

    action, pose, total = (_leaves(g) for g in three_gradients(state.params))
    weight = config.model.lit_pose_weight
    expected = {path: action[path] + weight * pose[path] for path in total}
    # relative to the largest gradient entry of the whole model: a leaf whose gradient is mathematically zero (the
    # aggregator's attention key bias, over which softmax is invariant) is float noise on its own scale
    scale = max(float(np.max(np.abs(g))) for g in expected.values())
    errors = {path: float(np.max(np.abs(total[path] - expected[path]))) / scale for path in total}
    worst_path = max(errors, key=errors.get)
    print(f"total gradient vs action + {weight} * pose: worst {errors[worst_path]:.2e} of the largest ({worst_path})")
    assert errors[worst_path] <= 1e-4, worst_path
    # and the pose gradient is real: it reaches the aggregator and the pose decoder
    assert any(np.any(g) for p, g in pose.items() if p.startswith("lit_aggregator"))
    assert any(np.any(g) for p, g in pose.items() if p.startswith("lit_pose_decoder"))
    assert not any(np.any(g) for p, g in pose.items() if p.startswith("PaliGemma/llm") and "_1" in p), (
        "the pose loss does not reach the expert"
    )


def test_the_stage2_guard_fires_inside_the_train_step(monkeypatch):
    config = _train_config(_model_config("stage2"))
    state, _, _ = _init_state(config)
    real = pi0.Pi0.compute_loss_and_aux

    def without_pose(self, *args, **kwargs):
        loss, aux = real(self, *args, **kwargs)
        return loss, {k: v for k, v in aux.items() if k != "pose_loss"}

    monkeypatch.setattr(pi0.Pi0, "compute_loss_and_aux", without_pose)
    with pytest.raises(ValueError, match="pose path is not wired"):
        jax.eval_shape(functools.partial(train.train_step, config), jax.random.key(0), state, _batch(config.model))


def test_the_train_step_of_both_scripts_agrees_on_a_lit_batch():
    config, state, new_state, info = _lit_step("stage2", "train")
    other_state, other = _run(_step_fn("train_multi_node", config), state, [_batch(config.model)])
    assert sorted(other[0]) == sorted(info)
    for key in info:
        np.testing.assert_array_equal(other[0][key], info[key], err_msg=key)
    _equal_trees(other_state.params, new_state.params)


# ---- the EMA ----


def test_ema_update_leaves_frozen_leaves_alone_where_the_stock_formula_shrinks_them(capsys):
    class Marker(nnx.Param):
        pass

    frozen = jax.random.normal(jax.random.key(0), (4096,)).astype(jnp.bfloat16)
    trained = jnp.full((4,), 1.0, jnp.float32)
    params = nnx.State({"frozen": nnx.VariableState(Marker, frozen), "trained": nnx.VariableState(Marker, trained)})
    only_trained = nnx_utils.PathRegex("trained")

    stock = jax.tree.map(lambda x: x, params)
    ours = jax.tree.map(lambda x: x, params)
    steps = 2000
    for _ in range(steps):
        stock = jax.tree.map(lambda old, new: 0.99 * old + (1 - 0.99) * new, stock, params)
        ours = training_utils.ema_update(ours, params, 0.99, only_trained)

    def rms_relative_error(x):
        x, ref = np.asarray(x, np.float32), np.asarray(frozen, np.float32)
        return float(np.sqrt(np.mean(np.square(x - ref))) / np.sqrt(np.mean(np.square(ref))))

    stock_error = rms_relative_error(stock["frozen"].value)
    with capsys.disabled():
        print(f"\nEMA of a frozen bf16 leaf after {steps} steps, rms relative error: stock {stock_error:.4f}, fixed 0")
    # bfloat16 0.99 and 0.01 sum to 0.99829: a leaf averaged against itself shrinks (the stock formula, measured here)
    assert stock_error > 0.01
    np.testing.assert_array_equal(np.asarray(ours["frozen"].value), np.asarray(frozen))
    assert ours["frozen"].value.dtype == jnp.bfloat16
    # a trainable leaf that did not move stays put too (float32 arithmetic)
    np.testing.assert_allclose(np.asarray(ours["trained"].value), 1.0, rtol=1e-5)


def test_ema_update_averages_a_trainable_leaf_like_the_stock_formula():
    class Marker(nnx.Param):
        pass

    old = nnx.State({"w": nnx.VariableState(Marker, jnp.full((3,), 1.0, jnp.float32))})
    new = nnx.State({"w": nnx.VariableState(Marker, jnp.full((3,), 3.0, jnp.float32))})
    out = training_utils.ema_update(old, new, 0.9, nnx_utils.PathRegex("w"))
    stock = jax.tree.map(lambda o, n: 0.9 * o + (1 - 0.9) * n, old, new)
    np.testing.assert_array_equal(np.asarray(out["w"].value), np.asarray(stock["w"].value))


@pytest.mark.parametrize("stage", ["stage1", "lora"])
def test_a_frozen_backbone_keeps_its_exact_weights_in_the_saved_ema(stage):
    """Stage 1 (frozen backbone, EMA on) and a LoRA config with an EMA override: after several steps every frozen
    EMA leaf is bit-equal to its initial value and the trainable EMA leaves moved."""
    if stage == "stage1":
        model = _model_config("stage1")
        freeze = model.get_freeze_filter()
    else:
        model = _model_config()
        freeze = nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
    config = _train_config(model, ema_decay=0.99, freeze=freeze)
    state, _, _ = _init_state(config)
    initial = _leaves(state.ema_params)
    final, _ = _run(_step_fn("train", config), state, [_batch(model, seed) for seed in range(3)])
    after = _leaves(final.ema_params)
    frozen_paths = {"/".join(map(str, p)) for p in state.params.filter(nnx.Not(config.trainable_filter)).flat_state()}
    assert frozen_paths
    assert any("llm" in p for p in frozen_paths)
    for path in frozen_paths:
        np.testing.assert_array_equal(after[path], initial[path], err_msg=path)
        assert after[path].dtype == jnp.bfloat16, path
    moved = [p for p in after if p not in frozen_paths and not np.array_equal(after[p], initial[p])]
    assert moved


# ---- the freeze-filter check of a LIT TrainConfig ----


def test_a_lit_train_config_must_freeze_what_the_model_says():
    model = _model_config("stage1")
    ok = _train_config(model)
    assert ok.freeze_filter == model.get_freeze_filter()
    # nothing frozen (the default): a stage-1 run that would train every leaf, SigLIP and the backbone included
    with pytest.raises(ValueError, match="freeze_filter"):
        _train_config(model, freeze=nnx.Nothing)
    # the expert frozen as well
    with pytest.raises(ValueError, match="freeze_filter"):
        _train_config(model, freeze=nnx_utils.PathRegex(".*llm.*|PaliGemma/img/.*"))
    # a different filter that selects the same leaves passes
    same = nnx.Any(
        nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
        nnx_utils.PathRegex("PaliGemma/img/.*"),
    )
    assert same != model.get_freeze_filter()
    _train_config(model, freeze=same)
    # stage 2 trains everything: freezing the backbone is refused, and so is the stock default filter of lit=off
    stage2 = _model_config("stage2")
    _train_config(stage2)
    with pytest.raises(ValueError, match="freeze_filter"):
        _train_config(stage2, freeze=nnx_utils.PathRegex(".*llm.*"))
    # lit=off configs are never checked
    _train_config(_model_config(), freeze=nnx_utils.PathRegex(".*llm.*"))
    # dataclasses.replace re-runs the check
    with pytest.raises(ValueError, match="freeze_filter"):
        dataclasses.replace(ok, freeze_filter=nnx.Nothing)


# ---- the Step line, and the checkpoint round trip with resume ----


class _Loader:
    """A data loader that cycles one batch (what train.main needs of it)."""

    def __init__(self, batch):
        self._batch = batch

    def data_config(self):
        return _config.DataConfig(repo_id="fake")

    def __iter__(self):
        while True:
            yield self._batch


_STEP_LINE = re.compile(r"(?:^|\s)Step\s+(\d+):\s*(.+)$")  # infinite-hands source/training/metrics.py
_STEP_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[\d.]+(?:[eE][-+]?\d+)?)")


def _parse(line: str):
    match = _STEP_LINE.search(line)
    return None if match is None else (int(match.group(1)), dict(_STEP_PAIR.findall(match.group(2))))


def _main_config(tmp_path, stage, **overrides):
    model = _model_config(stage)
    return dataclasses.replace(
        _train_config(model, ema_decay=0.99 if stage == "stage2" else None),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        overwrite=True,  # a new run: the trainer creates the directory first and refuses it without overwrite or resume
        resume=False,
        **overrides,
    )


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_main_writes_numeric_step_lines_to_the_log(tmp_path, monkeypatch, caplog, script, stage):
    config = _main_config(tmp_path, stage)
    monkeypatch.setattr(_data_loader, "create_data_loader", lambda *a, **k: _Loader(_batch(config.model)))
    monkeypatch.setattr(script, "init_logging", lambda: None)
    with caplog.at_level(logging.INFO):
        script.main(config)
    lines = [_parse(r.getMessage()) for r in caplog.records]
    parsed = {step: pairs for step, pairs in (line for line in lines if line is not None)}
    assert sorted(parsed) == [0, 1], "one Step line per log_interval, both captured by the logging system"
    for pairs in parsed.values():
        assert set(pairs) == _EXPECTED_KEYS[stage]
        assert all(np.isfinite(float(v)) for v in pairs.values())
    assert any(line is not None for line in lines)


def _manager(root: pathlib.Path):
    return _checkpoints.initialize_checkpoint_dir(root, keep_period=None, overwrite=False, resume=True)[0]


def _step_dirs(root: pathlib.Path):
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def test_orbax_round_trip_and_resume_of_a_stage2_state_with_ema(tmp_path, monkeypatch):
    """Save a stage-2 train state (EMA included) through the trainer's own checkpoint code, restore it, compare every
    leaf, and train on: the resumed run is the uninterrupted run, bit for bit."""
    config = _main_config(tmp_path, "stage2", num_train_steps=2)
    batch = _batch(config.model)
    monkeypatch.setattr(_data_loader, "create_data_loader", lambda *a, **k: _Loader(batch))
    monkeypatch.setattr(train, "init_logging", lambda: None)

    train.main(config)  # steps 0 and 1; a checkpoint at the last step
    root = config.checkpoint_dir
    assert _step_dirs(root)[-1] == 1

    # restore through the trainer's code path
    mesh = sharding.make_mesh(1)
    shape, _ = train.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
    restored = _checkpoints.restore_state(_manager(root), shape, _Loader(batch))
    assert int(restored.step) == 2
    assert restored.ema_params is not None

    # the params item of a stage-2 checkpoint is the EMA, with the lit_* leaves in it
    saved = _leaves(restored.ema_params)
    assert any(p.startswith("lit_aggregator/") for p in saved) and any(p.startswith("lit_pose_decoder/") for p in saved)
    assert not any(p.startswith("lit_goal_encoder/") for p in saved)

    # an uninterrupted four-step run versus two steps + resume + two steps
    straight = dataclasses.replace(config, exp_name="straight", num_train_steps=4)
    train.main(straight)
    resumed = dataclasses.replace(config, resume=True, overwrite=False, num_train_steps=4)
    train.main(resumed)
    shape, _ = train.init_train_state(straight, jax.random.key(config.seed), mesh, resume=True)
    final_straight, final_resumed = (
        _checkpoints.restore_state(_manager(run.checkpoint_dir), shape, _Loader(batch)) for run in (straight, resumed)
    )
    assert int(final_straight.step) == int(final_resumed.step) == 4
    _equal_trees(final_resumed.ema_params, final_straight.ema_params, "ema after resume")
    for x, y in zip(jax.tree.leaves(final_resumed.opt_state), jax.tree.leaves(final_straight.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # the checkpoint moved: step 4 is not step 2
    at_step_2, at_step_4 = _leaves(restored.ema_params), _leaves(final_resumed.ema_params)
    assert not all(np.array_equal(at_step_4[p], at_step_2[p]) for p in at_step_2)
