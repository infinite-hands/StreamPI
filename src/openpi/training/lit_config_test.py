"""The four LIT rows on the left-real recipe (ih_yam_config): that they build, what differs between them and what
must not, the driven-arm goal dims, the shared norm statistics, and the freeze-filter counts at real dimensions
(`nnx.eval_shape` only: nothing of the 3B model is allocated, and the image tower is the real SigLIP).
"""

import dataclasses
import pathlib
import re

import flax.nnx as nnx
import jax
import numpy as np
import pytest
import tyro

from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import weight_loaders
from openpi.training.misc import ih_yam_config as _yam

BASE = _yam.LIT_BASE_CONFIG
SUFFIXES = ("litctl", "lit1", "lit2", "litlite")
NAMES = {suffix: f"{BASE}_{suffix}" for suffix in SUFFIXES}
STAGE = {"litctl": "off", "lit1": "stage1", "lit2": "stage2", "litlite": "stage2"}
STATE_DIMS = 14


def _row(suffix) -> _config.TrainConfig:
    return _config.get_config(NAMES[suffix])


def _path(path) -> str:
    return "/".join(str(p) for p in path)


def test_the_four_rows_exist_under_their_exact_names_and_build():
    assert NAMES == {
        "litctl": "pi05_yam_stream5_i20_bagging_left_real_litctl",
        "lit1": "pi05_yam_stream5_i20_bagging_left_real_lit1",
        "lit2": "pi05_yam_stream5_i20_bagging_left_real_lit2",
        "litlite": "pi05_yam_stream5_i20_bagging_left_real_litlite",
    }
    registered = [c.name for c in _config._CONFIGS]  # noqa: SLF001
    assert len(registered) == len(set(registered))
    for suffix in SUFFIXES:
        row = _row(suffix)
        assert row.name == NAMES[suffix]
        assert row.model.lit == STAGE[suffix]
        # the data config builds on the model (the factory reads the model's lit)
        data = row.data.create(row.assets_dirs, row.model)
        assert data.repo_id == _yam.BAGGING_LEFT_REAL_REPO_ID
    assert BASE in registered, "the recipe the rows are built on is still registered"


def test_model_dims_of_every_row():
    for suffix in SUFFIXES:
        model = _row(suffix).model
        assert model.pi05 and (model.action_horizon, model.hist_horizon) == (30, 5)
        if suffix == "litctl":
            assert model.lit_goal_dims == ()
            continue
        assert model.lit_groups == 6, "6 parameter groups of 3 layers each over the 18 layers of Gemma 2B"
        assert model.lit_goal_dims == tuple(range(7))
        assert (model.lit_num_latents, model.lit_dim, model.lit_heads, model.lit_kv_dim) == (100, 768, 8, 256)
        assert (model.lit_pose_tokens, model.lit_goal_tokens, model.lit_pose_weight) == (8, 8, 0.3)
        assert model.lit_mask_image and model.lit_mask_language


def test_the_goal_dims_are_the_driven_arm_and_the_complement_of_the_held_dims():
    """YAM state layout [L j0..5, L grip, R j0..5, R grip]: the left arm is dims 0-6 (gripper 6), the right 7-13
    (gripper 13). The recipe hides the right arm from the tokens (active_state_dims) and trains it to hold still."""
    row = _row("lit2")
    data = row.data
    held, active = data.held_action_dims, data.active_state_dims
    assert held == tuple(range(7, 14)) and active == tuple(range(7))
    complement = tuple(d for d in range(STATE_DIMS) if d not in held)
    for suffix in ("lit1", "lit2", "litlite"):
        dims = _row(suffix).model.lit_goal_dims
        assert dims == complement == active == _yam.LEFT_ARM_DIMS == _yam.LIT_GOAL_DIMS
        assert 6 in dims and 13 not in dims
        assert max(dims) < STATE_DIMS <= _row(suffix).model.action_dim, "never the zero padding of the 32-wide state"


def test_the_rows_share_the_recipes_data_prompt_cameras_and_cadence():
    base = _config.get_config(BASE)
    fields = ("repo_id", "default_prompt", "active_image_keys", "active_state_dims", "held_action_dims")
    for suffix in SUFFIXES:
        row = _row(suffix)
        for field in fields:
            assert getattr(row.data, field) == getattr(base.data, field), (suffix, field)
        assert row.data.base_config == base.data.base_config, "history horizon, cadence and jitter"
        assert (row.model.action_horizon, row.model.hist_horizon) == (
            base.model.action_horizon,
            base.model.hist_horizon,
        )
        assert row.model.dtype == base.model.dtype and row.model.max_token_len == base.model.max_token_len
        # the hyperparameters the rows' comments call the BASE recipe's (not LIT's own release)
        assert (row.num_train_steps, row.lr_schedule, row.optimizer) == (
            base.num_train_steps,
            base.lr_schedule,
            base.optimizer,
        )


def test_the_training_knobs_of_every_row():
    ctl, one, two, lite = (_row(s) for s in SUFFIXES)
    for row in (ctl, one, two):  # full fine-tune rows: like the *_full configs
        assert (row.batch_size, row.fsdp_devices) == (32, 4)
        assert (row.model.paligemma_variant, row.model.action_expert_variant) == ("gemma_2b", "gemma_300m")
    assert (ctl.ema_decay, two.ema_decay) == (0.99, 0.99)
    assert one.ema_decay is None, "stage 1 freezes the backbone: no EMA"
    assert lite.ema_decay is None
    assert (lite.batch_size, lite.fsdp_devices) == (16, 1)
    assert (lite.model.paligemma_variant, lite.model.action_expert_variant) == ("gemma_2b_lora", "gemma_300m_lora")
    # PaliGemma init with a random expert for the control and stage 1; stage 2 from a stage-1 checkpoint given at
    # launch; the LoRA row warm-starts from pi05_base with the opt-in regex
    assert ctl.weight_loader == weight_loaders.PaliGemmaWeightLoader() == one.weight_loader
    assert two.weight_loader == weight_loaders.LitStage1WeightLoader(params_path="")
    assert lite.weight_loader == weight_loaders.CheckpointWeightLoader(
        _yam.PI05_BASE_PARAMS, missing_regex=".*lora.*|lit_.*"
    )
    for suffix in SUFFIXES:
        row = _row(suffix)
        assert row.freeze_filter == row.model.get_freeze_filter()
    assert one.freeze_filter != nnx.Nothing and ctl.freeze_filter == nnx.Nothing == two.freeze_filter


def test_stage2_params_path_is_supplied_at_launch_on_the_command_line():
    name = NAMES["lit2"]
    parsed = tyro.extras.overridable_config_cli(
        {name: (name, _row("lit2"))},
        args=[name, "--exp-name", "x", "--weight-loader.params-path", "/checkpoints/lit1/x/9999/params"],
    )
    assert parsed.weight_loader == weight_loaders.LitStage1WeightLoader("/checkpoints/lit1/x/9999/params")
    with pytest.raises(ValueError, match="params-path"):
        _row("lit2").weight_loader.load({})


# ---- the shared norm statistics ----


def _stats():
    lo, hi = np.full(STATE_DIMS, -2.0), np.full(STATE_DIMS, 40.0)
    state = _normalize.NormStats(mean=np.zeros(STATE_DIMS), std=np.ones(STATE_DIMS), q01=lo, q99=hi)
    return {"state": state, "actions": dataclasses.replace(state, q01=lo - 1, q99=hi + 1)}


def test_the_four_rows_read_one_set_of_norm_statistics(tmp_path):
    """All four rows read the PRODUCTION base recipe's stats: the dir the launcher's stats pass writes for the base
    config name, so no row depends on another having run first (a _lit* name's own pass writes elsewhere)."""
    production = f"/checkpoints/assets/{BASE}"
    pinned = _config.AssetsConfig(assets_dir=production, asset_id=_yam.BAGGING_LEFT_REAL_REPO_ID)
    for suffix in SUFFIXES:
        assert _row(suffix).data.assets == pinned, suffix
    assert _yam.LIT_NORM_ASSETS_DIR == production
    # the pinned file is the one the base recipe itself reads on the volume: assets_dirs (keyed by config name) / repo id
    base = _config.get_config(BASE)
    on_volume = dataclasses.replace(base, assets_base_dir="/checkpoints/assets")
    base_data = on_volume.data.create(on_volume.assets_dirs, on_volume.model)
    base_reads = pathlib.Path(on_volume.data.assets.assets_dir or on_volume.assets_dirs) / base_data.asset_id
    assert base_reads == pathlib.Path(pinned.assets_dir) / pinned.asset_id
    # no _lit* row's own assets dir (where a stats pass for that name would write) is the pinned one
    assert all(f"/checkpoints/assets/{NAMES[suffix]}" != pinned.assets_dir for suffix in SUFFIXES)

    # and with the stats present the four data configs load the very same ones, for the model of each row
    stats = _stats()
    _normalize.save(tmp_path / _yam.BAGGING_LEFT_REAL_REPO_ID, stats)
    loaded = []
    for suffix in SUFFIXES:
        row = _row(suffix)
        data = dataclasses.replace(row.data, assets=_config.AssetsConfig(
            assets_dir=str(tmp_path), asset_id=_yam.BAGGING_LEFT_REAL_REPO_ID))
        loaded.append(data.create(row.assets_dirs, row.model))
    for data in loaded:
        assert sorted(data.norm_stats) == ["actions", "state"], "no lit_goal entry: the saved assets stay the control's"
        for key in ("state", "actions"):
            for field in ("mean", "std", "q01", "q99"):
                np.testing.assert_array_equal(
                    getattr(data.norm_stats[key], field), getattr(loaded[0].norm_stats[key], field)
                )
        assert data.asset_id == _yam.BAGGING_LEFT_REAL_REPO_ID and data.use_quantile_norm


def test_lit_rows_carry_the_goal_and_the_control_does_not():
    for suffix in SUFFIXES:
        row = _row(suffix)
        data = row.data.create(row.assets_dirs, row.model)
        structure = data.repack_transforms.inputs[0].structure
        if suffix == "litctl":
            assert "lit_goal" not in structure and data.norm_aliases == {}
        else:
            assert structure["lit_goal"] == "lit_goal" and structure["lit_goal_mask"] == "lit_goal_mask"
            assert data.norm_aliases == {"lit_goal": "state"}


def test_stage1_decodes_every_camera_even_though_it_embeds_none():
    """compute_loss's preprocess_observation raises on an images dict that lacks a camera, so the loader must keep
    decoding all three for stage 1 (the recipe masks two of them to zero images, it does not drop them)."""
    row = _row("lit1")
    data = row.data.create(row.assets_dirs, row.model)
    assert tuple(data.hist_sequence_keys) == (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    assert getattr(row.model, "image_keys", ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert data.data_transforms.inputs[0].active_image_keys == frozenset({"left_wrist_0_rgb"})


def test_a_lit_row_with_a_wrong_freeze_filter_does_not_build():
    for suffix in ("lit1", "lit2", "litlite"):
        with pytest.raises(ValueError, match="freeze_filter"):
            dataclasses.replace(_row(suffix), freeze_filter=nnx.Nothing if suffix == "lit1" else
                                _row("lit1").freeze_filter)


# ---- the freeze-filter counts at real dimensions ----

# Measured with `nnx.eval_shape` at the rows' real dimensions (Gemma 2B + 300M expert, SigLIP So400m, action_dim 32,
# 100 latents, 6 groups, 7 goal dims): (trainable leaves, trainable params, frozen leaves, frozen params).
COUNTS = {
    "litctl": (51, 3_353_433_872, 0, 0),
    "lit1": (27, 433_910_304, 32, 2_923_335_408),
    "lit2": (114, 3_510_507_799, 0, 0),
    "litlite": (114, 624_030_999, 20, 2_936_464_384),
}
AGGREGATOR_PARAMS = 153_659_904  # 55 leaves
POSE_DECODER_PARAMS = 3_414_023  # 8 leaves
GOAL_ENCODER_PARAMS = 3_811_840  # 8 leaves


def _count(flat, paths):
    return int(sum(int(np.prod(flat[p].value.shape)) for p in paths))


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_freeze_filter_counts_at_real_dimensions(suffix, capsys):
    row = _row(suffix)
    model = nnx.eval_shape(lambda: row.model.create(jax.random.key(0)))
    state = nnx.state(model)
    flat = state.flat_state()
    train = set(state.filter(row.trainable_filter).flat_state())
    frozen = set(flat) - train
    measured = (len(train), _count(flat, train), len(frozen), _count(flat, frozen))
    with capsys.disabled():
        print(
            f"\n{suffix}: trainable {measured[0]} leaves / {measured[1]:,} params, "
            f"frozen {measured[2]} leaves / {measured[3]:,}"
        )
    assert measured == COUNTS[suffix]
    lit = {p for p in flat if _path(p).startswith("lit_")}
    assert lit <= train or suffix == "litctl", "every lit_* leaf trains"
    parts = {
        name: {p for p in flat if _path(p).startswith(name)}
        for name in ("lit_aggregator", "lit_pose_decoder", "lit_goal_encoder")
    }
    if suffix in ("lit2", "litlite"):
        assert _count(flat, parts["lit_aggregator"]) == AGGREGATOR_PARAMS
        assert _count(flat, parts["lit_pose_decoder"]) == POSE_DECODER_PARAMS
        assert not parts["lit_goal_encoder"]
    if suffix == "lit1":
        assert _count(flat, parts["lit_goal_encoder"]) == GOAL_ENCODER_PARAMS
        assert not parts["lit_aggregator"] and not parts["lit_pose_decoder"]
        # the expert and the goal encoder train; SigLIP and the 2B backbone are frozen
        assert all(
            _path(p).startswith(("lit_goal_encoder/", "action_in_proj", "action_out_proj", "time_mlp_"))
            or ("PaliGemma/llm/" in _path(p) and "_1" in _path(p))
            for p in train
        )
        assert any(_path(p).startswith("PaliGemma/img/") for p in frozen)
    if suffix == "litlite":
        assert all("lora" not in _path(p) for p in frozen)
        assert any("lora" in _path(p) for p in train)


def test_the_lit_regex_in_the_rows_matches_no_stock_leaf():
    stock = _config.get_config(BASE)
    model = nnx.eval_shape(lambda: stock.model.create(jax.random.key(0)))
    pattern = re.compile(weight_loaders.LIT_MISSING_REGEX)
    stray = [p for p in (_path(q) for q in nnx.state(model).flat_state()) if pattern.fullmatch(p) and "lora" not in p]
    assert not stray
