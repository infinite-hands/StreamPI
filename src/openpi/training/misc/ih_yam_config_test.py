import dataclasses

import flax.nnx as nnx
import jax
import pytest

from openpi.models import pi0_config
import openpi.training.config as config
from openpi.training.misc import ih_yam_config

BASE = "pi05_yam_stream5_i20_bagging_right_real"


def _differences(name: str) -> set[str]:
    twin, base = config.get_config(name), config.get_config(BASE)
    return {key for key, value in dataclasses.asdict(twin.model).items() if value != dataclasses.asdict(base.model)[key]}


def test_the_right_real_twins_differ_from_their_base_only_where_named():
    assert _differences(BASE + "_xr128") == {"action_expert_variant"}
    assert _differences(BASE + "_px336") == {"image_resolution"}
    assert _differences(BASE.replace("_i20_", "_i15_")) == set()


def test_twin_loaders_and_cadence():
    base = config.get_config(BASE)
    xr128, px336, i15 = (config.get_config(name) for name in (BASE + "_xr128", BASE + "_px336", BASE.replace("_i20_", "_i15_")))
    assert (base.weight_loader.expand_lora_rank, base.weight_loader.resize_posemb) == (False, False)
    assert (xr128.weight_loader.expand_lora_rank, xr128.weight_loader.resize_posemb) == (True, False)
    assert (px336.weight_loader.expand_lora_rank, px336.weight_loader.resize_posemb) == (False, True)
    assert (base.data.base_config.hist_interval, i15.data.base_config.hist_interval) == (20, 15)
    assert i15.data.repo_id == base.data.repo_id


def test_one_arm_recipes_carry_their_gripper_dim_at_weight_one():
    for name, dim in ((BASE, 13), ("pi05_yam_stream5_i20_bagging_left_real", 6)):
        model = config.get_config(name).model
        assert (model.gripper_loss_dim, model.gripper_loss_weight) == (dim, 1.0)


def test_cam2_twin_reads_the_overhead_camera_as_well():
    base, twin = config.get_config(BASE), config.get_config(BASE + "_cam2")
    assert base.model.image_keys == ("right_wrist_0_rgb",)
    assert twin.model.image_keys == ("base_0_rgb", "right_wrist_0_rgb")
    assert twin.data.active_image_keys == frozenset({"base_0_rgb", "right_wrist_0_rgb"})
    assert set(twin.data.hist_sequence_keys) == {"observation.images.cam_high", "observation.images.cam_right_wrist"}
    assert twin.data.active_state_dims == base.data.active_state_dims


I15 = BASE.replace("_i20_", "_i15_")


def test_the_memory_twin_differs_from_its_cadence_recipe_only_by_its_memory():
    twin, base = config.get_config(I15 + "_mem"), config.get_config(I15)
    model, plain = dataclasses.asdict(twin.model), dataclasses.asdict(base.model)
    assert {key for key, value in model.items() if value != plain[key]} == {"memory_frames", "memory_tokens"}
    assert (twin.model.memory_frames, twin.model.memory_stride, twin.model.memory_tokens, twin.model.memory_dim) == (16, 4, 8, 512)
    # 16 slots 4 calls apart past the 5-frame window, 15 frames per call: the oldest frame is 34 s back.
    assert twin.model.memory_ages[0] == 8
    assert twin.model.memory_ages[-1] == 68
    assert twin.model.memory_ages[-1] * twin.data.base_config.hist_interval / 30 == 34.0
    assert twin.data.base_config.hist_horizon == twin.model.hist_horizon
    assert twin.weight_loader == base.weight_loader
    assert twin.data.repo_id == base.data.repo_id
    assert base.model.memory_frames == 0


def test_the_memory_frames_ride_the_repack_for_a_memory_model_only():
    twin = config.get_config(I15 + "_mem")
    created = twin.data.create(twin.assets_dirs, twin.model)
    assert {"memory_image", "memory_mask"} <= set(created.repack_transforms.inputs[0].structure)

    off = twin.data.create(twin.assets_dirs, dataclasses.replace(twin.model, memory_frames=0))
    assert not {"memory_image", "memory_mask"} & set(off.repack_transforms.inputs[0].structure)
    plain = config.get_config(I15)
    assert not {"memory_image", "memory_mask"} & set(plain.data.create(plain.assets_dirs, plain.model).repack_transforms.inputs[0].structure)


LEFT_I20 = "pi05_yam_stream5_i20_bagging_left_real"
LEFT_I15 = LEFT_I20.replace("_i20_", "_i15_")
LEFT_COMBO = LEFT_I15 + "_xr128_px336_dino_mem_hist"
LEFT_ARM, RIGHT_ARM = tuple(range(7)), tuple(range(7, 14))
ALL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
MEMORY_KEYS = {"memory_image", "memory_mask"}
HISTORY_KEYS = {"state_history", "state_history_mask"}


def _repack_inputs(name: str) -> set[str]:
    train = config.get_config(name)
    return set(train.data.create(train.assets_dirs, train.model).repack_transforms.inputs[0].structure)


def test_the_left_i15_recipe_is_the_left_i20_recipe_at_the_cadence_of_a_one_camera_model():
    i15, i20 = config.get_config(LEFT_I15), config.get_config(LEFT_I20)
    assert (i15.data.base_config.hist_interval, i20.data.base_config.hist_interval) == (15, 20)
    # What the 15-frame recipe adds: it skips the cameras it masks, as the right-real recipe does.
    assert i15.model.image_keys == ("left_wrist_0_rgb",)
    assert i15.data.base_config.decode_only_hist_cameras is True
    assert i15.data.hist_sequence_keys == ("observation.images.cam_left_wrist",)
    # Everything else is the i20 recipe's: arms, loss, data, loader, schedule.
    assert (i15.data.active_image_keys, i15.data.active_state_dims, i15.data.held_action_dims) == (
        frozenset({"left_wrist_0_rgb"}), LEFT_ARM, RIGHT_ARM)
    assert i15.data.repo_id == i20.data.repo_id
    assert i15.data.norm_stats_config is None
    assert i15.weight_loader == i20.weight_loader
    assert (i15.model.gripper_loss_dim, i15.model.gripper_loss_weight) == (6, 1.0)
    assert {key for key, value in dataclasses.asdict(i15.model).items() if value != dataclasses.asdict(i20.model)[key]} == {"image_keys"}
    assert (i15.batch_size, i15.fsdp_devices, i15.num_train_steps) == (i20.batch_size, i20.fsdp_devices, i20.num_train_steps)


def test_the_left_combo_twin_is_the_left_i15_recipe_plus_every_single_camera_feature():
    twin, base = config.get_config(LEFT_COMBO), config.get_config(LEFT_I15)
    model, plain = dataclasses.asdict(twin.model), dataclasses.asdict(base.model)
    assert {key for key, value in model.items() if value != plain[key]} == {
        "action_expert_variant", "image_resolution", "dino_variant", "memory_frames", "memory_tokens", "state_history",
        "state_history_dims"}

    assert twin.model.action_expert_variant == "gemma_300m_lora_r128"
    assert twin.model.paligemma_variant == base.model.paligemma_variant == "gemma_2b_lora"
    assert twin.model.image_resolution == (336, 336)
    assert (twin.model.dino_variant, twin.model.dino_trainable) == ("base", False)
    assert (twin.model.memory_frames, twin.model.memory_stride, twin.model.memory_tokens, twin.model.memory_dim) == (16, 4, 8, 512)
    assert (twin.model.state_history, twin.model.state_history_first_age) == (8, 2)
    assert twin.model.state_history_dims == LEFT_ARM
    assert twin.model.state_history_ages == tuple(range(2, 10))
    assert twin.model.uses_memory_bank
    # One camera, 15 frames per call, 5-frame window: the memory slots start past the window, as the right-real _mem twin's do.
    assert (twin.model.hist_horizon, twin.model.memory_ages[0], twin.model.memory_ages[-1]) == (5, 8, 68)
    assert twin.model.memory_ages[-1] * twin.data.base_config.hist_interval / 30 == 34.0
    assert (twin.model.gripper_loss_dim, twin.model.gripper_loss_weight) == (6, 1.0)

    data, base_data = twin.data, base.data
    assert data.base_config.hist_interval == 15
    assert data.base_config.hist_horizon == twin.model.hist_horizon
    assert data.base_config.decode_only_hist_cameras is True
    assert twin.model.image_keys == ("left_wrist_0_rgb",)
    assert data.active_image_keys == frozenset({"left_wrist_0_rgb"})
    assert data.hist_sequence_keys == ("observation.images.cam_left_wrist",)
    assert (data.active_state_dims, data.held_action_dims) == (LEFT_ARM, RIGHT_ARM)
    assert data.repo_id == base_data.repo_id
    # The recipe's own norm stats, not a fresh set under the twin's name.
    assert data.norm_stats_config == LEFT_I15
    assert (twin.batch_size, twin.fsdp_devices, twin.num_train_steps) == (base.batch_size, base.fsdp_devices, base.num_train_steps)


def test_the_left_combo_loader_grows_the_checkpoint_in_every_way_the_model_outgrew_it():
    twin, base = config.get_config(LEFT_COMBO).weight_loader, config.get_config(LEFT_I15).weight_loader
    assert (twin.expand_lora_rank, twin.resize_posemb, twin.dino_variant) == (True, True, "base")
    assert (base.expand_lora_rank, base.resize_posemb, base.dino_variant) == (False, False, None)
    assert twin.params_path == base.params_path
    assert twin.dino_weights is None


def test_the_left_combo_freezes_the_dino_backbone_and_the_language_model_but_trains_every_new_weight():
    model = config.get_config(LEFT_COMBO).model
    abstract = nnx.eval_shape(lambda: model.create(jax.random.key(0)))
    frozen = nnx.state(abstract, nnx.All(nnx.Param, model.get_freeze_filter()))
    trainable = nnx.state(abstract, nnx.All(nnx.Param, nnx.Not(model.get_freeze_filter())))

    def names(state) -> set[str]:
        return {"/".join(str(getattr(part, "key", getattr(part, "name", part))) for part in path).removesuffix("/value")
                for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]}

    frozen_names, trainable_names = names(frozen), names(trainable)
    assert any(name.startswith("dino/") for name in frozen_names)
    assert not any(name.startswith("dino/") for name in trainable_names)
    assert "dino_proj/kernel" in trainable_names
    assert {"memory_query", "memory_state_in/kernel", "PaliGemma/llm/layers/attn/memory_gate"} <= trainable_names

    # Against the i15 recipe, the only new trainable weights are the projection and the memory modules; the rank growth is in shapes.
    base = config.get_config(LEFT_I15).model
    base_state = nnx.state(nnx.eval_shape(lambda: base.create(jax.random.key(0))),
                           nnx.All(nnx.Param, nnx.Not(base.get_freeze_filter())))
    added = trainable_names - names(base_state)
    assert added
    assert all(name.startswith("dino_proj/") or "memory" in name for name in added)
    assert names(base_state) <= trainable_names


def test_the_left_combo_repack_carries_the_memory_frames_and_the_state_history():
    assert {*MEMORY_KEYS, *HISTORY_KEYS} <= _repack_inputs(LEFT_COMBO)
    assert not {*MEMORY_KEYS, *HISTORY_KEYS} & _repack_inputs(LEFT_I15)
    assert not {*MEMORY_KEYS, *HISTORY_KEYS} & _repack_inputs(LEFT_I20)


def test_image_memory_refuses_a_second_camera_so_the_combo_cannot_gain_the_overhead_camera():
    model = config.get_config(LEFT_COMBO).model
    two = ("base_0_rgb", "left_wrist_0_rgb")
    with pytest.raises(ValueError, match="memory_frames is wired for one camera"):
        dataclasses.replace(model, image_keys=two)
    with pytest.raises(ValueError, match="memory_frames is wired for one camera"):
        dataclasses.replace(model, image_keys=ALL_CAMERAS)
    # The same two cameras are fine without the image memory (the state history is not tied to a camera count).
    assert dataclasses.replace(model, image_keys=two, memory_frames=0).image_keys == two
    # The recipe factory builds its model by the same config, so a combo twin of a two-camera recipe cannot be registered either.
    with pytest.raises(ValueError, match="memory_frames is wired for one camera"):
        pi0_config.Pi0Config(pi05=True, image_keys=two, memory_frames=16)


def test_the_right_real_cadence_twins_are_unchanged_by_the_left_recipes():
    i15 = config.get_config(I15)
    assert (i15.model.image_keys, i15.data.base_config.decode_only_hist_cameras) == (("right_wrist_0_rgb",), True)
    assert i15.data.hist_sequence_keys == ("observation.images.cam_right_wrist",)
    assert (i15.data.active_state_dims, i15.data.held_action_dims) == (RIGHT_ARM, LEFT_ARM)
    assert i15.data.base_config.hist_interval == 15
    assert i15.data.norm_stats_config is None
    assert (i15.model.dino_variant, i15.model.memory_frames, i15.model.state_history) == (None, 0, 0)
    assert i15.weight_loader == config.get_config(BASE).weight_loader

    mem, hist = config.get_config(I15 + "_mem"), config.get_config(I15 + "_hist")
    assert (mem.model.image_keys, mem.data.base_config.decode_only_hist_cameras, mem.model.state_history) == (
        ("right_wrist_0_rgb",), True, 0)
    assert (hist.model.image_keys, hist.data.base_config.decode_only_hist_cameras, hist.model.memory_frames) == (
        ("right_wrist_0_rgb",), True, 0)
    assert hist.model.state_history_dims == RIGHT_ARM
    assert (mem.data.norm_stats_config, hist.data.norm_stats_config) == (I15, I15)
    # The right-real recipe has no combo twin, and no other recipe gains one.
    assert [c.name for c in ih_yam_config.get_ih_yam_configs() if c.name.endswith(ih_yam_config.COMBO_SUFFIX)] == [LEFT_COMBO]


def test_the_left_i20_recipe_keeps_all_three_cameras():
    left = config.get_config(LEFT_I20)
    assert left.model.image_keys == ALL_CAMERAS
    assert left.data.base_config.decode_only_hist_cameras is False
    assert left.data.base_config.hist_interval == 20
    assert left.data.active_image_keys == frozenset({"left_wrist_0_rgb"})
    assert set(left.data.hist_sequence_keys) == {
        "observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"}
    assert (left.data.active_state_dims, left.data.held_action_dims) == (LEFT_ARM, RIGHT_ARM)
    assert left.data.norm_stats_config is None
    assert (left.model.dino_variant, left.model.memory_frames, left.model.state_history) == (None, 0, 0)
    assert left.model.image_resolution == (224, 224)
    assert left.model.action_expert_variant == "gemma_300m_lora"
    assert not left.weight_loader.expand_lora_rank
