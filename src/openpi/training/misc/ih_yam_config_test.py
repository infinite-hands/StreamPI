import dataclasses

import openpi.training.config as config

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
