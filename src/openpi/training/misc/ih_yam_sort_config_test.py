"""The sort-piles right-arm recipes: the overhead and right-wrist cameras, the left arm held, and a long-chunk shape."""

import pytest

import openpi.models.pi0_config as pi0_config
from openpi.training import config as _config
from openpi.training.misc import ih_yam_config

CHUNK_30 = "pi05_yam_stream5_i20_sort_right_real"
LONG_CHUNKS = {interval: f"pi05_yam_stream5_i{interval}_h50_sort_right_real" for interval in (40, 50)}
WIDE_TWINS = {interval: name + f"_vlash{ih_yam_config.SORT_WIDE_VLASH_OFFSET_BY_INTERVAL[interval]}mpc"
              for interval, name in LONG_CHUNKS.items()}
TIME_COMPRESSED = "pi05_yam_stream5_i50_h50_sort_right_real_timecompressed2x"


@pytest.mark.parametrize("name", [CHUNK_30, *LONG_CHUNKS.values(), *WIDE_TWINS.values(), CHUNK_30 + "_vlash12mpc"])
def test_sort_recipes_read_two_cameras_and_hold_the_left_arm(name):
    config = _config.get_config(name)
    assert config.data.repo_id == ih_yam_config.SORT_RIGHT_REAL_REPO_ID
    assert config.data.default_prompt == ih_yam_config.SORT_PROMPT
    assert config.data.active_image_keys == frozenset({"base_0_rgb", "right_wrist_0_rgb"})
    assert set(config.model.image_keys) == set(config.data.active_image_keys), "a masked camera is not encoded"
    assert config.data.active_state_dims == ih_yam_config.RIGHT_ARM_DIMS
    assert config.data.held_action_dims == ih_yam_config.LEFT_ARM_DIMS


def test_the_30_row_recipe_is_the_bagging_shape():
    config = _config.get_config(CHUNK_30)
    assert (config.model.action_horizon, config.data.base_config.hist_interval) == (30, 20)


@pytest.mark.parametrize("interval", LONG_CHUNKS)
def test_the_long_chunk_recipes_keep_pi05s_pretrained_horizon_and_play_within_it(interval):
    config = _config.get_config(LONG_CHUNKS[interval])
    assert config.model.action_horizon == pi0_config.Pi0Config().action_horizon == ih_yam_config.SORT_WIDE_ACTION_HORIZON
    assert config.data.base_config.hist_interval == interval <= config.model.action_horizon


@pytest.mark.parametrize("interval", WIDE_TWINS)
def test_each_wide_twin_leads_as_far_as_its_window_allows(interval):
    config = _config.get_config(WIDE_TWINS[interval])
    base = config.data.base_config
    offset = ih_yam_config.SORT_WIDE_VLASH_OFFSET_BY_INTERVAL[interval]
    assert (base.vlash_max_offset, base.vlash_branches) == (offset, offset + 1)
    assert base.vlash_max_offset <= base.hist_interval - 1, "a lead past the played window cannot be asked for"
    assert base.vlash_state_source == "state" and base.vlash_cond_now and config.model.state_cond
    assert config.num_train_steps == 10_000 and config.lr_schedule.decay_steps == 10_000


def test_wide_twins_exist_only_for_the_long_chunk_recipes():
    wide = {name for name in _config._CONFIGS_DICT if name.startswith("pi05_yam_stream5_") and "mpc" in name
            and name.rsplit("_vlash", 1)[-1] not in ("8mpc", "12mpc")}
    assert wide == {*WIDE_TWINS.values(), TIME_COMPRESSED + "_vlash49mpc"}


def test_the_time_compressed_recipe_is_the_whole_chunk_recipe_on_its_own_labelled_dataset():
    plain, compressed = _config.get_config(LONG_CHUNKS[50]), _config.get_config(TIME_COMPRESSED)
    assert "timecompressed2x" in compressed.data.repo_id and compressed.data.repo_id != plain.data.repo_id
    assert compressed.data.repo_id == ih_yam_config.SORT_RIGHT_REAL_2X_REPO_ID
    assert compressed.model == plain.model and compressed.data.base_config == plain.data.base_config
    assert (compressed.data.active_image_keys, compressed.data.default_prompt) == (plain.data.active_image_keys, plain.data.default_prompt)
    twin = _config.get_config(TIME_COMPRESSED + "_vlash49mpc")
    assert (twin.data.base_config.vlash_max_offset, twin.data.base_config.vlash_branches) == (49, 50)


def test_bagging_recipes_keep_their_horizon():
    for name in ("pi05_yam_stream5_i20_bagging_right_real", "pi05_yam_stream5_i20_bagging_right_real_vlash12mpc"):
        assert _config.get_config(name).model.action_horizon == ih_yam_config.ACTION_HORIZON == 30
