"""The sort-piles right-arm recipes: the overhead and right-wrist cameras, the left arm held, and a long-chunk shape."""

import pytest

import openpi.models.pi0_config as pi0_config
from openpi.training import config as _config
from openpi.training.misc import ih_yam_config

CHUNK_30 = "pi05_yam_stream5_i20_sort_right_real"
CHUNK_50 = "pi05_yam_stream5_i40_h50_sort_right_real"
WIDE_TWIN = CHUNK_50 + f"_vlash{ih_yam_config.SORT_WIDE_VLASH_OFFSET}mpc"


@pytest.mark.parametrize("name", [CHUNK_30, CHUNK_50, WIDE_TWIN, CHUNK_30 + "_vlash12mpc"])
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


def test_the_long_chunk_recipe_keeps_pi05s_pretrained_horizon_and_a_margin():
    config = _config.get_config(CHUNK_50)
    assert config.model.action_horizon == pi0_config.Pi0Config().action_horizon == ih_yam_config.SORT_WIDE_ACTION_HORIZON
    played = config.data.base_config.hist_interval
    assert played == ih_yam_config.SORT_WIDE_HIST_INTERVAL
    assert played < config.model.action_horizon, "no rows left past the played window for the lead"


def test_the_wide_twin_leads_a_second_within_the_played_window():
    config = _config.get_config(WIDE_TWIN)
    base = config.data.base_config
    assert (base.vlash_max_offset, base.vlash_branches) == (30, 31)
    assert base.vlash_max_offset <= base.hist_interval - 1, "a lead past the played window cannot be asked for"
    assert base.vlash_state_source == "state" and base.vlash_cond_now and config.model.state_cond
    assert config.num_train_steps == 10_000 and config.lr_schedule.decay_steps == 10_000


def test_the_wide_twin_exists_only_for_the_long_chunk_recipe():
    names = set(_config._CONFIGS_DICT)
    assert WIDE_TWIN in names
    assert not any(name.endswith(f"_vlash{ih_yam_config.SORT_WIDE_VLASH_OFFSET}mpc") for name in names if name != WIDE_TWIN)


def test_bagging_recipes_keep_their_horizon():
    for name in ("pi05_yam_stream5_i20_bagging_right_real", "pi05_yam_stream5_i20_bagging_right_real_vlash12mpc"):
        assert _config.get_config(name).model.action_horizon == ih_yam_config.ACTION_HORIZON == 30
