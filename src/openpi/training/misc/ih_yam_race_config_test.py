"""The `_race` twins: the base recipe's model and data with RACE on, its norm stats, and its targets on /misc."""

import dataclasses
import pathlib

import pytest

from openpi import transforms as _transforms
from openpi.training import config as _config
from openpi.training.misc import ih_yam_config


@pytest.mark.parametrize("base_name", ih_yam_config.RACE_TWIN_BASES)
def test_race_twin_is_its_base_recipe_with_race_on(base_name, monkeypatch):
    base, twin = _config.get_config(base_name), _config.get_config(base_name + "_race")
    assert twin.model.race and not base.model.race
    # Camera skip is numerically the same model; everything else about the model is the base recipe's.
    assert dataclasses.replace(twin.model, race=False, image_keys=base.model.image_keys) == base.model
    assert set(twin.model.image_keys) == set(base.data.active_image_keys)
    assert twin.data.repo_id == base.data.repo_id
    assert twin.data.active_image_keys == base.data.active_image_keys
    assert twin.data.active_state_dims == base.data.active_state_dims
    assert twin.data.held_action_dims == base.data.held_action_dims
    assert twin.data.base_config.hist_interval == base.data.base_config.hist_interval
    assert (twin.batch_size, twin.fsdp_devices, twin.ema_decay) == (base.batch_size, base.fsdp_devices, base.ema_decay)

    read_from = []
    monkeypatch.setattr(type(twin.data), "_load_norm_stats",
                        lambda self, assets_dir, asset_id: read_from.append(pathlib.Path(assets_dir) / asset_id))
    created = twin.data.create(twin.assets_dirs, twin.model)
    base.data.create(base.assets_dirs, base.model)
    assert read_from[0] == read_from[1], "the twin normalizes with its base recipe's norm stats"
    assert created.race_targets_dir == f"/misc/race-targets/{base.data.repo_id}"
    other = dataclasses.replace(twin.data, repo_id="local/other").create(twin.assets_dirs, twin.model)
    assert other.race_targets_dir == "/misc/race-targets/local/other", "follows a --data.repo-id override"
    repack = created.repack_transforms.inputs[0].structure
    assert all(repack[key] == key for key in _transforms.RACE_TARGET_KEYS)
    assert base.data.create(base.assets_dirs, base.model).race_targets_dir is None
