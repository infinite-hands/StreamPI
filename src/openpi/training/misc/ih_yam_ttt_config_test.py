"""The `_ttt` twins: the base recipe's model and data with TTT layers on, and its norm stats."""

import dataclasses
import pathlib

import pytest

from openpi.training import config as _config
from openpi.training.misc import ih_yam_config


@pytest.mark.parametrize("base_name", ih_yam_config.TTT_TWIN_BASES)
def test_ttt_twin_is_its_base_recipe_with_ttt_layers(base_name, monkeypatch):
    base, twin = _config.get_config(base_name), _config.get_config(base_name + "_ttt")
    assert twin.model.ttt_width == ih_yam_config.TTT_WIDTH and not base.model.ttt_width
    assert dataclasses.replace(twin.model, ttt_width=0) == base.model
    assert dataclasses.replace(twin.data, norm_stats_config=None) == base.data
    assert (twin.batch_size, twin.fsdp_devices, twin.ema_decay) == (base.batch_size, base.fsdp_devices, base.ema_decay)
    assert twin.lr_schedule == base.lr_schedule and twin.num_train_steps == base.num_train_steps

    read_from = []
    monkeypatch.setattr(type(twin.data), "_load_norm_stats",
                        lambda self, assets_dir, asset_id: read_from.append(pathlib.Path(assets_dir) / asset_id))
    twin.data.create(twin.assets_dirs, twin.model)
    base.data.create(base.assets_dirs, base.model)
    assert read_from[0] == read_from[1], "the twin normalizes with its base recipe's norm stats"


def test_only_the_named_recipes_get_a_ttt_twin():
    names = {config.name for config in ih_yam_config.get_ih_yam_configs()}
    assert {name for name in names if name.endswith("_ttt")} == {base + "_ttt" for base in ih_yam_config.TTT_TWIN_BASES}
