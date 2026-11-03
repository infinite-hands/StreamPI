import dataclasses
import types

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def _meta(*lengths, fps=30):
    return types.SimpleNamespace(fps=fps, episodes={i: {"length": n} for i, n in enumerate(lengths)})


def test_the_timestamp_tolerance_is_lerobots_default_for_ordinary_episodes():
    # A 2 min and a 7.5 min episode: float32 resolves 1/fps far inside 1e-4 s, so the check stays exactly as it was.
    assert _data_loader.timestamp_tolerance_s(_meta(3600, 13_500)) == _data_loader.LEROBOT_TIMESTAMP_TOLERANCE_S


def test_a_long_episodes_float32_timestamps_pass_lerobots_own_sync_check_only_under_the_widened_tolerance():
    from lerobot.common.datasets.utils import check_timestamps_sync

    frames, fps = 41_106, 30     # the 23 min sort_pile_both episode_0002, whose float32 timestamps are off by 1.1e-4 s
    timestamps = (np.arange(frames) / fps).astype(np.float32)
    episodes = {"to": np.array([frames])}
    indices = np.zeros(frames, dtype=np.int64)
    assert not check_timestamps_sync(timestamps, indices, episodes, fps, _data_loader.LEROBOT_TIMESTAMP_TOLERANCE_S, raise_value_error=False)
    tolerance = _data_loader.timestamp_tolerance_s(_meta(frames))
    assert tolerance > _data_loader.LEROBOT_TIMESTAMP_TOLERANCE_S
    assert check_timestamps_sync(timestamps, indices, episodes, fps, tolerance)
    assert tolerance < 1 / fps / 100, "still a small fraction of a frame"
