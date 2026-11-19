"""Per-episode sample weights (DataConfig.sample_weights_path): the sidecar's validation, its expansion to one
weight per item, the dataset wiring, and the --data.sample-weights-path override. Toy sidecars and a fake
LeRobot dataset, no real dataset."""

import json
import pathlib
import sys

import datasets
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

REPO_ID = "local/toy_weights"
EPISODE_LENGTHS = (5, 40, 7)
EPISODE_WEIGHTS = [1.0, 3.0, 5.0]


def _write_sidecar(path: pathlib.Path, **overrides) -> pathlib.Path:
    sidecar = {"version": 1, "repo_id": REPO_ID, "n_episodes": len(EPISODE_LENGTHS), "weights": EPISODE_WEIGHTS,
               "takes": [{"path": "ignored"}], **overrides}
    path.write_text(json.dumps(sidecar))
    return path


def test_check_episode_weights_accepts_a_fitting_sidecar(tmp_path):
    path = _write_sidecar(tmp_path / "weights.json")
    weights = _data_loader.check_episode_weights(str(path), REPO_ID, len(EPISODE_LENGTHS))
    assert weights.dtype == np.float64
    np.testing.assert_array_equal(weights, EPISODE_WEIGHTS)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version": 2}, "version"),
        ({"repo_id": "local/other"}, "written for"),
        ({"n_episodes": 4}, "is for 4 episodes"),
        ({"weights": [1.0, 2.0]}, "holds 2 weights"),
        ({"weights": [1.0, 0.0, 2.0]}, "not finite and > 0"),
        ({"weights": [1.0, -1.0, 2.0]}, "not finite and > 0"),
        ({"weights": [1.0, float("nan"), 2.0]}, "not finite and > 0"),
        ({"weights": [1.0, float("inf"), 2.0]}, "not finite and > 0"),
        ({"weights": [1.0, "2", 2.0]}, "not a list of numbers"),
        ({"weights": [1.0, True, 2.0]}, "not a list of numbers"),
        ({"weights": {"0": 1.0}}, "not a list of numbers"),
    ],
)
def test_check_episode_weights_refuses_a_sidecar_that_does_not_fit(tmp_path, overrides, message):
    path = _write_sidecar(tmp_path / "weights.json", **overrides)
    with pytest.raises(ValueError, match=message):
        _data_loader.check_episode_weights(str(path), REPO_ID, len(EPISODE_LENGTHS))


def test_check_episode_weights_refuses_a_missing_sidecar(tmp_path):
    with pytest.raises(FileNotFoundError, match="do not exist"):
        _data_loader.check_episode_weights(str(tmp_path / "absent.json"), REPO_ID, 3)


def test_every_item_takes_its_episodes_weight():
    episodes = np.array([2, 0, 1, 1, 2])
    weights = _data_loader.episode_sample_weights(np.array(EPISODE_WEIGHTS), episodes, "toy")
    np.testing.assert_array_equal(weights, [5.0, 1.0, 3.0, 3.0, 5.0])
    with pytest.raises(ValueError, match="cover"):
        _data_loader.episode_sample_weights(np.array(EPISODE_WEIGHTS), np.array([0, 3]), "toy")


class _FakeMeta:
    fps = 30
    tasks = {0: "toy"}
    episodes = {e: {"length": n} for e, n in enumerate(EPISODE_LENGTHS)}
    total_episodes = len(EPISODE_LENGTHS)
    total_frames = sum(EPISODE_LENGTHS)

    def __init__(self, repo_id):
        assert repo_id == REPO_ID


class _FakeLeRobot:
    def __init__(self, repo_id, **kwargs):
        episode_index = np.concatenate([np.full(n, e) for e, n in enumerate(EPISODE_LENGTHS)])
        table = datasets.Dataset.from_dict({"index": np.arange(len(episode_index)), "episode_index": episode_index})
        table.set_transform(lambda batch: {key: [torch.as_tensor(v) for v in values] for key, values in batch.items()})
        self.hf_dataset = table

    def __len__(self):
        return sum(EPISODE_LENGTHS)


@pytest.fixture
def fake_lerobot(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMeta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobot)


def _model_config(**kwargs):
    return pi0_config.Pi0Config(action_dim=24, action_horizon=10, max_token_len=48, **kwargs)


def test_dataset_carries_episode_weights_only_when_asked(tmp_path, fake_lerobot):
    path = _write_sidecar(tmp_path / "weights.json")
    weighted = _data_loader.create_torch_dataset(
        _config.DataConfig(repo_id=REPO_ID, sample_weights_path=str(path)), 10, _model_config())
    expected = np.repeat(EPISODE_WEIGHTS, EPISODE_LENGTHS)
    np.testing.assert_array_equal(weighted.sample_weights, expected)

    plain = _data_loader.create_torch_dataset(_config.DataConfig(repo_id=REPO_ID), 10, _model_config())
    assert plain.sample_weights is None


def test_dataset_refuses_weights_that_do_not_fit(tmp_path, fake_lerobot):
    path = _write_sidecar(tmp_path / "weights.json", n_episodes=2, weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="do not fit"):
        _data_loader.create_torch_dataset(
            _config.DataConfig(repo_id=REPO_ID, sample_weights_path=str(path)), 10, _model_config())


def test_sample_weights_and_race_targets_are_not_combined(tmp_path, fake_lerobot):
    path = _write_sidecar(tmp_path / "weights.json")
    data_config = _config.DataConfig(repo_id=REPO_ID, sample_weights_path=str(path),
                                     race_targets_dir=str(tmp_path / "race"))
    with pytest.raises(ValueError, match="set one"):
        _data_loader.create_torch_dataset(data_config, 10, _model_config())


def test_loader_draws_by_episode_weight(tmp_path, fake_lerobot, monkeypatch):
    """The weighted sampler sees one weight per item: frames of an episode are drawn in proportion to its weight."""
    path = _write_sidecar(tmp_path / "weights.json")
    data_config = _config.DataConfig(repo_id=REPO_ID, sample_weights_path=str(path))
    dataset = _data_loader.create_torch_dataset(data_config, 10, _model_config())
    sampler = _data_loader.race_weighted_sampler(dataset.sample_weights, seed=0)
    assert len(sampler) == sum(EPISODE_LENGTHS), "one epoch is still the dataset's size in draws"
    draws = []
    for _ in range(2_000):
        draws.extend(iter(sampler))
    episode_of = np.repeat(np.arange(len(EPISODE_LENGTHS)), EPISODE_LENGTHS)
    share = np.bincount(episode_of[draws], minlength=3) / len(draws)
    mass = np.array(EPISODE_WEIGHTS) * np.array(EPISODE_LENGTHS)
    np.testing.assert_allclose(share, mass / mass.sum(), atol=0.01)


def test_factory_passes_the_override_through(tmp_path, monkeypatch):
    """--data.sample-weights-path on a registered YAM recipe reaches the DataConfig; unset it stays None."""
    name = "pi05_yam_stream5_i15_h35_inspection_phases_real"
    monkeypatch.setattr(sys, "argv", ["train", name, "--exp-name", "t", "--data.sample-weights-path", "/w.json"])
    train_config = _config.cli()
    assert train_config.data.create(tmp_path, train_config.model).sample_weights_path == "/w.json"
    default = _config.get_config(name)
    assert default.data.create(tmp_path, default.model).sample_weights_path is None
