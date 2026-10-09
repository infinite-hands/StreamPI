"""RACE's training data: the sidecar of per-frame transition targets and sampling weights, its validation, the
per-sample target window, and the weighted sampler. Toy sidecars on disk, no LeRobot dataset."""

import json
import pathlib

import datasets
import numpy as np
import pytest
import torch

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.policies import agilex_policy
from openpi.training import data_loader as _data_loader

REPO_ID = "local/toy_race"
HORIZON = 30
# Three episodes: shorter than the horizon, longer than it, and a short last one.
EPISODE_LENGTHS = (5, 40, 7)


def _write_store(directory: pathlib.Path, *, meta_overrides=None, arrays=None) -> dict[str, np.ndarray]:
    episode_index = np.concatenate([np.full(n, e, dtype=np.int32) for e, n in enumerate(EPISODE_LENGTHS)])
    frame_index = np.concatenate([np.arange(n, dtype=np.int32) for n in EPISODE_LENGTHS])
    n_frames = len(episode_index)
    store = {
        "episode_index": episode_index,
        "frame_index": frame_index,
        # A distinct value per frame, so a slice shows exactly which frames it read.
        "transition": (np.arange(n_frames, dtype=np.float32) + 1) / (n_frames + 1),
        "weight": np.linspace(0.5, 2.0, n_frames, dtype=np.float32),
        **(arrays or {}),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in store.items():
        np.save(directory / f"{name}.npy", array)
    meta = {"version": 1, "repo_id": REPO_ID, "n_frames": n_frames, **(meta_overrides or {})}
    (directory / "meta.json").write_text(json.dumps(meta))
    return store


def _sample(store, index: int) -> dict:
    return {"index": np.int64(index), "episode_index": np.int64(store["episode_index"][index]),
            "frame_index": np.int64(store["frame_index"][index])}


def test_target_window_respects_episode_bounds(tmp_path):
    """(d) Rows i-1 .. i+H of the anchor's episode; frames before its start or past its end are masked and 0."""
    directory = tmp_path / "store"
    store = _write_store(directory)
    transform = _transforms.RaceTargets(str(directory), HORIZON)
    n_frames = sum(EPISODE_LENGTHS)

    def window(index):
        out = transform(_sample(store, index))
        assert out["transition_window"].shape == (HORIZON + 2,) and out["transition_window"].dtype == np.float32
        return out["transition_window"], out["transition_window_mask"]

    # The dataset's first frame: row i-1 does not exist; episode 0 has 5 frames, so rows 1..5 of the window.
    values, inside = window(0)
    assert inside.tolist() == [False] + [True] * 5 + [False] * (HORIZON - 4)
    np.testing.assert_array_equal(values[1:6], store["transition"][0:5])
    assert not values[~inside].any()

    # Episode 1's first frame (global 5): its previous frame belongs to episode 0, so it is masked.
    values, inside = window(5)
    assert not inside[0] and inside[1:].all()   # 40 frames cover rows 5 .. 5+H
    np.testing.assert_array_equal(values[1:], store["transition"][5:5 + HORIZON + 1])

    # Inside episode 1, near its end (global 40 = frame 35 of 40): rows 39, 40, ..., 44 then episode 2.
    values, inside = window(40)
    np.testing.assert_array_equal(np.flatnonzero(inside), np.arange(0, 6))   # frames 39 .. 44
    np.testing.assert_array_equal(values[:6], store["transition"][39:45])
    assert not values[6:].any()

    # The dataset's last frame: only itself and its predecessor.
    values, inside = window(n_frames - 1)
    assert inside.tolist() == [True, True] + [False] * HORIZON
    np.testing.assert_array_equal(values[:2], store["transition"][n_frames - 2:])

    # A sample whose episode the sidecar places elsewhere is refused.
    wrong = _sample(store, 7) | {"episode_index": np.int64(0)}
    with pytest.raises(ValueError, match="disagree"):
        transform(wrong)


def test_check_race_targets_accepts_a_fitting_store(tmp_path):
    directory = tmp_path / "store"
    _write_store(directory)
    store = _data_loader.check_race_targets(str(directory), REPO_ID, sum(EPISODE_LENGTHS))
    assert set(store) == set(_transforms.RACE_STORE_ARRAYS)


@pytest.mark.parametrize(
    ("meta", "arrays", "message"),
    [
        ({"version": 2}, None, "version"),
        ({"repo_id": "local/other"}, None, "written for"),
        ({"n_frames": 10}, None, "holds 10 frames"),
        (None, {"weight": np.zeros(sum(EPISODE_LENGTHS), dtype=np.float32)}, "weight"),
        (None, {"transition": np.full(sum(EPISODE_LENGTHS), 1.5, dtype=np.float32)}, "outside"),
        (None, {"transition": np.zeros(sum(EPISODE_LENGTHS), dtype=np.float64)}, "transition.npy is float64"),
        (None, {"frame_index": np.zeros(3, dtype=np.int32)}, "frame_index.npy is int32\\[3\\]"),
    ],
)
def test_check_race_targets_refuses_a_store_that_does_not_fit(tmp_path, meta, arrays, message):
    directory = tmp_path / "store"
    _write_store(directory, meta_overrides=meta, arrays=arrays)
    with pytest.raises(ValueError, match=message):
        _data_loader.check_race_targets(str(directory), REPO_ID, sum(EPISODE_LENGTHS))


def test_check_race_targets_refuses_a_missing_store(tmp_path):
    with pytest.raises(FileNotFoundError, match="meta.json"):
        _data_loader.check_race_targets(str(tmp_path / "absent"), REPO_ID, 1)
    directory = tmp_path / "store"
    _write_store(directory)
    (directory / "weight.npy").unlink()
    with pytest.raises(ValueError, match="weight.npy is missing"):
        _data_loader.check_race_targets(str(directory), REPO_ID, sum(EPISODE_LENGTHS))


def test_sample_weights_follow_each_items_frame(tmp_path):
    directory = tmp_path / "store"
    store = _write_store(directory)
    # Items in an order other than the frames': the weights must follow the frame each item holds.
    frames = np.array([10, 0, 51, 7])
    weights = _data_loader.race_sample_weights(store, frames, store["episode_index"][frames], str(directory))
    np.testing.assert_array_equal(weights, store["weight"][frames].astype(np.float64))
    with pytest.raises(ValueError, match="other episodes"):
        _data_loader.race_sample_weights(store, frames, np.zeros(4, dtype=np.int64), str(directory))
    with pytest.raises(ValueError, match="cover"):
        _data_loader.race_sample_weights(store, np.array([52]), np.array([2]), str(directory))


def test_lerobot_frame_columns_reads_the_hf_table():
    """The columns come off the dataset's hf_dataset in item order (the call LeRobotDataset supports)."""
    table = datasets.Dataset.from_dict({"index": [3, 4, 5], "episode_index": [1, 1, 2], "action": [0.0, 1.0, 2.0]})
    table.set_transform(lambda batch: {key: [torch.as_tensor(v) for v in values] for key, values in batch.items()})

    class FakeLeRobot:
        hf_dataset = table

    index, episode = _data_loader._lerobot_frame_columns(FakeLeRobot())
    assert index.tolist() == [3, 4, 5] and episode.tolist() == [1, 1, 2]


def test_weighted_sampler_draws_in_proportion_to_weight():
    """(e) Frequencies follow the weights."""
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    sampler = _data_loader.race_weighted_sampler(weights, seed=0)
    assert len(sampler) == 4
    draws = []
    for _ in range(25_000):   # 100k draws over repeated epochs, as the loader's restarts do
        draws.extend(iter(sampler))
    frequency = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(frequency, weights / weights.sum(), atol=0.005)
    again = list(iter(_data_loader.race_weighted_sampler(weights, seed=0)))
    assert again == draws[:4], "seeded"


def test_torch_loader_uses_the_weighted_sampler(monkeypatch):
    """A dataset carrying sample weights is drawn through the weighted sampler; one without keeps shuffling."""
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    weights = np.arange(1, 9, dtype=np.float64)

    def weighted_dataset(data_config, action_horizon, model_config):
        return _data_loader.TransformedDataset(_data_loader.FakeDataset(model_config, 8), [], sample_weights=weights)

    monkeypatch.setattr(_data_loader, "create_torch_dataset", weighted_dataset)
    data_config = _data_loader._config.DataConfig(repo_id="fake")
    loader = _data_loader.create_torch_data_loader(data_config, model_config, 50, 4, skip_norm_stats=True,
                                                   shuffle=True, num_batches=1)
    sampler = loader._data_loader.torch_loader.sampler
    assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
    np.testing.assert_array_equal(sampler.weights.numpy(), weights)
    assert len(list(loader)) == 1

    monkeypatch.setattr(_data_loader, "create_torch_dataset",
                        lambda data_config, action_horizon, model_config: _data_loader.FakeDataset(model_config, 8))
    plain = _data_loader.create_torch_data_loader(data_config, model_config, 50, 4, skip_norm_stats=True,
                                                  shuffle=True, num_batches=1)
    assert isinstance(plain._data_loader.torch_loader.sampler, torch.utils.data.RandomSampler)


def test_race_targets_survive_the_input_pipeline(tmp_path):
    """The target keys pass AgilexInputs (which rebuilds the sample) untouched."""
    directory = tmp_path / "store"
    store = _write_store(directory)
    sample = _transforms.RaceTargets(str(directory), HORIZON)(_sample(store, 6))
    raw = {"images": {"cam_left_wrist": np.zeros((1, 3, 8, 8), dtype=np.uint8)},
           "state": np.zeros(14, dtype=np.float32), "actions": np.zeros((HORIZON, 14), dtype=np.float32),
           **{key: sample[key] for key in _transforms.RACE_TARGET_KEYS}}
    out = agilex_policy.AgilexInputs()(raw)
    for key in _transforms.RACE_TARGET_KEYS:
        np.testing.assert_array_equal(out[key], sample[key])
