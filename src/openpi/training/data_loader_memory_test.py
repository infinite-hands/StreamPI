"""The memory frames' data path: which frames the loader asks for, how they are split off, and what the transforms do to them."""

import dataclasses
import random
import types

import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.policies import agilex_policy
import openpi.shared.normalize as normalize
import openpi.training.config as config
import openpi.training.data_loader as data_loader
import openpi.training.misc.ih_yam_config as ih_yam_config
import openpi.transforms as transforms

FPS = 30
HORIZON, INTERVAL, FRAMES, STRIDE = 5, 15, 16, 4
POSITION = 700  # the sample's frame index in its episode
MODEL = types.SimpleNamespace(race=False, vlash_branches=0, hist_horizon=HORIZON, memory_frames=FRAMES, memory_stride=STRIDE)


class _Meta:
    def __init__(self, repo_id):
        self.fps = FPS
        self.tasks = {}
        self.total_frames = 100


class _Dataset:
    """Every requested frame's value is its offset from the current frame; frames before the episode are flagged as LeRobot does."""

    requested = None
    position = POSITION

    def __init__(self, repo_id, delta_timestamps=None):
        type(self).requested = delta_timestamps
        self._deltas = delta_timestamps

    def __len__(self):
        return 3

    def __getitem__(self, index):
        item = {}
        for key, times in self._deltas.items():
            offsets = np.asarray([round(t * FPS) for t in times])
            item[key] = offsets.astype(np.float32).reshape(-1, 1)
            item[f"{key}_is_pad"] = offsets < -self.position
        return item


@pytest.fixture(autouse=True)
def _fake_lerobot(monkeypatch):
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _Meta)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", _Dataset)


def _dataset(model=MODEL, *, sparse: bool, jitter: bool = False, keys=("cam",), horizon=HORIZON):
    data_config = config.DataConfig(
        repo_id="r", hist_sequence_keys=keys, hist_horizon=horizon, hist_interval=INTERVAL,
        jitter_range=(-2, -1, 0, 1, 2), enable_jitter=jitter, sparse_hist_decode=sparse, prompt_from_task=False)
    return data_loader.create_torch_dataset(data_config, 4, model)


def test_the_loaders_offsets_are_the_models_slots():
    model = pi0_config.Pi0Config(
        pi05=True, hist_horizon=HORIZON, image_keys=("right_wrist_0_rgb",), memory_frames=FRAMES, memory_stride=STRIDE)
    offsets = transforms.memory_frame_offsets(HORIZON, INTERVAL, FRAMES, STRIDE)
    assert offsets == tuple(-age * INTERVAL for age in reversed(model.memory_ages))
    assert (offsets[0], offsets[-1], len(offsets)) == (-68 * INTERVAL, -8 * INTERVAL, FRAMES)
    # Beyond the history window (the oldest dense frame is 4 calls back), and 34 seconds at the most.
    assert offsets[-1] < -(HORIZON - 1) * INTERVAL
    assert -offsets[0] / FPS == 34.0


@pytest.mark.parametrize(("sparse", "window"), [(False, (HORIZON - 1) * INTERVAL + 1), (True, HORIZON)])
def test_the_memory_frames_are_fetched_beside_the_window_and_split_off(sparse, window):
    _dataset(sparse=sparse)
    requested = _Dataset.requested["cam"]
    assert len(requested) == FRAMES + window
    assert requested == sorted(requested)  # oldest first: the split reads the leading entries

    sample = _dataset(sparse=sparse)[0]
    older = [-age * INTERVAL for age in pi0_config.Pi0Config(
        pi05=True, hist_horizon=HORIZON, image_keys=("right_wrist_0_rgb",), memory_frames=FRAMES, memory_stride=STRIDE).memory_ages]
    # Slot i is the frame memory_ages[i] calls back, newest first, and exists only if the episode is that old.
    assert sample["memory_image"][:, 0].tolist() == older
    assert sample["memory_mask"].tolist() == [offset >= -POSITION for offset in older]
    assert sample["memory_mask"].sum() == 10
    # The window is exactly what a model without memory gets.
    assert np.array_equal(sample["cam"], _dataset(types.SimpleNamespace(race=False, vlash_branches=0), sparse=sparse)[0]["cam"])
    assert sample["memory_image"].flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("sparse", [False, True])
def test_the_jitter_moves_the_window_but_not_the_memory(sparse, monkeypatch):
    seen = []
    for choice in (-2, 0, 2):
        monkeypatch.setattr(random, "choice", lambda _seq, choice=choice: choice)
        sample = _dataset(sparse=sparse, jitter=True)[0]
        seen.append((sample["cam"][:, 0].tolist(), sample["memory_image"][:, 0].tolist()))
    assert len({tuple(window) for window, _ in seen}) == 3
    assert len({tuple(memory) for _, memory in seen}) == 1


def test_a_model_without_memory_fetches_what_it_always_did():
    _dataset(types.SimpleNamespace(race=False, vlash_branches=0), sparse=True)
    assert len(_Dataset.requested["cam"]) == HORIZON
    assert "memory_image" not in _dataset(types.SimpleNamespace(race=False, vlash_branches=0), sparse=True)[0]


def test_the_loader_refuses_what_the_memory_is_not_wired_for():
    with pytest.raises(ValueError, match="exactly one"):
        _dataset(sparse=True, keys=("cam", "other"))
    with pytest.raises(ValueError, match="must match the model's"):
        _dataset(sparse=True, horizon=3)


def test_the_split_without_pad_flags_treats_every_slot_as_present():
    sequence = np.arange(FRAMES + 3, dtype=np.float32).reshape(-1, 1)
    out = transforms.SplitMemoryFrames("cam", FRAMES)({"cam": sequence})
    assert out["memory_mask"].all()
    assert out["memory_image"][:, 0].tolist() == list(range(FRAMES - 1, -1, -1))
    assert out["cam"][:, 0].tolist() == [16.0, 17.0, 18.0]


def _chain_input(frames=FRAMES, camera_hw=(48, 64)):
    rng = np.random.default_rng(0)
    return {
        "images": {"cam_right_wrist": rng.uniform(size=(HORIZON, 3, *camera_hw)).astype(np.float32)},
        "state": np.zeros(14, np.float32), "actions": np.zeros((4, 14), np.float32),
        "memory_image": rng.uniform(size=(frames, 3, *camera_hw)).astype(np.float32),
        "memory_mask": np.asarray([True] * (frames - 3) + [False] * 3),
    }


def test_the_policy_inputs_and_resize_carry_the_memory_frames_like_the_windows_own():
    data = agilex_policy.AgilexInputs()(_chain_input())
    window, memory = data["image"]["right_wrist_0_rgb"], data["memory_image"]
    assert memory.dtype == window.dtype == np.uint8
    assert memory.shape == (FRAMES, 48, 64, 3)
    assert window.shape == (HORIZON, 48, 64, 3)
    assert data["memory_mask"].dtype == bool
    assert data["memory_mask"].tolist() == [True] * 13 + [False] * 3

    resized = transforms.ResizeImages(56, 56)(data)
    assert resized["memory_image"].shape == (FRAMES, 56, 56, 3)
    assert resized["image"]["right_wrist_0_rgb"].shape == (HORIZON, 56, 56, 3)

    # The same frame decodes to the same pixels whether it sits in the window or in the memory.
    shared = _chain_input()
    shared["memory_image"][0] = shared["images"]["cam_right_wrist"][0]
    again = agilex_policy.AgilexInputs()(shared)
    assert np.array_equal(again["memory_image"][0], again["image"]["right_wrist_0_rgb"][0])


def test_a_request_without_memory_frames_is_untouched():
    request = _chain_input()
    del request["memory_image"], request["memory_mask"]
    served = transforms.ResizeImages(56, 56)(agilex_policy.AgilexInputs()(request))
    assert "memory_image" not in served
    assert "memory_mask" not in served


def test_the_memory_recipe_repacks_the_frames_only_for_a_memory_model():
    base = ih_yam_config.yam_repack(("cam_right_wrist",))
    carried = ih_yam_config.with_memory_frames(base)
    assert set(carried.inputs[0].structure) - set(base.inputs[0].structure) == set(transforms.MEMORY_KEYS)

    sample = {"observation.images.cam_right_wrist": np.zeros((HORIZON, 3, 4, 4)), "observation.state": np.zeros(14),
              "action": np.zeros((4, 14)), "memory_image": np.ones((FRAMES, 3, 4, 4)), "memory_mask": np.ones(FRAMES, bool)}
    repacked = carried.inputs[0](dict(sample))
    assert set(transforms.MEMORY_KEYS) <= set(repacked)
    assert not set(transforms.MEMORY_KEYS) & set(base.inputs[0](dict(sample)))


HISTORY_MODEL = types.SimpleNamespace(
    race=False, vlash_branches=0, hist_horizon=HORIZON, state_history=3, state_history_first_age=2)


def test_the_state_history_offsets_are_the_models_ages():
    model = pi0_config.Pi0Config(
        pi05=True, state_history=3, state_history_first_age=2, state_history_dims=(0,), image_keys=("right_wrist_0_rgb",))
    offsets = transforms.state_history_offsets(INTERVAL, 2, 3)
    assert offsets == tuple(-age * INTERVAL for age in reversed(model.state_history_ages)) == (-60, -45, -30)


def test_the_state_column_is_fetched_at_the_history_offsets_and_split(monkeypatch):
    sample = _dataset(HISTORY_MODEL, sparse=True)[0]
    requested = _Dataset.requested["observation.state"]
    assert [round(t * FPS) for t in requested] == [-60, -45, -30, 0]
    # The current state is what the column always held; row j of the history is the state first_age + j calls back.
    assert sample["observation.state"].tolist() == [0.0]
    assert sample["state_history"][:, 0].tolist() == [-30.0, -45.0, -60.0]
    assert sample["state_history_mask"].tolist() == [True, True, True]
    assert sample["state_history"].flags["C_CONTIGUOUS"]
    assert "memory_image" not in sample

    monkeypatch.setattr(_Dataset, "position", 40)
    assert _dataset(HISTORY_MODEL, sparse=True)[0]["state_history_mask"].tolist() == [True, False, False]


def test_a_model_without_a_state_history_never_asks_for_the_state_column():
    _dataset(types.SimpleNamespace(race=False, vlash_branches=0), sparse=True)
    assert "observation.state" not in _Dataset.requested


def test_memory_frames_and_a_state_history_split_independently():
    both = types.SimpleNamespace(**{**vars(HISTORY_MODEL), "memory_frames": FRAMES, "memory_stride": STRIDE})
    sample = _dataset(both, sparse=True)[0]
    assert sample["memory_image"].shape == (FRAMES, 1)
    assert sample["state_history"].shape == (3, 1)
    assert sample["cam"].shape == (HORIZON, 1)
    assert sample["cam"][:, 0].tolist() == _dataset(types.SimpleNamespace(race=False, vlash_branches=0), sparse=True)[0]["cam"][:, 0].tolist()


def test_a_state_history_refuses_vlash_offsets():
    data_config = config.DataConfig(
        repo_id="r", hist_sequence_keys=("cam",), hist_horizon=HORIZON, hist_interval=INTERVAL, enable_jitter=False,
        sparse_hist_decode=True, prompt_from_task=False, vlash_max_offset=3)
    with pytest.raises(ValueError, match="VLASH"):
        data_loader.create_torch_dataset(data_config, 4, HISTORY_MODEL)


def test_the_policy_inputs_pass_the_state_history_through_and_nothing_else_changes():
    request = _chain_input()
    request["state_history"] = np.ones((3, 14), np.float64)
    request["state_history_mask"] = np.asarray([True, True, False])
    data = agilex_policy.AgilexInputs()(request)
    assert data["state_history"].dtype == np.float32
    assert data["state_history"].shape == (3, 14)
    assert data["state_history_mask"].tolist() == [True, True, False]
    assert "state_history" not in agilex_policy.AgilexInputs()(_chain_input())


def test_the_earlier_states_are_normalized_like_the_state_and_dropped_from_the_output_stats():
    rng = np.random.default_rng(0)
    state_stats = normalize.NormStats(
        mean=rng.normal(size=14), std=rng.uniform(1, 2, size=14), q01=-rng.uniform(1, 2, size=14), q99=rng.uniform(1, 2, size=14))
    stats = {"state": state_stats, transforms.STATE_HISTORY_NORM_KEY: state_stats}
    rows = rng.normal(size=(3, 14))
    out = transforms.Normalize(stats, use_quantiles=True)({"state": rows[0], "state_history": rows})
    assert np.allclose(out["state_history"][0], out["state"])
    assert out["state_history"].shape == (3, 14)
    assert set(transforms.output_norm_stats({**stats, transforms.NOW_STATE_KEY: state_stats})) == {"state"}


def test_the_history_recipe_repacks_and_normalizes_the_states_only_for_a_history_model():
    name = ih_yam_config.HIST_TWIN_BASES[0] + "_hist"
    twin, base = config.get_config(name), config.get_config(ih_yam_config.HIST_TWIN_BASES[0])
    changed = {key for key, value in vars(twin.model).items() if value != vars(base.model)[key]}
    # first_age 2 is the default, so only the rows and the dims differ from the base.
    assert changed == {"state_history", "state_history_dims"}
    assert twin.model.state_history_dims == twin.data.active_state_dims == (7, 8, 9, 10, 11, 12, 13)
    assert (twin.model.state_history, twin.model.state_history_first_age) == (8, 2)
    assert twin.weight_loader == base.weight_loader

    created = twin.data.create(twin.assets_dirs, twin.model)
    assert set(transforms.STATE_HISTORY_KEYS) <= set(created.repack_transforms.inputs[0].structure)
    assert created.norm_stats is None or transforms.STATE_HISTORY_NORM_KEY in created.norm_stats

    off = twin.data.create(twin.assets_dirs, dataclasses.replace(twin.model, state_history=0, state_history_dims=()))
    assert not set(transforms.STATE_HISTORY_KEYS) & set(off.repack_transforms.inputs[0].structure)
    assert off.norm_stats is None or transforms.STATE_HISTORY_NORM_KEY not in off.norm_stats
