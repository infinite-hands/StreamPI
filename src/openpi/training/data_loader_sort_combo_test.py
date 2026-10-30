"""The sort-right memory recipe's data path: the image memory's frames beside unpacked VLASH offsets (the measured state at the lead,
the state at the image's frame, the offset). The state history is refused beside VLASH offsets, and the loader says so."""

import types

import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config
from openpi.training import data_loader
from openpi.training.data_loader_memory_test import _Dataset
from openpi.training.data_loader_memory_test import _fake_lerobot  # noqa: F401 -- an autouse fixture, applied by import
import openpi.transforms as transforms

HORIZON, INTERVAL, ACTION_HORIZON, OFFSET = 5, 50, 50, 49
FRAMES, STRIDE = 16, 1
MODEL = types.SimpleNamespace(race=False, vlash_branches=0, hist_horizon=HORIZON, memory_frames=FRAMES, memory_stride=STRIDE)


def _data_config(**overrides) -> config.DataConfig:
    return config.DataConfig(
        repo_id="r", hist_sequence_keys=("cam",), hist_horizon=HORIZON, hist_interval=INTERVAL, enable_jitter=False,
        sparse_hist_decode=True, prompt_from_task=False, vlash_max_offset=OFFSET, vlash_state_source="state",
        vlash_cond_now=True, **overrides)


def test_the_memory_slots_span_an_episode_at_the_whole_chunk_cadence():
    model = pi0_config.Pi0Config(pi05=True, hist_horizon=HORIZON, image_keys=("right_wrist_0_rgb",), memory_frames=FRAMES,
                                 memory_stride=STRIDE)
    offsets = transforms.memory_frame_offsets(HORIZON, INTERVAL, FRAMES, STRIDE)
    assert offsets == tuple(-age * INTERVAL for age in reversed(model.memory_ages))
    # The first slot sits one stride beyond the oldest window frame (4 calls back): 5 to 20 calls of 50 rows, 8.3 to 33.3 seconds at
    # 30 Hz -- inside the median 43 s sort episode, and close to the 34 s the 15-row bagging recipe's memory spans.
    assert (-offsets[-1], -offsets[0]) == (5 * INTERVAL, 20 * INTERVAL)
    assert (-offsets[-1] / 30, -offsets[0] / 30) == (pytest.approx(8.33, abs=0.01), pytest.approx(33.33, abs=0.01))


def test_the_loader_builds_a_sample_with_memory_frames_and_the_vlash_conditioning():
    dataset = data_loader.create_torch_dataset(_data_config(), ACTION_HORIZON, MODEL)
    requested = _Dataset.requested
    assert len(requested["cam"]) == HORIZON + FRAMES, "the window's sparse frames plus the memory frames"
    assert len(requested["actions"]) == ACTION_HORIZON + OFFSET, "the action rows TemporalOffset slices its window from"
    sample = dataset[0]
    assert len(sample["actions"]) == ACTION_HORIZON
    assert sample["memory_image"].shape[0] == FRAMES and sample["cam"].shape[0] == HORIZON
    assert transforms.NOW_STATE_KEY in sample and transforms.DELTA_KEY in sample, "the lead and the state at the image are carried"


def test_the_loader_refuses_a_state_history_beside_the_vlash_offsets():
    model = types.SimpleNamespace(**{**vars(MODEL), "state_history": 8, "state_history_first_age": 2})
    with pytest.raises(ValueError, match="VLASH"):
        data_loader.create_torch_dataset(_data_config(), ACTION_HORIZON, model)
