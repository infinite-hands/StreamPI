"""The `_titans` twin: the two-arm sort recipe's model and data with a long-term memory on, its norm stats, and the
older frames of the overhead camera carried from the LeRobot sample to Observation.memory_images."""

import dataclasses
import pathlib

import jax
import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training.misc import ih_yam_config

FPS = 30
# A LeRobot video frame in the sample: float, channels first, in [0, 1]. Small: the chain resizes it to the model's.
FRAME_SHAPE = (3, 48, 64)


@pytest.fixture(scope="module", params=ih_yam_config.TITANS_TWIN_BASES)
def recipes(request):
    return _config.get_config(request.param), _config.get_config(request.param + "_titans")


def _created(config, monkeypatch, read_from=None):
    monkeypatch.setattr(type(config.data), "_load_norm_stats",
                        lambda self, assets_dir, asset_id: None if read_from is None else
                        read_from.append(pathlib.Path(assets_dir) / asset_id))
    return config.data.create(config.assets_dirs, config.model)


def test_titans_twin_is_its_base_recipe_with_the_memory_on(recipes, monkeypatch):
    base, twin = recipes
    assert twin.model.titans_frames == ih_yam_config.TITANS_FRAMES and not base.model.titans_frames
    assert twin.model.titans_stride == ih_yam_config.TITANS_STRIDE
    assert twin.model.titans_image_key == ih_yam_config.TITANS_IMAGE_KEY == "base_0_rgb"
    assert dataclasses.replace(twin.model, titans_frames=0) == base.model, "nothing else about the model differs"
    assert twin.data.repo_id == base.data.repo_id
    assert twin.data.base_config == base.data.base_config
    assert (twin.batch_size, twin.fsdp_devices, twin.ema_decay, twin.num_train_steps) == (
        base.batch_size, base.fsdp_devices, base.ema_decay, base.num_train_steps)
    assert twin.freeze_filter != base.freeze_filter, "the memory trains inside the frozen LLM"

    read_from = []
    created = _created(twin, monkeypatch, read_from)
    base_created = _created(base, monkeypatch, read_from)
    assert read_from[0] == read_from[1], "the twin normalizes with its base recipe's norm stats"
    assert created.memory_sequence_key == "observation.images.cam_high"
    assert base_created.memory_sequence_key is None
    repack = created.repack_transforms.inputs[0].structure
    assert all(repack[key] == key for key in _transforms.MEMORY_KEYS)
    assert set(repack) - set(base_created.repack_transforms.inputs[0].structure) == set(_transforms.MEMORY_KEYS)


def test_the_older_frames_are_the_calls_a_served_run_wrote_on():
    offsets = _transforms.titans_frame_offsets(hist_interval=15, frames=4, stride=4)
    assert offsets == (-240, -180, -120, -60), "oldest first: 4, 3, 2 and 1 writes back, 4 calls of 15 frames apart"
    assert _transforms.titans_frame_offsets(hist_interval=15, frames=16, stride=4)[0] / FPS == -32.0


def test_split_memory_frames_takes_the_older_frames_off_the_window():
    frames, window = 3, 5
    sequence = np.arange(frames + window)[:, None]
    padded = np.zeros(frames + window, dtype=bool)
    padded[0] = True   # the oldest older frame lies before the episode's start
    data = _transforms.SplitMemoryFrames("camera", frames)({"camera": sequence, "camera_is_pad": padded})
    assert data["memory_image"][:, 0].tolist() == [2, 1, 0], "newest first"
    assert data["memory_mask"].tolist() == [True, True, False]
    assert data["camera"][:, 0].tolist() == list(range(frames, frames + window))
    assert data["camera_is_pad"].tolist() == [False] * window
    unpadded = _transforms.SplitMemoryFrames("camera", frames)({"camera": sequence})
    assert unpadded["memory_mask"].all()


def test_a_sample_carries_its_older_frames_to_the_model(recipes, monkeypatch):
    """The loader's own transforms on a LeRobot-shaped sample: the older frames reach Observation.memory_images
    newest first, resized like the window, with the episode-start mask; the window and the other cameras are untouched."""
    _base, twin = recipes
    created = _created(twin, monkeypatch)
    frames, interval, horizon = twin.model.titans_frames, created.hist_interval, created.hist_horizon
    window = (horizon - 1) * interval + 1
    sample = {}
    for camera in ih_yam_config.YAM_CAMERAS:
        count = frames + window if f"observation.images.{camera}" == created.memory_sequence_key else window
        # Frame k is filled with k / 255, so the order survives as a pixel value.
        sample[f"observation.images.{camera}"] = np.stack(
            [np.full(FRAME_SHAPE, index / 255.0, np.float32) for index in range(count)])
        sample[f"observation.images.{camera}_is_pad"] = np.zeros(count, dtype=bool)
    sample[f"{created.memory_sequence_key}_is_pad"][:3] = True
    rng = np.random.default_rng(0)
    sample["observation.state"] = rng.normal(size=14).astype(np.float32)
    sample["action"] = rng.normal(size=(twin.model.action_horizon, 14)).astype(np.float32)

    split = _transforms.SplitMemoryFrames(created.memory_sequence_key, frames)
    jitter = _transforms.TemporalJitter(created.jitter_range, interval, horizon, created.hist_sequence_keys, False)
    chain = _transforms.compose([*created.repack_transforms.inputs, *created.data_transforms.inputs,
                                 *created.model_transforms.inputs])
    batch = jax.tree.map(lambda leaf: np.asarray(leaf)[None], chain(jitter(split(sample))))
    observation = _model.Observation.from_dict(batch)

    assert observation.memory_images.shape == (1, frames, *_model.IMAGE_RESOLUTION, 3)
    centre = np.round((np.asarray(observation.memory_images)[0, :, 112, 112, 0] + 1.0) / 2.0 * 255.0).astype(int)
    assert centre.tolist() == list(range(frames - 1, -1, -1)), "newest first"
    assert np.asarray(observation.memory_mask)[0].tolist() == [True] * (frames - 3) + [False] * 3
    window_centre = np.round((np.asarray(observation.images["base_0_rgb"])[0, :, 112, 112, 0] + 1.0) / 2.0 * 255.0)
    assert window_centre.astype(int).tolist() == [frames + slot * interval for slot in range(horizon)]
    assert observation.images["left_wrist_0_rgb"].shape == (1, horizon, *_model.IMAGE_RESOLUTION, 3)
