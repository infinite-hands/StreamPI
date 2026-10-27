"""create_torch_dataset's history window: sparse decoding requests fewer frames and yields the same samples."""

import random
import types

import numpy as np
import pytest

import openpi.training.config as config
import openpi.training.data_loader as data_loader
import openpi.transforms as transforms

FPS = 30
JITTER = (-2, -1, 0, 1, 2)


class _Meta:
    def __init__(self, repo_id):
        self.fps = FPS
        self.tasks = {}
        self.total_frames = 100


class _Dataset:
    """Every requested frame's value is its offset from the current frame, so a sample shows which frames it used."""

    requested = None

    def __init__(self, repo_id, delta_timestamps=None):
        type(self).requested = delta_timestamps
        self._deltas = delta_timestamps

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {key: np.asarray([round(t * FPS) for t in times], np.float32).reshape(-1, 1) for key, times in self._deltas.items()}


@pytest.fixture(autouse=True)
def _fake_lerobot(monkeypatch):
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _Meta)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", _Dataset)


def _dataset(*, sparse: bool, jitter: bool):
    data_config = config.DataConfig(
        repo_id="r", hist_sequence_keys=("cam",), hist_horizon=5, hist_interval=20, jitter_range=JITTER,
        enable_jitter=jitter, sparse_hist_decode=sparse, prompt_from_task=False)
    return data_loader.create_torch_dataset(data_config, 4, types.SimpleNamespace(race=False, vlash_branches=0))


@pytest.mark.parametrize(("jitter", "dense_frames", "sparse_frames"), [(True, 81, 21), (False, 81, 5)])
def test_sparse_decoding_requests_only_the_frames_the_jitter_can_pick(jitter, dense_frames, sparse_frames):
    _dataset(sparse=False, jitter=jitter)
    assert len(_Dataset.requested["cam"]) == dense_frames
    _dataset(sparse=True, jitter=jitter)
    assert len(_Dataset.requested["cam"]) == sparse_frames
    assert len(_Dataset.requested["actions"]) == 4


@pytest.mark.parametrize("jitter", [True, False])
def test_dense_and_sparse_datasets_give_the_same_history_frames(jitter, monkeypatch):
    for choice in JITTER:
        monkeypatch.setattr(random, "choice", lambda _seq, choice=choice: choice)
        dense = _dataset(sparse=False, jitter=jitter)[0]["cam"]
        sparse = _dataset(sparse=True, jitter=jitter)[0]["cam"]
        assert dense.shape == (5, 1)
        assert np.array_equal(dense, sparse), choice


class _Tokenizer:
    def __init__(self):
        self.states = []

    def tokenize(self, prompt, state=None, *, task_only=False):
        self.states.append(np.array(state))
        return np.zeros(4, np.int32), np.ones(4, bool)


@pytest.mark.parametrize(("dropout", "hidden"), [(0.0, False), (1.0, True)])
def test_the_training_loader_switches_state_dropout_on_only_when_asked(dropout, hidden):
    tokenizer = _Tokenizer()
    model_transforms = transforms.Group(inputs=[transforms.TokenizePrompt(tokenizer, discrete_state_input=True)])
    data_config = config.DataConfig(repo_id="fake", model_transforms=model_transforms, state_dropout=dropout)
    sample = {"prompt": "p", "state": np.linspace(-0.8, 0.8, 14)}

    data_loader.transform_dataset([sample], data_config, skip_norm_stats=True)[0]

    assert (not tokenizer.states[0].any()) == hidden
    # The serving path builds its transforms straight from the config, so the config's own transform stays unchanged.
    assert model_transforms.inputs[0].state_dropout == 0.0
