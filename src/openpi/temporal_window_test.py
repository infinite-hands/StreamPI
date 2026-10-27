import random

import numpy as np
import pytest

import openpi.transforms as transforms

JITTER = (-2, -1, 0, 1, 2)


def _jitter(horizon, interval, enable, window_offsets=None):
    return transforms.TemporalJitter(JITTER, interval, horizon, ["cam"], enable, window_offsets)


@pytest.mark.parametrize(("horizon", "interval", "enable"), [(5, 20, True), (5, 10, True), (5, 20, False), (3, 5, True), (8, 15, True)])
def test_the_sparse_window_gives_exactly_the_dense_samples(horizon, interval, enable, monkeypatch):
    full = (horizon - 1) * interval + 1
    dense_sequence = np.arange(full * 6, dtype=np.float32).reshape(full, 2, 3)
    offsets = transforms.history_frame_offsets(horizon, interval, JITTER, enable_jitter=enable)
    sparse_sequence = dense_sequence[[full - 1 + offset for offset in offsets]]

    for jitter in JITTER:
        monkeypatch.setattr(random, "choice", lambda _seq, jitter=jitter: jitter)
        dense = _jitter(horizon, interval, enable)({"cam": dense_sequence.copy()})["cam"]
        sparse = _jitter(horizon, interval, enable, offsets)({"cam": sparse_sequence.copy()})["cam"]
        assert np.array_equal(dense, sparse), jitter


def test_how_many_frames_the_sparse_window_decodes():
    assert len(transforms.history_frame_offsets(5, 20, JITTER, enable_jitter=True)) == 21
    assert len(transforms.history_frame_offsets(5, 20, JITTER, enable_jitter=False)) == 5
    assert transforms.history_frame_offsets(5, 20, JITTER, enable_jitter=False) == (-80, -60, -40, -20, 0)
    offsets = transforms.history_frame_offsets(5, 20, JITTER, enable_jitter=True)
    assert offsets == tuple(sorted(offsets))
    assert offsets[0] == -80
    assert offsets[-1] == 0
    assert 81 / len(offsets) > 3.8
