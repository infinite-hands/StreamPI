import jax
import numpy as np
import pytest

import openpi.models.siglip as siglip

KEY = "PaliGemma/img/pos_embedding"


def _shape(*dims):
    return jax.ShapeDtypeStruct(dims, np.float32)


def test_resize_posemb_resamples_the_patch_grid_and_keeps_a_constant_field_constant():
    loaded = {KEY: np.full((1, 256, 8), 0.5, np.float32), "PaliGemma/img/other": np.ones(3)}
    grown = siglip.resize_posemb(loaded, {KEY: _shape(1, 576, 8), "PaliGemma/img/other": _shape(3)})
    assert grown[KEY].shape == (1, 576, 8)
    assert np.allclose(grown[KEY], 0.5, atol=1e-5)
    assert grown["PaliGemma/img/other"] is loaded["PaliGemma/img/other"]


def test_resize_posemb_follows_a_gradient_across_the_grid():
    side = 16
    ramp = np.tile(np.linspace(0.0, 1.0, side, dtype=np.float32), (side, 1)).reshape(1, side * side, 1)
    grown = siglip.resize_posemb({KEY: ramp}, {KEY: _shape(1, 576, 1)})[KEY].reshape(24, 24)
    assert np.all(np.diff(grown[0]) > 0)
    assert grown[0, 0] < 0.1
    assert grown[0, -1] > 0.9


def test_resize_posemb_leaves_equal_shapes_alone_and_refuses_what_it_cannot_resample():
    same = {KEY: np.ones((1, 256, 8), np.float32)}
    assert siglip.resize_posemb(same, {KEY: _shape(1, 256, 8)})[KEY] is same[KEY]
    for reference in (_shape(1, 300, 8), _shape(1, 576, 4), _shape(2, 576, 8)):
        with pytest.raises(ValueError, match="cannot resample"):
            siglip.resize_posemb(same, {KEY: reference})
