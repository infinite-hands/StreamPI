import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model

SIDE = 56
KEYS = ("base_0_rgb", "right_wrist_0_rgb")


def _image() -> np.ndarray:
    ramp = np.linspace(-1.0, 1.0, SIDE, dtype=np.float32)
    plane = np.add.outer(ramp, ramp) / 2
    return np.broadcast_to(plane[None, None, :, :, None], (1, 1, SIDE, SIDE, 3)).copy()


def _observation() -> _model.Observation:
    return _model.Observation(
        images={key: jnp.asarray(_image()) for key in KEYS},
        image_masks={key: jnp.ones((1,), dtype=bool) for key in KEYS},
        state=jnp.zeros((1, 14), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((1, 4), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 4), dtype=bool),
    )


def _preprocess(*, train: bool, wrist_augmentation: bool, seed: int = 0):
    return _model.preprocess_observation(
        jax.random.key(seed), _observation(), train=train, image_keys=KEYS, image_resolution=(SIDE, SIDE),
        wrist_augmentation=wrist_augmentation).images


def test_off_by_default_changes_nothing():
    plain = _model.preprocess_observation(
        jax.random.key(0), _observation(), train=True, image_keys=KEYS, image_resolution=(SIDE, SIDE)).images
    off = _preprocess(train=True, wrist_augmentation=False)
    for key in KEYS:
        assert np.array_equal(np.asarray(plain[key]), np.asarray(off[key]))


def test_it_changes_only_the_wrist_camera_and_only_in_training():
    off, on = _preprocess(train=True, wrist_augmentation=False), _preprocess(train=True, wrist_augmentation=True)
    assert np.array_equal(np.asarray(off["base_0_rgb"]), np.asarray(on["base_0_rgb"]))
    assert not np.allclose(np.asarray(off["right_wrist_0_rgb"]), np.asarray(on["right_wrist_0_rgb"]))

    evaluation = _preprocess(train=False, wrist_augmentation=True)
    assert np.allclose(np.asarray(evaluation["right_wrist_0_rgb"]), _image())


def test_the_augmentation_is_light_bounded_and_repeatable():
    on = np.asarray(_preprocess(train=True, wrist_augmentation=True)["right_wrist_0_rgb"])
    off = np.asarray(_preprocess(train=True, wrist_augmentation=False)["right_wrist_0_rgb"])
    again = np.asarray(_preprocess(train=True, wrist_augmentation=True)["right_wrist_0_rgb"])
    assert on.shape == off.shape == (1, 1, SIDE, SIDE, 3)
    assert on.min() >= -1.0
    assert on.max() <= 1.0
    assert np.array_equal(on, again)
    assert np.corrcoef(on.ravel(), off.ravel())[0, 1] > 0.9


def test_degradation_pieces_do_what_they_say():
    rng = np.random.default_rng(0)
    noisy = jnp.asarray(rng.uniform(0, 1, size=(SIDE, SIDE, 3)), dtype=jnp.float32)
    key = jax.random.key(3)

    untouched = _model.wrist_degradation(key, noisy, blur_prob=0.0, noise_max=0.0)
    assert np.allclose(np.asarray(untouched), np.asarray(noisy), atol=1e-5)

    blurred = _model.wrist_degradation(key, noisy, blur_prob=1.0, sigma_range=(1.2, 1.2), noise_max=0.0)
    assert np.abs(np.diff(np.asarray(blurred), axis=0)).mean() < 0.7 * np.abs(np.diff(np.asarray(noisy), axis=0)).mean()

    smooth = jnp.full((SIDE, SIDE, 3), 0.5, dtype=jnp.float32)
    grainy = _model.wrist_degradation(key, smooth, blur_prob=0.0, noise_max=0.02)
    assert 0.0 < float(jnp.abs(grainy - smooth).mean()) < 0.03
