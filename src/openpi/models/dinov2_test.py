"""The Flax DINOv2 against the Hugging Face (PyTorch) reference, with random weights so nothing is downloaded."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers

import openpi.models.dinov2 as dinov2

TINY = dinov2.Config(width=64, depth=2, heads=4, mlp_dim=256, grid=4)


def _reference():
    torch.manual_seed(0)
    config = transformers.Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, mlp_ratio=4,
                                       image_size=56, patch_size=14)
    model = transformers.Dinov2Model(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    return model


def _flax_params(model):
    return dinov2.convert_hf_state_dict({key: value.numpy() for key, value in model.state_dict().items()}, TINY)


def _both(model, side):
    image = np.random.default_rng(0).uniform(-1, 1, size=(2, side, side, 3)).astype(np.float32)
    pixels = ((image + 1.0) / 2.0 - np.asarray(dinov2.IMAGENET_MEAN, np.float32)) / np.asarray(dinov2.IMAGENET_STD, np.float32)
    with torch.no_grad():
        expected = model(pixel_values=torch.from_numpy(pixels).permute(0, 3, 1, 2)).last_hidden_state.numpy()
    actual = dinov2.Dinov2(TINY).apply({"params": _flax_params(model)}, jnp.asarray(image))
    return expected, np.asarray(actual)


@pytest.mark.parametrize(("in_size", "out_size"), [(4, 6), (16, 24), (37, 24), (37, 16), (24, 24)])
def test_bicubic_resize_matches_torch(in_size, out_size):
    x = np.random.default_rng(1).normal(size=(2, in_size, in_size, 5)).astype(np.float32)
    # Against torch in float64: torch's own float32 result is off from it by up to 1e-5 when downsampling.
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(x.astype(np.float64)).permute(0, 3, 1, 2), size=(out_size, out_size), mode="bicubic", align_corners=False
    ).permute(0, 2, 3, 1).numpy()
    actual = np.asarray(dinov2.resize_bicubic(jnp.asarray(x), (out_size, out_size)))
    assert np.allclose(actual, expected, atol=2e-6)


def test_the_network_matches_the_reference_at_its_native_grid():
    expected, actual = _both(_reference(), side=56)
    assert actual.shape == (2, 1 + 16, 64)
    assert np.allclose(actual, expected, atol=1e-4), np.abs(actual - expected).max()
    assert np.abs(expected).max() > 0.5


def test_the_network_matches_the_reference_when_the_position_embedding_is_resampled():
    expected, actual = _both(_reference(), side=84)
    assert actual.shape == (2, 1 + 36, 64)
    assert np.allclose(actual, expected, atol=1e-4), np.abs(actual - expected).max()


def test_the_converter_reads_the_released_attention_names_and_refuses_unknown_keys():
    model = _reference()
    new_style = {key: value.numpy() for key, value in model.state_dict().items()}
    released = {}
    for key, value in new_style.items():
        for new, old in (("attention.q_proj", "attention.attention.query"), ("attention.k_proj", "attention.attention.key"),
                         ("attention.v_proj", "attention.attention.value"), ("attention.o_proj", "attention.output.dense")):
            key = key.replace(new, old)  # noqa: PLW2901
        released[key] = value
    assert "encoder.layer.0.attention.attention.query.weight" in released

    from_new = dinov2.convert_hf_state_dict(new_style, TINY)
    from_released = dinov2.convert_hf_state_dict({"dinov2." + key: value for key, value in released.items()}, TINY)
    assert jax.tree.structure(from_new) == jax.tree.structure(from_released)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(from_new), jax.tree.leaves(from_released), strict=True))

    with pytest.raises(ValueError, match="does not use"):
        dinov2.convert_hf_state_dict({**new_style, "embeddings.surprise": np.zeros(3)}, TINY)


def test_the_real_base_variant_has_the_published_size():
    config = dinov2.get_config("base")
    shapes = jax.eval_shape(dinov2.Dinov2(config).init, jax.random.key(0), jax.ShapeDtypeStruct((1, 224, 224, 3), jnp.float32))
    count = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(shapes))
    assert 85e6 < count < 88e6
    assert "layer_11" in shapes["params"]
    assert shapes["params"]["pos_embedding"].shape == (1, 1 + 37 * 37, 768)
