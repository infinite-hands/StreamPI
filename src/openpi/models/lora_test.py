import flax.linen as nn
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.lora as lora


def test_lora_einsum_params_shape():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    lora0 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2))
    lora1 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, axes=(1, 2)))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    # Ensure that lora parameters are not initialized when LoRA is not used.
    params = einsum.init(key, eqn, x)
    assert "lora_a" not in params["params"]
    assert "lora_b" not in params["params"]

    # Check that default axes work.
    params_lora0 = lora0.init(key, eqn, x)
    assert params_lora0["params"]["lora_a"].shape == (3, 8, 32, 2)
    assert params_lora0["params"]["lora_b"].shape == (3, 8, 2, 4)

    # Check that user provided axes work.
    params_lora1 = lora1.init(key, eqn, x)
    assert params_lora1["params"]["lora_a"].shape == (3, 8, 2, 4)
    assert params_lora1["params"]["lora_b"].shape == (3, 2, 32, 4)


def test_lora_einsum_same_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    einsum_lora = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    params = einsum.init(key, eqn, x)
    output = einsum.apply(params, eqn, x)

    params_lora = einsum_lora.init(key, eqn, x)
    output_lora = einsum_lora.apply(params_lora, eqn, x)

    # Results are the same since the LoRA parameters are initialized to zeros.
    assert jnp.allclose(output, output_lora)


def test_lora_ffn_params_shape():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    assert params["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params["params"]["linear"].shape == (32, 8)

    params_lora = ffn_lora.init(key, x)
    assert params_lora["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params_lora["params"]["linear"].shape == (32, 8)
    assert params_lora["params"]["gating_einsum_lora_a"].shape == (2, 8, 2)
    assert params_lora["params"]["gating_einsum_lora_b"].shape == (2, 2, 32)
    assert params_lora["params"]["linear_lora_a"].shape == (32, 2)
    assert params_lora["params"]["linear_lora_b"].shape == (2, 8)


def test_lora_ffn_same_output():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    output = ffn.apply(params, x)

    params_lora = ffn_lora.init(key, x)
    output_lora = ffn_lora.apply(params_lora, x)

    assert jnp.allclose(output, output_lora)


def _grown_params(small_params, large_params):
    flat = lambda tree: flax.traverse_util.flatten_dict(tree, sep="/")  # noqa: E731
    grown = lora.expand_rank(flat(small_params), flat(large_params))
    return flax.traverse_util.unflatten_dict(grown, sep="/")


def test_expand_rank_keeps_einsum_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    loud = nn.initializers.normal(1.0)
    small = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, alpha=2.0, init_fn=loud))
    large = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=6, alpha=6.0, init_fn=loud))
    base = lora.Einsum(shape)

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    small_params = small.init(key, eqn, x)
    grown = _grown_params(small_params, large.init(jax.random.key(1), eqn, x))

    assert grown["params"]["lora_a"].shape == (3, 8, 32, 6)
    assert grown["params"]["lora_b"].shape == (3, 8, 6, 4)
    assert jnp.allclose(large.apply(grown, eqn, x), small.apply(small_params, eqn, x), atol=1e-5)
    # The LoRA path matters here, so the equality above is not the base model's output.
    assert not jnp.allclose(small.apply(small_params, eqn, x), base.apply({"params": {"w": small_params["params"]["w"]}}, eqn, x))


def test_expand_rank_keeps_ffn_output():
    loud = nn.initializers.normal(1.0)
    small = lora.FeedForward(features=8, hidden_dim=32, lora_config=lora.LoRAConfig(rank=2, alpha=2.0, init_fn=loud))
    large = lora.FeedForward(features=8, hidden_dim=32, lora_config=lora.LoRAConfig(rank=6, alpha=6.0, init_fn=loud))

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    small_params = small.init(key, x)
    grown = _grown_params(small_params, large.init(jax.random.key(1), x))

    assert grown["params"]["gating_einsum_lora_a"].shape == (2, 8, 6)
    assert grown["params"]["gating_einsum_lora_b"].shape == (2, 6, 32)
    assert grown["params"]["linear_lora_a"].shape == (32, 6)
    assert grown["params"]["linear_lora_b"].shape == (6, 8)
    assert jnp.allclose(large.apply(grown, x), small.apply(small_params, x), atol=1e-5)


def test_expand_rank_adds_random_columns_and_zero_rows_under_a_layer_axis():
    layers, heads, width, head_dim = 4, 8, 32, 16
    a = np.ones((layers, heads, width, 2), np.float32)
    b = np.ones((layers, heads, 2, head_dim), np.float32)
    loaded = {"llm/layers/attn/q_einsum_1/lora_a": a, "llm/layers/attn/q_einsum_1/lora_b": b, "llm/layers/attn/q_einsum_1/w": a}
    reference = {
        "llm/layers/attn/q_einsum_1/lora_a": jax.ShapeDtypeStruct((layers, heads, width, 6), np.float32),
        "llm/layers/attn/q_einsum_1/lora_b": jax.ShapeDtypeStruct((layers, heads, 6, head_dim), np.float32),
        "llm/layers/attn/q_einsum_1/w": jax.ShapeDtypeStruct(a.shape, np.float32),
    }

    grown = lora.expand_rank(loaded, reference, init_std=0.01)

    new_a, new_b = grown["llm/layers/attn/q_einsum_1/lora_a"], grown["llm/layers/attn/q_einsum_1/lora_b"]
    assert new_a.shape == (layers, heads, width, 6)
    assert new_b.shape == (layers, heads, 6, head_dim)
    assert np.array_equal(new_a[..., :2], a)
    assert np.array_equal(new_b[:, :, :2], b)
    assert 0.005 < new_a[..., 2:].std() < 0.02
    assert not new_b[:, :, 2:].any()
    assert grown["llm/layers/attn/q_einsum_1/w"] is loaded["llm/layers/attn/q_einsum_1/w"]


def test_expand_rank_keeps_dtype_and_leaves_equal_shapes_alone():
    loaded = {"x/lora_a": np.ones((3, 2), jnp.bfloat16), "x/lora_b": np.ones((2, 5), jnp.bfloat16), "x/w": np.ones((3, 5))}
    same = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in loaded.items()}
    assert all(lora.expand_rank(loaded, same)[k] is loaded[k] for k in loaded)

    bigger = {**same, "x/lora_a": jax.ShapeDtypeStruct((3, 4), jnp.bfloat16), "x/lora_b": jax.ShapeDtypeStruct((4, 5), jnp.bfloat16)}
    grown = lora.expand_rank(loaded, bigger)
    assert grown["x/lora_a"].dtype == jnp.bfloat16
    assert grown["x/lora_b"].dtype == jnp.bfloat16


@pytest.mark.parametrize(
    ("loaded_shape", "reference_shape"),
    [((3, 6), (3, 2)), ((3, 2), (4, 6)), ((3, 2), (1, 3, 6))],
    ids=["shrink", "other-axis-differs", "different-ndim"],
)
def test_expand_rank_refuses_shapes_it_cannot_grow(loaded_shape, reference_shape):
    with pytest.raises(ValueError, match="cannot grow"):
        lora.expand_rank({"x/lora_a": np.ones(loaded_shape)}, {"x/lora_a": jax.ShapeDtypeStruct(reference_shape, np.float32)})
