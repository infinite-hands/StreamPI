import flax.linen as nn
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

import openpi.models.lora as lora
import openpi.models.model as _model
from openpi.training import blend


def _trees():
    base = {"PaliGemma/img/w": np.full((3,), 1.0, np.float32), "PaliGemma/llm/w": np.full((3,), 2.0, np.float32)}
    tuned = {
        "PaliGemma/img/w": np.full((3,), 3.0, np.float32),
        "PaliGemma/llm/w": np.full((3,), 2.0, jnp.bfloat16),
        "PaliGemma/llm/q_einsum_1/lora_a": np.full((2, 4), 0.5, np.float32),
        "PaliGemma/llm/q_einsum_1/lora_b": np.full((4, 2), 0.5, np.float32),
        "state_proj/kernel": np.full((2,), 7.0, np.float32),
    }
    return base, tuned


def test_blend_mixes_shared_weights_scales_lora_b_and_keeps_the_rest():
    base, tuned = _trees()
    half = blend.blend_params(base, tuned, 0.5)
    assert np.allclose(half["PaliGemma/img/w"], 2.0)
    assert half["PaliGemma/llm/w"].dtype == jnp.bfloat16
    assert np.allclose(np.asarray(half["PaliGemma/llm/w"], np.float32), 2.0)
    assert np.allclose(half["PaliGemma/llm/q_einsum_1/lora_b"], 0.25)
    assert np.array_equal(half["PaliGemma/llm/q_einsum_1/lora_a"], tuned["PaliGemma/llm/q_einsum_1/lora_a"])
    assert half["state_proj/kernel"] is tuned["state_proj/kernel"]

    assert np.allclose(blend.blend_params(base, tuned, 1.0)["PaliGemma/img/w"], 3.0)
    zero = blend.blend_params(base, tuned, 0.0)
    assert np.allclose(zero["PaliGemma/img/w"], 1.0)
    assert not np.asarray(zero["PaliGemma/llm/q_einsum_1/lora_b"]).any()
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend.blend_params(base, tuned, 1.5)


def test_scaling_lora_b_scales_the_lora_contribution_linearly():
    shape = (3, 8, 32, 4)
    loud = nn.initializers.normal(1.0)
    module = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, alpha=2.0, init_fn=loud))
    eqn = "BSD,3KDH->3BSKH"
    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))
    tuned = module.init(key, eqn, x)
    flat = flax.traverse_util.flatten_dict(tuned, sep="/")
    base_flat = {name: value for name, value in flat.items() if not name.endswith(("lora_a", "lora_b"))}

    def output(alpha):
        blended = blend.blend_params(base_flat, flat, alpha)
        return module.apply(flax.traverse_util.unflatten_dict(blended, sep="/"), eqn, x)

    zero, half, one = output(0.0), output(0.5), output(1.0)
    assert not np.allclose(np.asarray(zero), np.asarray(one))
    assert np.allclose(np.asarray(half), 0.5 * np.asarray(zero) + 0.5 * np.asarray(one), atol=1e-4)


def test_a_blended_checkpoint_loads_like_any_other_and_carries_its_assets(tmp_path):
    base, tuned = _trees()
    for name, tree in (("base", base), ("tuned", tuned)):
        directory = tmp_path / name
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(directory / "params", {"params": flax.traverse_util.unflatten_dict(tree, sep="/")})
    (tmp_path / "tuned" / "assets" / "repo").mkdir(parents=True)
    (tmp_path / "tuned" / "assets" / "repo" / "norm_stats.json").write_text("{}")

    out = blend.write_blended_checkpoint(str(tmp_path / "base" / "params"), str(tmp_path / "tuned" / "params"), 0.5, str(tmp_path / "out"))

    loaded = flax.traverse_util.flatten_dict(_model.restore_params(out / "params", restore_type=np.ndarray), sep="/")
    assert np.allclose(loaded["PaliGemma/img/w"], 2.0)
    assert np.allclose(loaded["PaliGemma/llm/q_einsum_1/lora_b"], 0.25)
    assert (out / "assets" / "repo" / "norm_stats.json").is_file()
