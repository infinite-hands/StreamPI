import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config
import openpi.training.optimizer as _optimizer


def _adam():
    return _optimizer.create_optimizer(_optimizer.AdamW(weight_decay=0.0), _optimizer.CosineDecaySchedule(warmup_steps=1))


def test_scale_updates_scales_only_the_matching_blocks():
    params = {
        "PaliGemma": {"img": {"w": jnp.ones(3)}, "llm": {"w": jnp.ones(3)}},
        "action_in_proj": {"kernel": jnp.ones(3)},
    }
    grads = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.5), params)

    plain, _ = _adam().update(grads, _adam().init(params), params)
    tx = _optimizer.scale_updates(_adam(), 2.5, _optimizer.VISION_PARAMS_REGEX)
    scaled, _ = tx.update(grads, tx.init(params), params)

    assert np.allclose(scaled["PaliGemma"]["img"]["w"], 2.5 * plain["PaliGemma"]["img"]["w"])
    assert np.allclose(scaled["PaliGemma"]["llm"]["w"], plain["PaliGemma"]["llm"]["w"])
    assert np.allclose(scaled["action_in_proj"]["kernel"], plain["action_in_proj"]["kernel"])
    assert np.abs(plain["PaliGemma"]["img"]["w"]).min() > 0


def test_vision_regex_labels_exactly_siglip_in_the_real_model():
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=14,
                                  action_horizon=4, max_token_len=16)
    params = nnx.state(nnx.eval_shape(lambda: config.create(jax.random.key(0))), nnx.Param)

    labels = _optimizer._path_labels(_optimizer.VISION_PARAMS_REGEX)(params)  # noqa: SLF001
    flat = jax.tree_util.tree_flatten_with_path(labels)[0]
    scaled = ["/".join(str(getattr(key, "key", getattr(key, "name", key))) for key in path) for path, label in flat if label == "scaled"]
    same = ["/".join(str(getattr(key, "key", getattr(key, "name", key))) for key in path) for path, label in flat if label == "same"]

    assert scaled
    assert all("PaliGemma/img/" in path for path in scaled)
    assert same
    assert not any("PaliGemma/img/" in path for path in same)
    assert any("PaliGemma/llm/" in path for path in same)
    assert any("action_in_proj" in path for path in same)


def test_scaled_optimizer_runs_on_the_trainers_nnx_state():
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_dim=14,
                                  action_horizon=4, max_token_len=16)
    params = nnx.state(nnx.eval_shape(lambda: config.create(jax.random.key(0))), nnx.Param)
    tx = _optimizer.scale_updates(_adam(), 2.0, _optimizer.VISION_PARAMS_REGEX)

    opt_state = jax.eval_shape(tx.init, params)
    updates, _ = jax.eval_shape(tx.update, params, opt_state, params)

    assert jax.tree.structure(updates) == jax.tree.structure(params)
