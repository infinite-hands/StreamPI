import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np
import pytest

import openpi.models.lora as lora
import openpi.models.model as _model
import openpi.shared.download as download
import openpi.training.config as config
import openpi.training.weight_loaders as weight_loaders


def _tree(rank: int, make):
    block = {
        "gating_einsum_lora_a": make((2, 8, rank)),
        "gating_einsum_lora_b": make((2, rank, 32)),
        "linear": make((32, 8)),
    }
    return {"PaliGemma": {"llm": {"layers": {"mlp_1": block}}}}


@pytest.fixture
def checkpoint(monkeypatch):
    saved = _tree(2, lambda shape: np.ones(shape, np.float32))
    monkeypatch.setattr(_model, "restore_params", lambda *args, **kwargs: saved)
    monkeypatch.setattr(download, "maybe_download", lambda path, **kwargs: path)
    return saved


def test_loader_grows_lora_rank_only_when_asked(checkpoint):
    reference = _tree(6, lambda shape: jax.ShapeDtypeStruct(shape, np.float32))

    grown = weight_loaders.CheckpointWeightLoader("ignored", expand_lora_rank=True).load(reference)
    assert jax.tree.map(lambda leaf: leaf.shape, grown) == jax.tree.map(lambda leaf: leaf.shape, reference)

    kept = weight_loaders.CheckpointWeightLoader("ignored").load(reference)
    assert kept["PaliGemma"]["llm"]["layers"]["mlp_1"]["gating_einsum_lora_a"].shape == (2, 8, 2)


def _real_param_shapes(config_name: str):
    model_config = config.get_config(config_name).model
    return jax.eval_shape(lambda rng: nnx.state(model_config.create(rng)).to_pure_dict(), jax.random.key(0))


def _flat(tree):
    return flax.traverse_util.flatten_dict(tree, sep="/")


def _lora_size(shapes, *, expert: bool) -> int:
    return sum(int(np.prod(leaf.shape)) for key, leaf in shapes.items() if "lora" in key and ("_1/" in key) == expert)


def test_rank_32_checkpoint_grows_into_the_real_rank_128_expert():
    small = _flat(_real_param_shapes("pi05_yam_stream5_i20_bagging_right_real"))
    large = _flat(_real_param_shapes("pi05_yam_stream5_i20_bagging_right_real_xr128"))
    # Zero-copy stand-ins for a checkpoint: only the adapters that grow are materialised.
    saved = {key: np.broadcast_to(np.zeros((), np.float32), leaf.shape) for key, leaf in small.items()}

    grown = lora.expand_rank(saved, large)

    assert grown.keys() == large.keys()
    assert all(grown[key].shape == large[key].shape for key in large)
    changed = {key for key in large if small[key].shape != large[key].shape}
    assert changed
    assert all("lora" in key and "_1/" in key for key in changed)
    assert _lora_size(small, expert=True) == 22_118_400
    assert _lora_size(large, expert=True) == 88_473_600
    assert _lora_size(small, expert=False) == _lora_size(large, expert=False) == 27_869_184
