"""The memory bank beside the VLASH lead, on the sort-right recipe's one wrist camera: what the config refuses, and whether the
unpacked route (state_cond carries the lead; no packed branches) composes with the image memory and the state history.

Dummy gemma and DINOv2 variants (the SigLIP is the real one), CPU: this shows the pieces build, train a finite loss and reach the
output together. It does not show the combination learns anything on real data, or acts well on an arm."""

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_combo_test as combo

RIGHT_ARM = tuple(range(7, 14))
# Right-arm dims: the arm the sort-right recipe drives. state_cond carries the VLASH conditioning vector the same way for any offset.
RIGHT = {**combo.COMBO, "state_history_dims": RIGHT_ARM, "state_cond": True, "state_cond_dims": RIGHT_ARM}
CAMERA = "right_wrist_0_rgb"


def _config(**overrides):
    return combo._config(**{"image_keys": (CAMERA,), **overrides})


def test_the_config_refuses_packed_vlash_branches_beside_the_memory_bank():
    with pytest.raises(ValueError, match="memory bank is wired for plain pi0.5"):
        _config(**RIGHT, vlash_branches=4)


def test_the_config_accepts_state_conditioning_beside_the_memory_bank():
    config = _config(**RIGHT)
    assert config.state_cond and config.uses_memory_bank and not config.discrete_state_input


@pytest.fixture(scope="module")
def model():
    built = combo.pi0_testing.awaken(_config(**RIGHT).create(jax.random.key(0)), jax.random.key(7))
    yield built
    del built
    gc.collect()
    jax.clear_caches()


def _observation(model_input_state: np.ndarray | None = None) -> _model.Observation:
    observation = combo._training_observation()
    observation = observation.replace(images={CAMERA: observation.images[combo.CAMERA]},
                                      image_masks={CAMERA: observation.image_masks[combo.CAMERA]})
    return observation if model_input_state is None else observation.replace(state=jnp.asarray(model_input_state))


def test_the_combined_model_trains_a_finite_loss(model):
    for train in (False, True):
        loss = combo._loss(model, _observation(), train=train)
        assert loss.shape == (1, combo.HORIZON) and np.isfinite(loss).all()


def test_opening_the_memory_gate_changes_the_loss_with_state_conditioning_on(model):
    closed = combo._loss(model, _observation())
    opened = combo._loss(combo._open_gate(combo._copy(model)), _observation())
    assert not np.allclose(closed, opened), "the memory bank does not reach the output beside state_cond"


def test_the_conditioning_state_still_reaches_the_output_with_the_memory_bank_present(model):
    """The VLASH lead rides in the state vector (ConcatVlashCond), so the model must read it through state_cond. That path ends in
    the zero-initialised state_mlp_out, so the state is a no-op until it moves: move it, then show the output follows the state."""
    def move(pure):
        pure["state_mlp_out"]["kernel"] = 0.3 * jax.random.normal(jax.random.key(11), pure["state_mlp_out"]["kernel"].shape)

    moved = combo._edit(combo._copy(model), move)
    base = np.random.default_rng(2).normal(size=(1, combo.ACTION_DIM)).astype(np.float32)
    shifted = base.copy()
    shifted[:, list(RIGHT_ARM)] += 1.0
    assert not np.allclose(combo._loss(moved, _observation(base)), combo._loss(moved, _observation(shifted))), \
        "the conditioning state does not reach the output beside the memory bank"


def test_a_served_episode_gives_finite_chunks_with_state_conditioning_and_the_bank(model, monkeypatch):
    monkeypatch.setattr(combo, "CAMERA", CAMERA)   # the shared serving helper builds its requests for this camera
    chunks, fills = combo._serve(model, combo._fresh_memory())
    assert all(np.isfinite(chunk).all() for chunk in chunks)
    assert fills[-1][0] and fills[-1][1], "the serving rings never filled"
