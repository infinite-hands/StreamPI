"""Titans' long-term memory on its own (arXiv 2501.00663): what a write does to the weights, what a read returns, and
that training's unroll and the served call-by-call writes are one computation. Small widths, float32, CPU."""

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import titans
from openpi.shared import nnx_utils

TOKEN_WIDTH, DIM, HIDDEN, GRID, PATCHES, BATCH = 48, 32, 64, 2, 16, 3   # 4 x 4 patches pooled to 2 x 2 cells
CLOSED, OPEN = -30.0, 30.0   # gate logits whose sigmoid is 0 and 1 in float32


def _memory(depth: int = 2, seed: int = 0) -> titans.TitansMemory:
    return titans.TitansMemory(TOKEN_WIDTH, DIM, HIDDEN, depth, GRID, nnx.Rngs(seed))


def _frames(memory: titans.TitansMemory, count: int, seed: int = 0, batch: int = BATCH):
    """`count` frames of random image tokens per sample, as the memory's input: (batch, count, cells, DIM)."""
    rng = np.random.default_rng(seed)
    return memory.tokens(jnp.asarray(rng.normal(size=(batch, count, PATCHES, TOKEN_WIDTH)), jnp.float32))


def _set_gates(memory: titans.TitansMemory, *, theta: float | None = None, eta: float | None = None,
               alpha: float | None = None) -> None:
    bias = np.asarray(memory.gate_bias.value).copy()
    for index, logit in enumerate((theta, eta, alpha)):
        if logit is not None:
            bias[index] = logit
    memory.gate_bias.value = jnp.asarray(bias)


def _norm(layers) -> np.ndarray:
    """The global norm of a state's per-layer arrays, per sample."""
    return np.sqrt(sum(np.sum(np.square(np.asarray(layer)), axis=(-2, -1)) for layer in layers))


def _same(first, second, atol: float = 0.0) -> bool:
    leaves = zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    return all(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=0.0) for a, b in leaves)


def test_grid_pool_averages_each_cell_of_the_picture():
    patches = jnp.arange(16, dtype=jnp.float32).reshape(1, 16, 1)   # a 4 x 4 picture, row-major, value = its index
    cells = titans.grid_pool(patches, 2)
    # Top-left cell holds patches 0, 1, 4, 5; top-right 2, 3, 6, 7; bottom-left 8, 9, 12, 13; bottom-right the rest.
    np.testing.assert_allclose(np.asarray(cells)[0, :, 0], [2.5, 4.5, 10.5, 12.5])
    np.testing.assert_allclose(np.asarray(titans.grid_pool(patches, 4))[0, :, 0], np.arange(16))


def test_grid_pool_refuses_tokens_its_cells_do_not_divide():
    with pytest.raises(ValueError, match="not a square grid"):
        titans.grid_pool(jnp.zeros((1, 15, 4)), 2)
    with pytest.raises(ValueError, match="not a square grid"):
        titans.grid_pool(jnp.zeros((1, 16, 4)), 3)


def test_a_frame_counts_its_tokens_once_whether_their_keys_differ_or_agree():
    """Orthogonal keys count 1 each (the paper's sum over a chunk); identical keys share 1 (the mean); keys near zero
    count at most 1, so they cannot magnify a write."""
    orthogonal = jnp.eye(4, 8)
    np.testing.assert_allclose(np.asarray(titans.token_weights(orthogonal)), 1.0)
    identical = jnp.tile(titans._unit(jnp.arange(1.0, 9.0))[None], (4, 1))
    np.testing.assert_allclose(np.asarray(titans.token_weights(identical)), 0.25, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(titans.token_weights(1e-4 * orthogonal)), 1.0)
    pair = jnp.asarray([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])   # unit keys, the middle one between the others
    np.testing.assert_allclose(np.asarray(titans.token_weights(pair)), [1 / 1.6, 1 / 2.4, 1 / 1.8], rtol=1e-6)


def test_the_query_starts_as_the_key_so_a_read_recalls_what_was_written():
    memory = _memory()
    np.testing.assert_array_equal(np.asarray(memory.query_proj.value), np.asarray(memory.key_proj.value))
    assert memory.query_proj is not memory.key_proj, "a copy, trained on its own"
    frame = _frames(memory, 1)[:, 0]
    _keys, values = memory._pairs(frame)

    def agreement(state):
        tokens, _ = memory.read(state, frame)
        return np.asarray(jnp.mean(jnp.sum(titans._unit(tokens) * values, axis=-1), axis=-1))

    written = memory.write(memory.initial_state(BATCH), frame, jnp.ones((BATCH,), bool))
    assert (agreement(written) > agreement(memory.initial_state(BATCH)) + 0.05).all()


def test_the_memory_is_given_unit_rms_tokens_one_per_cell():
    x = _frames(_memory(), 2)
    assert x.shape == (BATCH, 2, GRID * GRID, DIM)
    assert x.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(jnp.mean(jnp.square(x), axis=-1)), 1.0, atol=1e-4)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_write_makes_the_memory_recall_the_frame_better(depth):
    memory = _memory(depth)
    frame = _frames(memory, 1)[:, 0]
    empty = memory.initial_state(BATCH)
    once = memory.write(empty, frame, jnp.ones((BATCH,), bool))
    again = once
    for _ in range(10):
        again = memory.write(again, frame, jnp.ones((BATCH,), bool))
    before, after_one, after_many = (np.asarray(memory.surprise(state, frame)) for state in (empty, once, again))
    assert (after_one < before).all()
    assert (after_many < after_one).all()
    # It recalls THIS frame better, not everything: an unseen frame gains less.
    other = _frames(memory, 1, seed=5)[:, 0]
    gain_other = np.asarray(memory.surprise(empty, other)) - np.asarray(memory.surprise(again, other))
    assert (gain_other < before - after_many).all()


def test_a_masked_write_changes_nothing():
    memory = _memory()
    frames = _frames(memory, 2)
    state = memory.write(memory.initial_state(BATCH), frames[:, 0], jnp.ones((BATCH,), bool))
    valid = jnp.asarray([True, False, True])
    written = memory.write(state, frames[:, 1], valid)
    for name in ("deviation", "momentum"):
        for old, new in zip(state[name], written[name], strict=True):
            assert np.array_equal(np.asarray(old)[1], np.asarray(new)[1])
            assert not np.array_equal(np.asarray(old)[0], np.asarray(new)[0])
            assert not np.array_equal(np.asarray(old)[2], np.asarray(new)[2])
    # A sample that has never had a valid write is still empty.
    never = memory.write(memory.initial_state(BATCH), frames[:, 0], valid)
    assert np.asarray(never["written"]).tolist() == [True, False, True]
    assert np.asarray(written["written"]).all()


@pytest.mark.parametrize("depth", [1, 2])
def test_the_unroll_is_the_same_writes_made_one_call_at_a_time(depth):
    """Training unrolls the frames in one scan; serving writes one frame per call. They must agree."""
    memory = _memory(depth)
    frames = _frames(memory, 6)
    valid = jnp.ones((BATCH, 6), bool).at[1, :3].set(False).at[2].set(False)   # one late start, one never written
    unrolled = memory.unroll(frames, valid)
    served = memory.initial_state(BATCH)
    for index in range(6):
        served = memory.write(served, frames[:, index], valid[:, index])
    assert _same(unrolled, served, atol=1e-6)
    assert np.asarray(unrolled["written"]).tolist() == [True, True, False]
    now = _frames(memory, 1, seed=9)[:, 0]
    assert _same(memory.read(unrolled, now), memory.read(served, now), atol=1e-6)


def test_an_empty_memory_reads_as_nothing_and_a_written_one_counts():
    memory = _memory()
    frames = _frames(memory, 2)
    tokens, mask = memory.read(memory.initial_state(BATCH), frames[:, 1])
    assert tokens.shape == (BATCH, GRID * GRID, DIM)
    assert mask.shape == (BATCH, GRID * GRID)
    assert not np.asarray(mask).any()
    state = memory.write(memory.initial_state(BATCH), frames[:, 0], jnp.asarray([True, False, True]))
    tokens, mask = memory.read(state, frames[:, 1])
    assert np.asarray(mask).all(axis=1).tolist() == [True, False, True]
    np.testing.assert_allclose(np.asarray(jnp.mean(jnp.square(tokens), axis=-1)), 1.0, atol=1e-3)


def test_what_is_read_depends_on_what_was_written():
    memory = _memory()
    now = _frames(memory, 1, seed=9)[:, 0]
    everything = jnp.ones((BATCH, 4), bool)
    first, _ = memory.read(memory.unroll(_frames(memory, 4, seed=1), everything), now)
    second, _ = memory.read(memory.unroll(_frames(memory, 4, seed=2), everything), now)
    unwritten, _ = memory.read(memory.initial_state(BATCH), now)
    assert not np.allclose(np.asarray(first), np.asarray(second), atol=1e-3)
    assert not np.allclose(np.asarray(first), np.asarray(unwritten), atol=1e-3)


def test_the_momentum_carries_the_previous_surprise():
    """Eq. 14: S_t = eta S_{t-1} - (this frame's surprise). The frame's own term is what a write from the same weights
    with no momentum leaves, so S_t - eta S_{t-1} must equal that."""
    memory = _memory()
    frames = _frames(memory, 2)
    everything = jnp.ones((BATCH,), bool)
    first = memory.write(memory.initial_state(BATCH), frames[:, 0], everything)
    second = memory.write(first, frames[:, 1], everything)
    still = {**first, "momentum": tuple(jnp.zeros_like(layer) for layer in first["momentum"])}
    alone = memory.write(still, frames[:, 1], everything)
    eta = float(jax.nn.sigmoid(memory.gate_bias.value[1]))
    assert 0.0 < eta < 1.0
    for carried, previous, own in zip(second["momentum"], first["momentum"], alone["momentum"], strict=True):
        np.testing.assert_allclose(np.asarray(carried), eta * np.asarray(previous) + np.asarray(own), atol=1e-6)

    _set_gates(memory, eta=CLOSED)
    forgetful = memory.write(first, frames[:, 1], everything)
    assert _same(forgetful["momentum"], alone["momentum"], atol=1e-6)


def test_forgetting_decays_what_the_run_wrote_not_the_starting_weights():
    """Eq. 13 on the deviation: D_t = (1 - alpha) D_{t-1} + S_t."""
    memory = _memory()
    frames = _frames(memory, 2)
    everything = jnp.ones((BATCH,), bool)
    first = memory.write(memory.initial_state(BATCH), frames[:, 0], everything)
    alpha = float(jax.nn.sigmoid(memory.gate_bias.value[2]))
    assert 0.0 < alpha < 1.0
    second = memory.write(first, frames[:, 1], everything)
    for moved, previous, step in zip(second["deviation"], first["deviation"], second["momentum"], strict=True):
        np.testing.assert_allclose(np.asarray(moved), (1.0 - alpha) * np.asarray(previous) + np.asarray(step),
                                   atol=1e-6)

    # With the gate fully open the run's earlier writes are gone: only the last step is left.
    _set_gates(memory, alpha=OPEN)
    forgotten = memory.write(first, frames[:, 1], everything)
    assert _same(forgotten["deviation"], forgotten["momentum"], atol=1e-6)
    # And with nothing left to add, the memory is back at its starting weights, where a read is the unwritten one.
    _set_gates(memory, theta=CLOSED, eta=CLOSED)
    reset = memory.write(first, frames[:, 1], everything)
    assert _norm(reset["deviation"]).max() < 1e-6
    now = _frames(memory, 1, seed=9)[:, 0]
    np.testing.assert_allclose(np.asarray(memory.read(reset, now)[0]),
                               np.asarray(memory.read(memory.initial_state(BATCH), now)[0]), atol=1e-5)


@pytest.mark.parametrize("depth", [1, 2])
def test_one_write_is_bounded_whatever_the_weights_have_become(depth):
    memory = _memory(depth)
    for name in memory.layers:
        memory.initial_weights[name].value = 1e3 * memory.initial_weights[name].value
    frame = _frames(memory, 1)[:, 0]
    state = memory.write(memory.initial_state(BATCH), frame, jnp.ones((BATCH,), bool))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state))
    assert (_norm(state["momentum"]) <= titans.MAX_SURPRISE_NORM * (1 + 1e-5)).all()
    tokens, mask = memory.read(state, frame)
    assert np.isfinite(np.asarray(tokens)).all()
    assert np.asarray(mask).all()


def test_a_long_run_stays_inside_the_bound_its_gates_give():
    """With constant gates |S| <= C / (1 - eta) and |D| <= C / ((1 - eta) alpha), C the surprise clip: a served run
    writes far more frames than training ever unrolls, and its state cannot grow past this."""
    memory = _memory()
    writes = 2000
    frames = _frames(memory, writes, batch=2)
    state = nnx_utils.module_jit(memory.unroll)(frames, jnp.ones((2, writes), bool))
    eta, alpha = (float(jax.nn.sigmoid(memory.gate_bias.value[index])) for index in (1, 2))
    momentum_bound = titans.MAX_SURPRISE_NORM / (1.0 - eta)
    assert (_norm(state["momentum"]) <= momentum_bound * (1 + 1e-4)).all()
    assert (_norm(state["deviation"]) <= momentum_bound / alpha * (1 + 1e-4)).all()
    tokens, mask = memory.read(state, frames[:, -1])
    assert np.isfinite(np.asarray(tokens)).all()
    assert np.asarray(mask).all()


def test_a_memory_that_is_not_finite_reads_as_an_empty_one():
    memory = _memory()
    frames = _frames(memory, 2)
    state = memory.write(memory.initial_state(BATCH), frames[:, 0], jnp.ones((BATCH,), bool))
    broken = {**state, "deviation": tuple(layer.at[1].set(jnp.inf) for layer in state["deviation"])}
    tokens, mask = memory.read(broken, frames[:, 1])
    assert np.isfinite(np.asarray(tokens)).all()
    assert np.asarray(mask).all(axis=1).tolist() == [True, False, True]
    assert not np.asarray(tokens)[1].any()


@pytest.mark.parametrize("depth", [1, 2])
def test_training_reaches_every_parameter_through_the_writes(depth):
    """The outer loss is on the READ, after the unroll: its gradient must be finite and reach the projections, the
    gates and the starting weights through the inner gradient steps, also with a sample whose memory stays empty."""
    memory = _memory(depth)
    rng = np.random.default_rng(3)
    older = jnp.asarray(rng.normal(size=(BATCH, 5, PATCHES, TOKEN_WIDTH)), jnp.float32)
    current = jnp.asarray(rng.normal(size=(BATCH, PATCHES, TOKEN_WIDTH)), jnp.float32)
    target = jnp.asarray(rng.normal(size=(BATCH, GRID * GRID, DIM)), jnp.float32)
    valid = jnp.ones((BATCH, 5), bool).at[1, :2].set(False).at[2].set(False)

    def loss(model):
        tokens, mask = model.read(model.unroll(model.tokens(older), valid), model.tokens(current))
        return jnp.mean(jnp.square(tokens - target) * mask[..., None])

    gradients = traverse_util.flatten_dict(nnx.grad(loss)(memory).to_pure_dict(), sep="/")
    assert set(gradients) == set(traverse_util.flatten_dict(nnx.state(memory, nnx.Param).to_pure_dict(), sep="/"))
    for name, gradient in gradients.items():
        assert np.isfinite(np.asarray(gradient)).all(), name
        assert np.abs(np.asarray(gradient)).max() > 0, name


def test_gates_that_read_the_frame_change_the_write():
    """The gates' kernels start at zero (constant gates); once trained they depend on the frame."""
    memory = _memory()
    frames = _frames(memory, 2)
    everything = jnp.ones((BATCH,), bool)
    constant = memory.write(memory.initial_state(BATCH), frames[:, 0], everything)
    memory.gate_kernel.value = jnp.asarray(
        np.random.default_rng(4).normal(size=memory.gate_kernel.value.shape), jnp.float32)
    dependent = memory.write(memory.initial_state(BATCH), frames[:, 0], everything)
    assert not _same(constant["deviation"], dependent["deviation"], atol=1e-6)


def test_a_memory_needs_a_layer_and_positive_sizes():
    with pytest.raises(ValueError, match="depth >= 1"):
        titans.TitansMemory(TOKEN_WIDTH, DIM, HIDDEN, 0, GRID, nnx.Rngs(0))
    with pytest.raises(ValueError, match="depth >= 1"):
        titans.TitansMemory(TOKEN_WIDTH, 0, HIDDEN, 2, GRID, nnx.Rngs(0))
