"""The action expert's memory-read branch: zero-gated, ignores masked tokens, content-based, leaves the VLM alone."""

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_testing
import openpi.models.gemma as gemma

WIDTH, BANK_WIDTH, PREFIX, SUFFIX = 64, 24, 3, 2


def _llm(dtype="float32"):
    configs = [gemma.get_config("dummy"), gemma.get_config("dummy")]
    llm = nnx_bridge.ToNNX(gemma.Module(configs=configs, embed_dtype=dtype, adarms=True))
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, True], memory_width=BANK_WIDTH)
    return pi0_testing.awaken(llm, jax.random.key(7))


def _set_gate(llm, value):
    state = nnx.state(llm, nnx.Param)
    flat = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    gates = [key for key in flat if key.endswith("memory_gate")]
    assert len(gates) == 1
    flat[gates[0]] = jnp.full(flat[gates[0]].shape, value, flat[gates[0]].dtype)
    state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(flat, sep="/"))
    nnx.update(llm, state)


def _run(llm, bank=None, mask=None):
    rng = np.random.default_rng(0)
    prefix = jnp.asarray(rng.normal(size=(1, PREFIX, WIDTH)), jnp.float32)
    suffix = jnp.asarray(rng.normal(size=(1, SUFFIX, WIDTH)), jnp.float32)
    cond = jnp.asarray(rng.normal(size=(1, SUFFIX, WIDTH)), jnp.float32)
    total = PREFIX + SUFFIX
    attention = jnp.tril(jnp.ones((total, total), dtype=bool))[None]
    outputs, _ = llm([prefix, suffix], mask=attention, positions=jnp.arange(total)[None], adarms_cond=[None, cond],
                     memory_bank=None if bank is None else (bank, mask))
    return [np.asarray(out) for out in outputs]


def _bank(seed=3, tokens=4):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(1, tokens, BANK_WIDTH)), jnp.float32)


def test_a_closed_gate_leaves_the_model_exactly_as_it_was():
    llm = _llm()
    plain = _run(llm)
    closed = _run(llm, _bank(), jnp.ones((1, 4), bool))
    assert all(np.array_equal(a, b) for a, b in zip(plain, closed, strict=True))


def test_an_open_gate_changes_only_the_expert():
    llm = _llm()
    plain = _run(llm)
    _set_gate(llm, 1.0)
    opened = _run(llm, _bank(), jnp.ones((1, 4), bool))
    assert np.array_equal(plain[0], opened[0])
    assert not np.allclose(plain[1], opened[1], atol=1e-5)


def test_a_gate_that_is_only_slightly_open_already_moves_the_output():
    llm = _llm()
    plain = _run(llm)
    _set_gate(llm, 1e-2)
    nudged = _run(llm, _bank(), jnp.ones((1, 4), bool))
    assert not np.allclose(plain[1], nudged[1], atol=1e-8)


def test_masked_tokens_add_nothing_and_their_contents_do_not_matter():
    llm = _llm()
    plain = _run(llm)
    _set_gate(llm, 1.0)
    nothing_valid = _run(llm, _bank(), jnp.zeros((1, 4), bool))
    assert all(np.array_equal(a, b) for a, b in zip(plain, nothing_valid, strict=True))

    mask = jnp.asarray([[True, True, False, False]])
    first = _run(llm, _bank(seed=3), mask)
    other = np.asarray(_bank(seed=3)).copy()
    other[:, 2:] = np.random.default_rng(9).normal(size=(1, 2, BANK_WIDTH)) * 5
    second = _run(llm, jnp.asarray(other), mask)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(first, second, strict=True))
    assert not np.allclose(first[1], plain[1], atol=1e-5)


def test_the_read_is_content_based_so_the_bank_order_does_not_matter():
    llm = _llm()
    _set_gate(llm, 1.0)
    bank = _bank()
    mask = jnp.ones((1, 4), bool)
    ordered = _run(llm, bank, mask)
    shuffled = _run(llm, bank[:, jnp.asarray([2, 0, 3, 1])], mask)
    assert all(np.allclose(a, b, atol=1e-5) for a, b in zip(ordered, shuffled, strict=True))
    different = _run(llm, _bank(seed=11), mask)
    assert not np.allclose(ordered[1], different[1], atol=1e-5)


def test_a_bf16_model_stays_bf16_with_the_read_open():
    mask = jnp.ones((1, 4), bool)
    reference = _llm()
    _set_gate(reference, 1.0)
    expected = _run(reference, _bank(), mask)

    llm = _llm("bfloat16")
    closed = _run(llm, _bank(), mask)
    _set_gate(llm, 1.0)
    opened = _run(llm, _bank(), mask)  # the bank is float32, as the model's pooling produces it

    assert all(out.dtype == jnp.bfloat16 for out in opened)
    assert not np.allclose(opened[1].astype(np.float32), closed[1].astype(np.float32), atol=0.1)
    for got, want in zip(opened, expected, strict=True):
        np.testing.assert_allclose(got.astype(np.float32), want, atol=0.15, rtol=0.1)
