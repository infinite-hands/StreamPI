"""lit=off must be the stock model: exact equality against the fixture generated from the unmodified pin 96d891f2.

The fixture (lit_golden/baseline_*) holds compute_loss outputs, sample_actions outputs with an empty and with a carried
memory, the memory arrays' fingerprints and the param pytree's fingerprint, all produced by lit_golden/gen_baseline.py.
Every check recomputes with the same generator code on the current source and compares bit for bit, so a LIT change
that perturbs the stock path (rng stream, param init, attention, memory contract) fails here. Do not regenerate the
fixture from a LIT head: the generator refuses to, and a regenerated fixture would prove nothing.

Exact equality holds on the toolchain the fixture was made on (see baseline_meta.json); test_toolchain says so first
when it differs.

The whole file takes five to six minutes on CPU (the 27-layer SigLIP dominates). For a quicker gate (about 80 seconds)
during development run `-k "fixture or toolchain or lit_off or param_tree or f32_rand_jit"`, then the full file before
handing off.
"""

import functools
import json

import numpy as np
import pytest

from openpi.models import lit_test_utils as _utils
from openpi.models import pi0_config
from openpi.models.lit_golden import gen_baseline as _gen


@functools.cache
def _fixture(name: str):
    path = _gen.FIXTURE_DIR / name
    if name.endswith(".npz"):
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text())


@functools.cache
def _loss(case: str):
    return _gen.run_loss(case)


@functools.cache
def _sample(case: str):
    return _gen.run_sample(case)


def test_fixture_was_generated_from_the_pin():
    meta = _fixture("baseline_meta.json")
    assert meta["git"]["pin"] == _gen.PIN
    assert meta["git"]["head"] == _gen.PIN
    assert meta["git"]["files_differing_from_pin"] == []


def test_fixture_matches_the_generator_definition():
    meta = _fixture("baseline_meta.json")
    current = _gen.meta()
    for key in ("seeds", "num_steps", "mask_num_per_loss_seed", "cases", "config"):
        assert meta[key] == current[key], f"{key} changed since the fixture was generated"
    assert sorted(meta["mask_num_per_loss_seed"].values()) == list(range(_utils.HIST_HORIZON)), (
        "every mask_num is covered"
    )


def test_toolchain():
    recorded, current = _fixture("baseline_meta.json")["toolchain"], _gen.meta()["toolchain"]
    keys = ("jax", "flax", "numpy", "machine", "jax_platform")
    assert {k: recorded[k] for k in keys} == {k: current[k] for k in keys}, "exact equality needs the same toolchain"


def test_stock_config_defaults_to_lit_off():
    config = pi0_config.Pi0Config()
    assert getattr(config, "lit", "off") == "off"
    assert getattr(_utils.make_tiny_config(), "lit", "off") == "off"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_param_tree_is_unchanged(dtype):
    expected, got = _fixture("baseline_params.json")[dtype], _utils.param_manifest(_gen.get_model(dtype, "raw"))
    assert sorted(got) == sorted(expected), (
        f"added {sorted(set(got) - set(expected))}, removed {sorted(set(expected) - set(got))}"
    )
    for path, want in expected.items():
        have = got[path]
        assert (have["type"], have["shape"], have["dtype"]) == (want["type"], want["shape"], want["dtype"]), path
    changed = [path for path in expected if got[path]["sha256"] != expected[path]["sha256"]]
    assert not changed, f"init values changed (rng stream moved?) for {changed}"


@pytest.mark.parametrize("case", list(_gen.CASES))
def test_compute_loss_is_bit_identical(case):
    expected = {k: v for k, v in _fixture("baseline_loss.npz").items() if k.startswith(f"{case}__")}
    got = _loss(case)
    assert expected
    assert sorted(got) == sorted(expected)
    for key, want in expected.items():
        assert (got[key].dtype, got[key].shape) == (want.dtype, want.shape), key
        np.testing.assert_array_equal(got[key], want, err_msg=key)


@pytest.mark.parametrize("case", list(_gen.CASES))
def test_sample_actions_and_memory_are_bit_identical(case):
    expected = {k: v for k, v in _fixture("baseline_actions.npz").items() if k.startswith(f"{case}__")}
    actions, memory = _sample(case)
    assert expected
    assert sorted(actions) == sorted(expected)
    for key, want in expected.items():
        assert (actions[key].dtype, actions[key].shape) == (want.dtype, want.shape), key
        np.testing.assert_array_equal(actions[key], want, err_msg=key)
    # The memory contract: exactly these keys, and (shape, dtype, sha256) of every array, after call 1 and call 2.
    expected_memory = _fixture("baseline_memory.json")[case]
    assert sorted(memory) == sorted(expected_memory) == ["call1", "call2"]
    for call in expected_memory:
        assert sorted(memory[call]) == ["memory_kv_cache", "memory_prefix_mask", "memory_tokens"]
        assert memory[call] == expected_memory[call], call
