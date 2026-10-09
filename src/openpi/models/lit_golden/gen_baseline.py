"""Generates the lit=off golden fixture from the UNMODIFIED StreamPI pin (96d891f2).

    PYTHONPATH=src JAX_PLATFORMS=cpu python -m openpi.models.lit_golden.gen_baseline [--out DIR]

It records what the stock Pi0 (tiny dummy variants, real SigLIP; see lit_test_utils) computes, so that the LIT
changes can be proven not to touch lit=off:

    baseline_loss.npz     compute_loss outputs, key "<case>__s<rng seed>_train<0|1>"
    baseline_actions.npz  sample_actions outputs, key "<case>__call1_actions" (memory empty) / "__call2_actions"
                          (memory carried over from call 1, new observation)
    baseline_memory.json  shape/dtype/sum/sha256 of every array in the returned memory dict, per call
    baseline_params.json  the param pytree: path, type, shape, dtype, sum, sha256 of the values of every leaf
    baseline_meta.json    seeds, configs, mask_num per loss rng, toolchain versions, the git state it ran on

Every case is run on a randomized model ("rand": randomize_zero_init, so the prefix and images matter) and some also on
the raw init ("raw", prefix-blind), in eager and module_jit mode, float32 and bfloat16. The generator refuses to run
unless src/ equals the pin apart from lit_* files, so a regenerated fixture can only ever describe the stock model.
"""

import argparse
import functools
import json
import pathlib
import platform
import subprocess
import sys

import flax
import jax
import numpy as np

from openpi.models import lit_test_utils as _utils

PIN = "96d891f24db2bdbf6a832bbe7f35a2b79c9ce425"
FIXTURE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

MODEL_SEED = 0
RANDOMIZE_SEED = 0
ACTIONS_SEED = 1
OBSERVATION_SEEDS = (0, 1)  # sample_actions call 1 and call 2; compute_loss uses the first
NOISE_SEEDS = (0, 1)
SAMPLE_RNG_SEED = 2
NUM_STEPS = 3
# The mask_num compute_loss draws for these rngs is 0, 2, 1: the three values hist_horizon=3 allows.
LOSS_SEEDS = (0, 1, 2)

CONFIG_FIELDS = (
    "paligemma_variant",
    "action_expert_variant",
    "pi05",
    "action_dim",
    "action_horizon",
    "hist_horizon",
    "max_token_len",
)
# case name -> (dtype, "raw" | "rand", "eager" | "jit")
CASES = {
    "f32_rand_eager": ("float32", "rand", "eager"),
    "f32_rand_jit": ("float32", "rand", "jit"),
    "f32_raw_eager": ("float32", "raw", "eager"),
    "bf16_rand_eager": ("bfloat16", "rand", "eager"),
}
# case name -> [(rng seed, train)]
LOSS_RUNS = {
    "f32_rand_eager": [(s, t) for s in LOSS_SEEDS for t in (False, True)],
    "f32_rand_jit": [(0, False), (0, True)],
    "f32_raw_eager": [(0, False)],
    "bf16_rand_eager": [(0, False)],
}


@functools.cache
def get_model(dtype: str, params: str):
    model = _utils.build_model(_utils.make_tiny_config(dtype=dtype), MODEL_SEED)
    return _utils.randomize_zero_init(model, RANDOMIZE_SEED) if params == "rand" else model


def loss_key(case: str, seed: int, *, train: bool) -> str:
    return f"{case}__s{seed}_train{int(train)}"


def run_loss(case: str) -> dict[str, np.ndarray]:
    dtype, params, mode = CASES[case]
    config, model = _utils.make_tiny_config(dtype=dtype), get_model(dtype, params)
    observation = _utils.make_observation(OBSERVATION_SEEDS[0], config=config)
    actions = _utils.make_actions(ACTIONS_SEED, config=config)
    fn = _utils.jit_loss(model) if mode == "jit" else model.compute_loss
    return {
        loss_key(case, seed, train=train): np.asarray(fn(jax.random.key(seed), observation, actions, train=train))
        for seed, train in LOSS_RUNS[case]
    }


def run_sample(case: str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Two sample_actions calls on one memory dict. Returns the actions and, per call, the memory summary taken right
    after the call (eager calls mutate the dict, so the first summary must be taken before the second call)."""
    dtype, params, mode = CASES[case]
    config, model = _utils.make_tiny_config(dtype=dtype), get_model(dtype, params)
    fn = _utils.jit_sample(model) if mode == "jit" else model.sample_actions
    memory = _utils.empty_memory()
    actions, summaries = {}, {}
    for call, (obs_seed, noise_seed) in enumerate(zip(OBSERVATION_SEEDS, NOISE_SEEDS, strict=True), start=1):
        observation = _utils.make_observation(obs_seed, config=config)
        noise = _utils.fixed_noise(noise_seed, config=config)
        out, memory = fn(jax.random.key(SAMPLE_RNG_SEED), observation, num_steps=NUM_STEPS, noise=noise, memory=memory)
        actions[f"{case}__call{call}_actions"] = np.asarray(out)
        summaries[f"call{call}"] = _utils.memory_summary(memory)
    return actions, summaries


def param_manifests() -> dict[str, dict]:
    return {dtype: _utils.param_manifest(get_model(dtype, "raw")) for dtype in ("float32", "bfloat16")}


def loss_mask_nums() -> dict[str, int]:
    config = _utils.make_tiny_config()
    return {str(s): int(_utils.training_noise_and_time(jax.random.key(s), config=config)[2]) for s in LOSS_SEEDS}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout


def stock_source_state() -> dict:
    """The git state the fixture is generated on: files under src/ that differ from the pin, other than lit_*."""
    paths = _git("diff", "--name-only", PIN, "--", "src").split()
    paths += _git("ls-files", "--others", "--exclude-standard", "--", "src").split()
    changed = [
        path for path in paths if not pathlib.PurePosixPath(path).name.startswith("lit_") and "/lit_golden/" not in path
    ]
    return {"head": _git("rev-parse", "HEAD").strip(), "pin": PIN, "files_differing_from_pin": changed}


def toolchain() -> dict:
    """What the exact-equality fixture is a function of: the first five keys (lit_test_utils.fixture_exact_skip_reason)."""
    return {
        "jax": jax.__version__,
        "flax": flax.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "platform": platform.platform(),
        "jax_platform": jax.default_backend(),
    }


def meta(*, git: bool = True) -> dict:
    """The fixture's metadata. `git=False` leaves out the git state, which needs a checkout (a test that does not
    compare it must run without one)."""
    config = _utils.make_tiny_config()
    return {
        **({"git": stock_source_state()} if git else {}),
        "toolchain": toolchain(),
        "seeds": {
            "model": MODEL_SEED,
            "randomize": RANDOMIZE_SEED,
            "actions": ACTIONS_SEED,
            "observations": list(OBSERVATION_SEEDS),
            "noise": list(NOISE_SEEDS),
            "sample_rng": SAMPLE_RNG_SEED,
            "loss": list(LOSS_SEEDS),
        },
        "num_steps": NUM_STEPS,
        "mask_num_per_loss_seed": loss_mask_nums(),
        "cases": {name: {"dtype": d, "params": p, "mode": m} for name, (d, p, m) in CASES.items()},
        "config": {k: getattr(config, k) for k in CONFIG_FIELDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=FIXTURE_DIR)
    args = parser.parse_args()

    state = stock_source_state()
    if state["files_differing_from_pin"]:
        sys.exit(f"refusing to generate: src/ differs from the pin in {state['files_differing_from_pin']}")

    losses: dict[str, np.ndarray] = {}
    actions: dict[str, np.ndarray] = {}
    memory: dict[str, dict] = {}
    for case in CASES:
        losses.update(run_loss(case))
        case_actions, memory[case] = run_sample(case)
        actions.update(case_actions)
        print(f"{case}: done", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "baseline_loss.npz", **losses)
    np.savez(args.out / "baseline_actions.npz", **actions)
    for name, payload in (
        ("baseline_memory.json", memory),
        ("baseline_params.json", param_manifests()),
        ("baseline_meta.json", meta()),
    ):
        (args.out / name).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    assert all(np.isfinite(a).all() for a in (*losses.values(), *actions.values())), "non-finite baseline output"
    print(f"wrote {len(losses)} loss arrays, {len(actions)} action arrays to {args.out}")


if __name__ == "__main__":
    main()
