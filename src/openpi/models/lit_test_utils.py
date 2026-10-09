"""Shared CPU harness for the LIT (Latent Interface Training) tests: a tiny StreamPI Pi0 and deterministic inputs.

The public API below is stable; later LIT tests import it unchanged.

    make_tiny_config(**overrides) -> Pi0Config
    make_observation(seed, *, config=None, batch=BATCH, masked=..., **fields) -> Observation
    make_actions(seed, *, config=None, batch=BATCH) -> Actions
    build_model(config, seed=0) -> Pi0
    randomize_zero_init(model, seed=0, scale=0.02) -> Pi0
    fixed_noise(seed, *, config=None, batch=BATCH)                 noise for sample_actions(noise=...)
    training_noise_and_time(rng, batch, config=None)               what compute_loss draws from `rng`
    empty_memory() / memory_summary(memory)                        the policy's memory dict, and a checksum view
    jit_loss(model) / jit_sample(model)                            module_jit wrappers, as Policy uses them
    param_manifest(model) / array_digest(x)                        path -> shape, dtype, sha256 of the values
    stub_image_encoder()                                           opt-in one-Dense stand-in for SigLIP (see below)
    fixture_exact_skip_reason(recorded_toolchain, current_toolchain)   why an exact-equality fixture test cannot run

What the tiny model is. Both Gemma experts use the "dummy" variant (width 64, depth 4, 1 kv head of head_dim 16),
pi05=True, float32, action_dim 14, action_horizon 4, hist_horizon 3, max_token_len 8. The image tower is the REAL
SigLIP So400m/14: Pi0.__init__ hard-codes it and the config offers no smaller variant, and it is cheap enough
(measured on this CPU: model creation ~4 s, one compute_loss or sample_actions call at batch 2 ~5-10 s eager), so the
default is the real encoder. Images must be 224x224 (other sizes are resized inside preprocess_observation), giving
256 image
tokens per camera. With 3 cameras and max_token_len 8 one history block is 3*256 + 8 = 776 tokens, 3 blocks = 2328.

Two init facts the tests depend on. A freshly created pi0.5 is prefix-blind: every adaRMS modulation kernel is
zero-initialised, so the action expert's attention contributes nothing. And the SigLIP head (head_zeroinit) is
zero-initialised, so every image token is exactly zero. Call randomize_zero_init(model, seed) to get a model that
actually reads the prefix and the images; tests of masking, gradients or image/language dependence need it.

Stub image encoder, OFF by default. `stub_image_encoder()` swaps SigLIP for one Dense layer over 14x14 patch means for
the duration of the `with` block (the Pi0 constructor reads `siglip.Module` when a model is created), and setting
LIT_TEST_STUB_SIGLIP=1 in the environment applies it to the whole process. It exists for the trainer, loader and data
tests, which check the optimizer, the checkpoint tree and the batch plumbing and not the image tower: a train step with
gradients through the real So400m does not fit this CPU next to another JAX process. It changes every numerics that
involves an image token, so nothing that compares against the lit=off fixture may run under it
(fixture_exact_skip_reason says so) and the model tests of lit_pi0_test / lit_test keep the real encoder.

LIT fields. `lit` and `lit_*` overrides are forwarded to Pi0Config only when the field exists, so this file works
before and after the LIT fields land. On a Pi0Config without them, `lit="off"` (or no lit override) is accepted and
ignored, and any other lit request raises ValueError instead of silently building a stock model. With lit != "off"
the tiny defaults in TINY_LIT_DEFAULTS fill every lit_* field the config has and the caller did not set
(lit_groups=2 divides the dummy depth 4; lit_kv_dim=16 equals the dummy head_dim, as 256 does for the real model).
make_observation passes extra Observation fields (for example the LIT goal) through **fields and rejects names the
dataclass does not have.
"""

import contextlib
import dataclasses
import hashlib
import os
import zlib

from flax import linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import siglip as _siglip
from openpi.shared import nnx_utils

ACTION_DIM = 14
ACTION_HORIZON = 4
HIST_HORIZON = 3
MAX_TOKEN_LEN = 8
BATCH = 2
# Dimensions of the state the fake robot actually drives; the rest of state and actions is zero padding.
ROBOT_DIMS = 7
# One camera of the second sample is missing (zero image, mask False), as for an unplugged or masked camera.
DEFAULT_MASKED = (("right_wrist_0_rgb", 1),)

TINY_LIT_DEFAULTS = {
    "lit_num_latents": 6,
    "lit_dim": 32,
    "lit_groups": 2,
    "lit_heads": 4,
    "lit_kv_dim": 16,
    "lit_pose_tokens": 2,
    "lit_goal_tokens": 2,
    "lit_goal_dims": tuple(range(ROBOT_DIMS)),
}

_CONFIG_DEFAULTS = {
    "paligemma_variant": "dummy",
    "action_expert_variant": "dummy",
    "pi05": True,
    "dtype": "float32",
    "action_dim": ACTION_DIM,
    "action_horizon": ACTION_HORIZON,
    "hist_horizon": HIST_HORIZON,
    "max_token_len": MAX_TOKEN_LEN,
}


STUB_ENV = "LIT_TEST_STUB_SIGLIP"
_REAL_SIGLIP_MODULE = _siglip.Module


class _StubSiglip(nn.Module):
    """The real tower's interface (image (n, 224, 224, 3) -> ((n, 256, num_classes) tokens, aux)) with one Dense layer
    over the 16x16 grid of 14x14 patch means. Unlike the real head it is not zero-initialised: image tokens carry the
    image from the start."""

    num_classes: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        n, h, w, c = image.shape
        patches = jnp.asarray(image, jnp.float32).reshape(n, 16, h // 16, 16, w // 16, c).mean(axis=(2, 4))
        tokens = nn.Dense(self.num_classes, name="head")(patches.reshape(n, 256, c))
        return tokens.astype(self.dtype_mm), {}


def _stub_module(num_classes=None, *, variant=None, **kw):
    return _StubSiglip(num_classes, dtype_mm=kw.get("dtype_mm", "float32"))


def stub_active() -> bool:
    return _siglip.Module is not _REAL_SIGLIP_MODULE


@contextlib.contextmanager
def stub_image_encoder():
    """Models created inside the block use the one-Dense stub instead of SigLIP (see the module docstring)."""
    previous = _siglip.Module
    _siglip.Module = _stub_module
    try:
        yield
    finally:
        _siglip.Module = previous


if os.environ.get(STUB_ENV) == "1":
    _siglip.Module = _stub_module


def fixture_exact_skip_reason(recorded_toolchain: dict, current_toolchain: dict) -> str | None:
    """Why a test that compares exactly against a stored fixture (a bit-for-bit lit=off golden) cannot run here, or
    None when it can. Such a fixture is a function of the toolchain it was generated on (the compiler's reductions
    and the CPU kernels) and of the real image tower: on another machine, jax or numpy, or under the stub encoder,
    "equal" and "different" say nothing, so the test must report not run, never passed. The tolerance-based LIT tests
    do not use this."""
    if stub_active():
        return f"not run: the stub image encoder ({STUB_ENV}) changes the numerics the fixture was generated with"
    keys = ("jax", "flax", "numpy", "machine", "jax_platform")
    differing = {k: (recorded_toolchain.get(k), current_toolchain.get(k)) for k in keys}
    differing = {k: v for k, v in differing.items() if v[0] != v[1]}
    if differing:
        return (
            "not run: toolchain. The exact-equality fixture was generated on "
            + ", ".join(f"{k} {a}" for k, (a, _) in differing.items())
            + "; this run has "
            + ", ".join(f"{k} {b}" for k, (_, b) in differing.items())
        )
    return None


def _is_lit(name: str) -> bool:
    return name == "lit" or name.startswith("lit_")


def make_tiny_config(**overrides) -> _pi0_config.Pi0Config:
    """The tiny pi0.5 config. `overrides` replace any default (and any Pi0Config field); see the module docstring for
    how lit / lit_* overrides are handled."""
    fields = {f.name for f in dataclasses.fields(_pi0_config.Pi0Config)}
    lit_overrides = {k: overrides.pop(k) for k in list(overrides) if _is_lit(k)}
    lit_mode = lit_overrides.get("lit", "off")
    lit_kwargs = dict(lit_overrides)
    if lit_mode != "off":
        lit_kwargs = {**{k: v for k, v in TINY_LIT_DEFAULTS.items() if k in fields}, **lit_overrides}
    missing = sorted(k for k in lit_kwargs if k not in fields)
    if missing:
        if lit_mode != "off":
            raise ValueError(f"Pi0Config has no field(s) {missing}: cannot build lit={lit_mode!r} yet")
        lit_kwargs = {k: v for k, v in lit_kwargs.items() if k in fields}
    return _pi0_config.Pi0Config(**{**_CONFIG_DEFAULTS, **overrides, **lit_kwargs})


def build_model(config: _pi0_config.Pi0Config, seed: int = 0):
    return config.create(jax.random.key(seed))


def _flat_state(model: nnx.Module) -> dict:
    return nnx.state(model).flat_state()


def _path_str(path) -> str:
    return "/".join(str(p) for p in path)


def randomize_zero_init(model: nnx.Module, seed: int = 0, scale: float = 0.02):
    """A copy of `model` where every all-zero floating-point leaf gets scale * N(0, 1) noise.

    Each leaf's key is fold_in(key(seed), crc32(path)), so the values of existing leaves do not change when other
    parameters are added to the model. Non-zero leaves are untouched."""
    graphdef, state = nnx.split(model)
    base = jax.random.key(seed)
    flat = {}
    for path, var in state.flat_state().items():
        x = var.value
        if jnp.issubdtype(x.dtype, jnp.floating) and not bool(jnp.any(x)):
            key = jax.random.fold_in(base, zlib.crc32(_path_str(path).encode()))
            var = var.replace(value=x + scale * jax.random.normal(key, x.shape, x.dtype))  # noqa: PLW2901
        flat[path] = var
    return nnx.merge(graphdef, nnx.State.from_flat_path(flat))


def make_observation(
    seed: int,
    *,
    config: _pi0_config.Pi0Config | None = None,
    batch: int = BATCH,
    masked: tuple[tuple[str, int], ...] = DEFAULT_MASKED,
    **fields,
) -> _model.Observation:
    """A deterministic observation: uniform [-1, 1] images (b, hist, 224, 224, 3) for the three cameras, a state that
    is nonzero on the first ROBOT_DIMS dims, and prompts of different lengths (max_token_len - 3 * i for sample i, at
    least 2) padded with token 0 and mask False. `masked` lists (camera, sample) pairs that are missing: zero image,
    mask False. `fields` replace Observation fields (e.g. the LIT goal) after construction."""
    config = config or make_tiny_config()
    rng = np.random.default_rng(seed)
    hist = config.hist_horizon
    images, image_masks = {}, {}
    for name in _model.IMAGE_KEYS:
        image = rng.uniform(-1.0, 1.0, (batch, hist, *_model.IMAGE_RESOLUTION, 3)).astype(np.float32)
        valid = np.ones((batch,), dtype=bool)
        for camera, sample in masked:
            if camera == name and sample < batch:
                image[sample] = 0.0
                valid[sample] = False
        images[name], image_masks[name] = jnp.asarray(image), jnp.asarray(valid)
    state = np.zeros((batch, config.action_dim), dtype=np.float32)
    state[:, :ROBOT_DIMS] = rng.normal(size=(batch, ROBOT_DIMS))
    lengths = np.maximum(2, config.max_token_len - 3 * np.arange(batch))
    prompt_mask = np.arange(config.max_token_len)[None, :] < lengths[:, None]
    prompt = np.where(prompt_mask, rng.integers(2, 250, (batch, config.max_token_len)), 0).astype(np.int32)
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.asarray(state),
        tokenized_prompt=jnp.asarray(prompt),
        tokenized_prompt_mask=jnp.asarray(prompt_mask),
    )
    if fields:
        known = {f.name for f in dataclasses.fields(_model.Observation)}
        if unknown := sorted(set(fields) - known):
            raise ValueError(f"Observation has no field(s) {unknown}")
        observation = observation.replace(**fields)
    return observation


def make_actions(seed: int, *, config: _pi0_config.Pi0Config | None = None, batch: int = BATCH) -> jax.Array:
    config = config or make_tiny_config()
    actions = np.zeros((batch, config.action_horizon, config.action_dim), dtype=np.float32)
    actions[..., :ROBOT_DIMS] = 0.5 * np.random.default_rng(seed).normal(size=(*actions.shape[:2], ROBOT_DIMS))
    return jnp.asarray(actions)


def fixed_noise(seed: int, *, config: _pi0_config.Pi0Config | None = None, batch: int = BATCH) -> jax.Array:
    config = config or make_tiny_config()
    return jax.random.normal(jax.random.key(seed), (batch, config.action_horizon, config.action_dim))


def training_noise_and_time(rng, batch: int = BATCH, config: _pi0_config.Pi0Config | None = None):
    """(noise, time, mask_num): what Pi0.compute_loss draws from `rng` (4-way split: preprocess, noise, time,
    mask). Mirrors the stock code, so a LIT test can rebuild x_t = time * noise + (1 - time) * actions."""
    config = config or make_tiny_config()
    _, noise_rng, time_rng, mask_rng = jax.random.split(rng, 4)
    shape = (batch, config.action_horizon, config.action_dim)
    noise = jax.random.normal(noise_rng, shape)
    time = jax.random.beta(time_rng, 1.5, 1, (batch, 1)) * 0.999 + 0.001
    time = jnp.broadcast_to(time, shape[:2])
    return noise, time, jax.random.randint(mask_rng, (), 0, config.hist_horizon)


def empty_memory() -> dict:
    """The memory dict a Policy passes on the first call (and after reset_memory). sample_actions fills it in place."""
    return {"memory_tokens": None, "memory_kv_cache": None, "memory_prefix_mask": None}


def jit_loss(model):
    return nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))


def jit_sample(model):
    return nnx_utils.module_jit(model.sample_actions)


def array_digest(x) -> dict:
    """Shape, dtype, float64 sum and sha256 of the raw bytes: an exact-equality fingerprint of an array."""
    a = np.ascontiguousarray(np.asarray(x))
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sum": float(np.sum(a.astype(np.float64))),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def memory_summary(memory: dict) -> dict:
    """array_digest of every array in a memory dict (None stays None, the kv cache becomes a list). Call it right
    after sample_actions: eager calls mutate the dict they were given."""

    def digest(value):
        if value is None:
            return None
        return [array_digest(a) for a in value] if isinstance(value, tuple | list) else array_digest(value)

    return {key: digest(value) for key, value in sorted(memory.items())}


def param_manifest(model: nnx.Module) -> dict:
    """{path: {type, shape, dtype, sum, sha256}} over every variable of the model, sorted by path."""
    return {
        _path_str(path): {"type": var.type.__name__, **array_digest(var.value)}
        for path, var in sorted(_flat_state(model).items(), key=lambda kv: _path_str(kv[0]))
    }
