"""DINOv2 (ViT-S/B/L with 14-pixel patches) in Flax, following the reference implementation, with a converter from
its released Hugging Face weights. pi0 adds its patch tokens to SigLIP's (Pi0Config.dino_variant)."""

from collections.abc import Mapping
import dataclasses
import os
import pathlib
from typing import Any

import flax.linen as nn
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy

PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HF_REPOS = {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base", "large": "facebook/dinov2-large"}


@dataclasses.dataclass(frozen=True)
class Config:
    width: int
    depth: int
    heads: int
    mlp_dim: int
    grid: int = 37  # the side of the position-embedding grid the released weights carry (518 / 14)


def get_config(variant: str) -> Config:
    configs = {
        "small": Config(384, 12, 6, 1536),
        "base": Config(768, 12, 12, 3072),
        "large": Config(1024, 24, 16, 4096),
        "dummy": Config(32, 2, 2, 64, grid=4),
    }
    if variant not in configs:
        raise ValueError(f"Unknown DINOv2 variant: {variant}")
    return configs[variant]


def _resize_axis(x: jax.Array, out_size: int, axis: int, a: float) -> jax.Array:
    axis %= x.ndim
    in_size = x.shape[axis]
    if in_size == out_size:
        return x
    source = (np.arange(out_size) + 0.5) * (in_size / out_size) - 0.5
    floor = np.floor(source)
    t = source - floor

    def near(distance):
        return ((a + 2) * distance - (a + 3)) * distance * distance + 1

    def far(distance):
        return ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a

    weights = np.stack([far(t + 1), near(t), near(1 - t), far(2 - t)], axis=-1).astype(np.float32)
    index = np.clip(floor[:, None].astype(np.int64) + np.arange(-1, 3)[None], 0, in_size - 1)
    taken = jnp.take(x, jnp.asarray(index.reshape(-1)), axis=axis)
    taken = taken.reshape(*x.shape[:axis], out_size, 4, *x.shape[axis + 1 :])
    broadcast = weights.reshape(*([1] * axis), out_size, 4, *([1] * (x.ndim - axis - 1)))
    return jnp.sum(taken * broadcast, axis=axis + 1)


def resize_bicubic(x: jax.Array, size: tuple[int, int], a: float = -0.75) -> jax.Array:
    """(..., h, w, c) to (..., size[0], size[1], c) the way torch.nn.functional.interpolate does for
    mode="bicubic", align_corners=False (cubic convolution with a = -0.75, indices clamped at the edge)."""
    return _resize_axis(_resize_axis(x, size[0], -3, a), size[1], -2, a)


class _Block(nn.Module):
    config: Config
    dtype_mm: Any

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        head_dim = cfg.width // cfg.heads

        def norm(name, value):
            return nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32, name=name)(value.astype(jnp.float32)).astype(self.dtype_mm)

        y = norm("norm1", x)
        batch, tokens, _ = y.shape
        q, k, v = (
            nn.Dense(cfg.width, dtype=self.dtype_mm, name=name)(y).reshape(batch, tokens, cfg.heads, head_dim)
            for name in ("q", "k", "v")
        )
        logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) / np.sqrt(head_dim)
        weights = jax.nn.softmax(logits, axis=-1).astype(self.dtype_mm)
        y = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(batch, tokens, cfg.width)
        y = nn.Dense(cfg.width, dtype=self.dtype_mm, name="o")(y)
        x = x + y * self.param("layer_scale1", nn.initializers.ones, (cfg.width,)).astype(self.dtype_mm)

        y = nn.Dense(cfg.mlp_dim, dtype=self.dtype_mm, name="fc1")(norm("norm2", x))
        y = nn.Dense(cfg.width, dtype=self.dtype_mm, name="fc2")(nn.gelu(y, approximate=False))
        return x + y * self.param("layer_scale2", nn.initializers.ones, (cfg.width,)).astype(self.dtype_mm)


class Dinov2(nn.Module):
    config: Config
    dtype_mm: Any = jnp.float32

    @nn.compact
    def __call__(self, image):
        """(b, h, w, 3) in [-1, 1] to (b, 1 + (h / 14) * (w / 14), width): the CLS token, then the patch tokens."""
        cfg = self.config
        batch, height, width, _ = image.shape
        grid_h, grid_w = height // PATCH, width // PATCH
        x = (image.astype(jnp.float32) + 1.0) / 2.0
        x = (x - jnp.asarray(IMAGENET_MEAN, jnp.float32)) / jnp.asarray(IMAGENET_STD, jnp.float32)
        x = nn.Conv(cfg.width, (PATCH, PATCH), strides=(PATCH, PATCH), padding="VALID", dtype=self.dtype_mm,
                    name="patch_embed")(x.astype(self.dtype_mm))
        x = x.reshape(batch, grid_h * grid_w, cfg.width)

        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, cfg.width))
        position = self.param("pos_embedding", nn.initializers.normal(0.02), (1, 1 + cfg.grid**2, cfg.width))
        patch_position = position[:, 1:]
        if (grid_h, grid_w) != (cfg.grid, cfg.grid):
            patch_position = resize_bicubic(
                patch_position.reshape(1, cfg.grid, cfg.grid, cfg.width).astype(jnp.float32), (grid_h, grid_w)
            ).reshape(1, grid_h * grid_w, cfg.width)
        x = jnp.concatenate([jnp.broadcast_to(cls, (batch, 1, cfg.width)).astype(x.dtype), x], axis=1)
        x = x + jnp.concatenate([position[:, :1], patch_position], axis=1).astype(x.dtype)

        block = nn.remat(_Block)
        for index in range(cfg.depth):
            x = block(cfg, self.dtype_mm, name=f"layer_{index}")(x)
        return nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32, name="norm")(x.astype(jnp.float32)).astype(self.dtype_mm)


_ATTENTION_NAMES = {
    "q": ("attention.q_proj", "attention.attention.query"),
    "k": ("attention.k_proj", "attention.attention.key"),
    "v": ("attention.v_proj", "attention.attention.value"),
    "o": ("attention.o_proj", "attention.output.dense"),
}


def convert_hf_state_dict(state: Mapping[str, Any], config: Config) -> dict[str, Any]:
    """The Flax parameters of Dinov2 from a Hugging Face DINOv2 state dict (the released checkpoints' attention names
    and the current transformers' both work). Refuses a state dict with a key it does not use."""
    state = {key.removeprefix("dinov2."): np.asarray(value) for key, value in state.items()}
    used: set[str] = set()
    params: dict[str, Any] = {}

    def take(key: str) -> np.ndarray:
        used.add(key)
        return state[key]

    def put(path: tuple[str, ...], value: np.ndarray) -> None:
        node = params
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value

    def dense(path: tuple[str, ...], prefix: str) -> None:
        put((*path, "kernel"), take(f"{prefix}.weight").T)
        put((*path, "bias"), take(f"{prefix}.bias"))

    def norm(path: tuple[str, ...], prefix: str) -> None:
        put((*path, "scale"), take(f"{prefix}.weight"))
        put((*path, "bias"), take(f"{prefix}.bias"))

    put(("patch_embed", "kernel"), take("embeddings.patch_embeddings.projection.weight").transpose(2, 3, 1, 0))
    put(("patch_embed", "bias"), take("embeddings.patch_embeddings.projection.bias"))
    put(("cls_token",), take("embeddings.cls_token"))
    put(("pos_embedding",), take("embeddings.position_embeddings"))
    for index in range(config.depth):
        layer = f"encoder.layer.{index}"
        target = (f"layer_{index}",)
        norm((*target, "norm1"), f"{layer}.norm1")
        norm((*target, "norm2"), f"{layer}.norm2")
        for name, candidates in _ATTENTION_NAMES.items():
            prefix = next((f"{layer}.{candidate}" for candidate in candidates if f"{layer}.{candidate}.weight" in state), None)
            if prefix is None:
                raise ValueError(f"layer {index} has no {candidates[0]} or {candidates[1]}")
            dense((*target, name), prefix)
        dense((*target, "fc1"), f"{layer}.mlp.fc1")
        dense((*target, "fc2"), f"{layer}.mlp.fc2")
        put((*target, "layer_scale1"), take(f"{layer}.layer_scale1.lambda1"))
        put((*target, "layer_scale2"), take(f"{layer}.layer_scale2.lambda1"))
    norm(("norm",), "layernorm")

    unused = sorted(set(state) - used - {"embeddings.mask_token"})
    if unused:
        raise ValueError(f"DINOv2 state dict has keys this model does not use: {unused[:5]}")
    if params["pos_embedding"].shape != (1, 1 + config.grid**2, config.width):
        raise ValueError(f"position embedding {params['pos_embedding'].shape} does not fit {config}")
    return params


def load_hf_params(source: str, config: Config) -> dict[str, Any]:
    """The Flax parameters from a local directory holding model.safetensors, or from a Hugging Face repo id."""
    if os.path.isdir(source):
        path = pathlib.Path(source) / "model.safetensors"
    else:
        path = pathlib.Path(huggingface_hub.hf_hub_download(source, "model.safetensors"))
    return convert_hf_state_dict(safetensors.numpy.load_file(path), config)
