"""Blending a fine-tuned checkpoint toward the checkpoint it started from (weight-space interpolation)."""

import pathlib
import shutil
from typing import Any

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.download as download


def blend_params(base: dict[str, Any], tuned: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Interpolates flat "a/b/c" parameter maps from `base` (alpha 0) to `tuned` (alpha 1).

    Weights both checkpoints carry mix linearly. A LoRA detour has no base counterpart, so its `lora_b` is scaled by
    alpha, which scales its contribution to the output by alpha. Anything else only `tuned` has is kept as it is.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    blended = {}
    for key, value in tuned.items():
        name = key.rsplit("/", 1)[-1]
        array = np.asarray(value)
        if name.endswith("lora_b"):
            blended[key] = (alpha * array.astype(np.float32)).astype(array.dtype)
        elif "lora" in name or key not in base or np.shape(base[key]) != array.shape:
            blended[key] = value
        else:
            mixed = (1.0 - alpha) * np.asarray(base[key], np.float32) + alpha * array.astype(np.float32)
            blended[key] = mixed.astype(array.dtype)
    return blended


def write_blended_checkpoint(base_params: str, tuned_params: str, alpha: float, out_dir: str) -> pathlib.Path:
    """Writes `<out_dir>/params` (and the tuned checkpoint's assets, which hold its norm stats) blended at `alpha`."""
    base = _model.restore_params(download.maybe_download(base_params), restore_type=np.ndarray)
    tuned = _model.restore_params(download.maybe_download(tuned_params), restore_type=np.ndarray)
    blended = blend_params(
        flax.traverse_util.flatten_dict(base, sep="/"), flax.traverse_util.flatten_dict(tuned, sep="/"), alpha)
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(out / "params", {"params": flax.traverse_util.unflatten_dict(blended, sep="/")})
    assets = pathlib.Path(str(tuned_params)).parent / "assets"
    if assets.is_dir():
        shutil.copytree(assets, out / "assets", dirs_exist_ok=True)
    return out
