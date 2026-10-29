import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.dinov2 as _dinov2
import openpi.models.lora as _lora
import openpi.models.model as _model
from openpi.models.pi0_config import MEMORY_PARAMS_REGEX
from openpi.models.pi0_config import RACE_PARAMS_REGEX
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Grow the checkpoint's LoRA weights to this model's larger rank (lora.expand_rank) instead of failing the
    # shape check, for a config whose LoRA is bigger than the run the checkpoint came from.
    expand_lora_rank: bool = False
    # Resample SigLIP's learned position embedding to this model's image size (siglip.resize_posemb), for a config
    # that reads larger images than the run the checkpoint came from.
    resize_posemb: bool = False
    # Add DINOv2 weights (a Hugging Face repo id, or a local directory with model.safetensors) for a config with a
    # DINOv2 branch, unless the checkpoint already carries a trained branch: a warm start keeps its own.
    dino_variant: str | None = None
    dino_weights: str | None = None

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        if self.dino_variant is not None and "dino" not in loaded_params:
            source = self.dino_weights or _dinov2.HF_REPOS[self.dino_variant]
            loaded_params = {**loaded_params, "dino": _dinov2.load_hf_params(source, _dinov2.get_config(self.dino_variant))}
        grows = [grow for grow, wanted in ((_lora.expand_rank, self.expand_lora_rank),
                                           (_siglip.resize_posemb, self.resize_posemb)) if wanted]
        if grows:
            loaded_params = _grow_to_model(loaded_params, params, *grows)
        # Add all missing LoRA weights, pi0.5's optional state-conditioning MLP (Pi0Config.state_cond) and RACE's
        # weights (Pi0Config.race), which a base checkpoint does not carry; the latter two start the model as the
        # checkpoint was (zero-initialised outputs).
        return _merge_params(loaded_params, params,
                             missing_regex=f".*lora.*|.*state_(proj|mlp_in|mlp_out).*|.*dino_proj.*|{RACE_PARAMS_REGEX}|{MEMORY_PARAMS_REGEX}")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _grow_to_model(loaded_params: at.Params, params: at.Params, *grows) -> at.Params:
    reference = flax.traverse_util.flatten_dict(params, sep="/")
    flat = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    for grow in grows:
        flat = grow(flat, reference)
    return flax.traverse_util.unflatten_dict(flat, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
