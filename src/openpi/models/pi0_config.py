import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0
    from openpi.models.pi0_rtc import Pi0Rtc


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    hist_horizon: int = 1
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # VLASH (arXiv 2512.01031, appendix III): the state reaches the action expert as an adaRMS conditioning
    # signal (a zero-initialised MLP beside the time MLP) and leaves the prompt, which then reads
    # "Task: ...;\nAction: ". What lets one observation prefix carry several state/action branches.
    state_cond: bool = False
    # State dims the conditioning may read; every other dim is zeroed first. None: all of them.
    state_cond_dims: tuple[int, ...] | None = None
    # VLASH's shared-observation training: each sample carries this many (state, action chunk) branches
    # at different temporal offsets behind one observation (transforms.TemporalOffset). 0: one branch,
    # the plain layout. Needs state_cond: the branches differ only in their state, which the prompt
    # cannot carry per branch.
    vlash_branches: int = 0
    # The cameras this model encodes. A recipe that masks a camera (AgilexInputs.active_image_keys)
    # can leave it out here: a masked camera's tokens are padding -- never attended to, never shifting
    # a real token's position -- so skipping them is the same model with fewer SigLIP passes and a
    # shorter prefix. Checkpoints carry no per-camera weights, so they load under either setting.
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS
    # RACE (arXiv 2610.05719): a transition-timing head reads an auxiliary one-step denoising pass and the
    # current call's final-layer VLM features, and its prior modulates every adaptive RMSNorm of the action
    # expert (zero-initialised, so a checkpoint without RACE weights starts as the base model). Trains on
    # Observation.transition_window (DataConfig.race_targets_dir) with L_full + race_aux_weight * L_aux +
    # race_timing_weight * L_timing.
    race: bool = False
    # Sec. 4.1: "we set lambda_aux = 0.1 and lambda_timing = 0.05".
    race_aux_weight: float = 0.1
    race_timing_weight: float = 0.05
    # Titans (arXiv 2501.00663): a neural long-term memory (openpi.models.titans) whose weights are written at test
    # time. Training unrolls its writes over this many older frames of one camera, titans_stride calls apart (a call
    # is hist_interval frames; Observation.memory_images), and the action expert reads what the memory returns for
    # the current frame through gemma's zero-gated memory branch, so a checkpoint without these weights starts as
    # itself. Serving keeps one memory per client and writes the current frame every titans_stride calls
    # (Pi0.step_long_memory). 0: none.
    titans_frames: int = 0
    titans_stride: int = 4
    # The camera the memory is given, and the cells per side each of its frames is averaged over.
    titans_image_key: str = "base_0_rgb"
    titans_grid: int = 4
    # The memory network: its width, the width of its hidden layers, and its layers (1: a linear memory).
    titans_dim: int = 256
    titans_hidden_dim: int = 512
    titans_depth: int = 2
    # Training only: the chance a sample keeps just its newest n older frames, n uniform in 0..titans_frames - 1,
    # as the first calls of a served run see the memory (pi0.cut_memory_short).
    titans_dropout: float = 0.1

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if not self.image_keys or set(self.image_keys) - set(_model.IMAGE_KEYS):
            raise ValueError(f"image_keys must be a non-empty subset of {_model.IMAGE_KEYS}, got {self.image_keys}")
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05 and not self.state_cond)
        if self.state_cond and not self.pi05:
            raise ValueError("state_cond is pi0.5's state path: pi0 already feeds the state as a suffix token")
        if self.state_cond and self.discrete_state_input:
            raise ValueError("state_cond moves the state out of the prompt: discrete_state_input must be False")
        if self.vlash_branches < 0:
            raise ValueError(f"vlash_branches must be >= 0, got {self.vlash_branches}")
        if self.vlash_branches and not self.state_cond:
            raise ValueError("vlash_branches needs state_cond: a shared prompt cannot carry one state per branch")
        if self.state_cond_dims is not None and not self.state_cond:
            raise ValueError("state_cond_dims masks the conditioning state; it needs state_cond")
        if self.race and not self.pi05:
            raise ValueError("race modulates pi0.5's adaptive RMSNorm; pi0 has none")
        if self.race and self.vlash_branches:
            raise ValueError("race is wired for one action chunk per sample, not vlash_branches")
        if self.titans_frames < 0:
            raise ValueError(f"titans_frames must be >= 0, got {self.titans_frames}")
        if self.titans_frames:
            if not self.pi05:
                raise ValueError("the Titans memory is wired for pi0.5")
            if self.race or self.vlash_branches:
                raise ValueError("the Titans memory is wired for the plain pi0.5 loss, not race or vlash_branches")
            if self.titans_image_key not in self.image_keys:
                raise ValueError(f"titans_image_key {self.titans_image_key!r} is not one of image_keys {self.image_keys}")
            if min(self.titans_stride, self.titans_grid, self.titans_dim, self.titans_hidden_dim, self.titans_depth) < 1:
                raise ValueError("titans_stride, titans_grid, titans_dim, titans_hidden_dim and titans_depth must be positive")
            if not 0.0 <= self.titans_dropout < 1.0:
                raise ValueError(f"titans_dropout must be in [0, 1), got {self.titans_dropout}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        # With branches, the state and the action chunk carry a branch axis behind the one observation.
        branch_shape = [self.vlash_branches] if self.vlash_branches else []

        # A Titans model's older frames of its memory camera, newest first, and which of them exist.
        memory_spec = {} if not self.titans_frames else {
            "memory_images": jax.ShapeDtypeStruct(
                [batch_size, self.titans_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32),
            "memory_mask": jax.ShapeDtypeStruct([batch_size, self.titans_frames], jnp.bool_)}
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={key: image_spec for key in self.image_keys},
                image_masks={key: image_mask_spec for key in self.image_keys},
                state=jax.ShapeDtypeStruct([batch_size, *branch_shape, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                **memory_spec,
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, *branch_shape, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        if self.race:
            # RACE's weights are new and always trained, including the modulation inside the frozen LLM.
            filters.append(nnx.Not(nnx_utils.PathRegex(RACE_PARAMS_REGEX)))
        if self.titans_frames:
            # The memory's weights are new and always trained, including its read branch inside the frozen LLM.
            filters.append(nnx.Not(nnx_utils.PathRegex(TITANS_PARAMS_REGEX)))
        return nnx.All(*filters)


# Every RACE parameter's path contains this (Pi0's race_* attributes, gemma's race_modulation).
RACE_PARAMS_REGEX = ".*race_.*"
# Every parameter a Titans model adds: Pi0's titans_memory, and the action expert's read branch in gemma
# (memory_k, memory_v, memory_gate).
TITANS_PARAMS_REGEX = ".*titans_.*|.*memory_(k|v|gate).*"


@dataclasses.dataclass(frozen=True)
class Pi0RtcConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    hist_horizon: int = 1
    max_delay: int = 0
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05Rtc
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Rtc":
        from openpi.models.pi0_rtc import Pi0Rtc

        return Pi0Rtc(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
