"""Infinite Hands YAM configurations for StreamPI (pi0.5 + streaming temporal KV-cache memory)."""

import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import tyro
from typing_extensions import override

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"
BAGGING_REPO_ID = "local/yam_bagging_three"
FIRSTTRY_REPO_ID = "local/yam_fullcorpus_teleop_firsttry_20260914"  # the studio's full-corpus train split, 1127 episodes
# 483 REAL (not mirrored) native left-arm teleop episodes, --leaders left: the right arm's real
# gravity-comp trajectory is recorded, not a mirrored/synthetic one.
BAGGING_LEFT_REAL_REPO_ID = "local/yam_bagging_left_real_20260924"
BAGGING_PROMPT = "place one part in the bag"
# Driven/held/pad weights for a single-arm dataset on this 14-of-32-dim bimanual embodiment (see
# Pi0Config.action_dim_weights). Duplicated from infinite-hands/VLA-Precision's identical constants
# rather than imported -- these are two independently-evolving forks with no dependency between them.
DRIVEN_ARM_WEIGHT = 1.0
# 0, not VLA-Precision's 0.07: that held arm was pinned to a constant, this one is a real arm in
# gravity comp. Its delta band is +/-0.0004 rad, so the ~0.1% of windows where it drifted normalize
# to 400-2,200 and carried 95% of the weighted target energy at 0.07 (measured over all 483 episodes
# of yam_bagging_left_real_20260924). Served, that same band keeps its untrained output a hold.
HELD_ARM_WEIGHT = 0.0
PAD_WEIGHT = 0.0
LEFT_ARM_DIMS = tuple(range(7))    # [L j0..5, L grip]
RIGHT_ARM_DIMS = tuple(range(7, 14))  # [R j0..5, R grip]


def single_arm_weights(driven: str, action_dim: int) -> tuple[float, ...]:
    """Per-dimension action loss weights for a corpus in which only `driven` moves."""
    if driven not in ("left", "right"):
        raise ValueError(f"driven arm must be 'left' or 'right', got {driven!r}")
    left = DRIVEN_ARM_WEIGHT if driven == "left" else HELD_ARM_WEIGHT
    right = HELD_ARM_WEIGHT if driven == "left" else DRIVEN_ARM_WEIGHT
    weights = [left] * 7 + [right] * 7
    return tuple(weights + [PAD_WEIGHT] * (action_dim - len(weights)))
# T = 5 frames of temporal context, the setting behind every real-robot result in the paper.
HIST_HORIZON = 5
# Control frames between two policy calls at the YAM cell's 30 Hz. The deploy loop MUST call the
# policy every hist_interval frames (its chunk_play), or the served KV-cache history is spaced
# differently from the history the model was trained on. This is therefore a per-config fact, not a
# global one: a checkpoint carries the cadence it was trained at, and the config name is what says
# which. The two coexist so they can be compared without rebuilding the image.
HIST_INTERVAL = 10
# The cadence is also the inference budget: the loop only ever executes rows 0..hist_interval-1 of
# each chunk, so at 30 Hz a value of 10 is a 333 ms budget against a measured 300-600 ms StreamPI
# round trip -- the cell executes only the tail of many chunks. 20 frames is 667 ms. ACTION_HORIZON
# is the ceiling (the loop cannot execute rows the model did not predict); 20 leaves 10 rows of
# margin for RTC. In exchange the history reaches (HIST_HORIZON - 1) * 20 = 80 frames back rather
# than 40, so the same five frames span 2.67 s instead of 1.33 s and the policy sees a fresh
# observation half as often. Model shape is untouched -- HIST_HORIZON and ACTION_HORIZON set the
# token count and KV-cache size -- so fsdp_devices and batch_size below are unchanged.
HIST_INTERVAL_WIDE = 20
ACTION_HORIZON = 30
# The YAM LeRobot layout: three cameras, 14-dim state/action [L j0..5, L grip, R j0..5, R grip].
YAM_REPACK = _transforms.Group(
    inputs=[
        _transforms.RepackTransform(
            {
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
            }
        )
    ]
)


def get_ih_yam_configs():
    # Deferred: config.py imports this module while building its registry.
    from openpi.training.config import DataConfig, LeRobotAgilexDataConfig, ModelTransformFactory, TrainConfig

    @dataclasses.dataclass(frozen=True)
    class LeRobotYamStreamDataConfig(LeRobotAgilexDataConfig):
        """The AgileX transforms (14-dim bimanual, three cameras, streaming history on every camera)
        with a served default prompt: the YAM datasets carry no prompt column, so InjectDefaultPrompt
        fills it at train and serve time exactly as the openpi fork's YAM configs do."""

        repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=YAM_REPACK)

        @override
        def create(self, assets_dirs, model_config):
            return dataclasses.replace(
                super().create(assets_dirs, model_config),
                model_transforms=ModelTransformFactory(
                    default_prompt=self.default_prompt, active_state_dims=self.active_state_dims,
                )(model_config),
                # pi0.5's own default (the AgileX base config forces z-score). The YAM left gripper never
                # moves in the corpus, so its std is ~1e-3 and z-scoring turns its noise into a loss of
                # tens of thousands; the quantile band is floored by the stats writer instead.
                use_quantile_norm=True,
            )

    def stream_config(name: str, repo_id: str, prompt: str, *, lora: bool,
                      hist_interval: int = HIST_INTERVAL,
                      active_image_keys: frozenset[str] | None = None,
                      active_state_dims: tuple[int, ...] | None = None,
                      action_dim_weights: tuple[float, ...] | None = None):
        model = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=ACTION_HORIZON,
            hist_horizon=HIST_HORIZON,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
            action_dim_weights=action_dim_weights,
        )
        return TrainConfig(
            name=name,
            model=model,
            data=LeRobotYamStreamDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                active_image_keys=active_image_keys,
                active_state_dims=active_state_dims,
                base_config=DataConfig(
                    prompt_from_task=False,
                    hist_horizon=HIST_HORIZON,
                    hist_interval=hist_interval,
                    enable_jitter=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE_PARAMS),
            # LoRA fits one H100 (make_mesh refuses a device count the FSDP count does not divide).
            # The FULL fine-tune does not: at batch 32 with T=5 it ran out of memory on one 80 GB
            # H100 (a 24 GB allocation on top of ~53 GB of fp32 params + grads + Adam state), so it
            # shards over the fork's default four devices, the paper's regime.
            fsdp_devices=1 if lora else 4,
            # T=5 multiplies the image tokens by five; LoRA's batch is a single-H100 starting point.
            batch_size=16 if lora else 32,
            # One sample decodes three cameras x T=5 history frames, ~150 ms each: at eight workers the
            # loader fed ~54 frames/s and the trainer waited on it (8.95 s/step, 5-17 s swings). The
            # compute profile pins 32 CPUs; leave a few for the main process and JAX.
            num_workers=24,
            num_train_steps=20_000,
            lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000),
            save_interval=1_000,
            keep_period=5_000,
            freeze_filter=model.get_freeze_filter(),
            ema_decay=None if lora else 0.99,
        )

    return [
        stream_config("pi05_yam_stream5_bagging", BAGGING_REPO_ID, BAGGING_PROMPT, lora=True),
        stream_config("pi05_yam_stream5_bagging_full", BAGGING_REPO_ID, BAGGING_PROMPT, lora=False),
        stream_config("pi05_yam_stream5_firsttry", FIRSTTRY_REPO_ID, BAGGING_PROMPT, lora=True),
        stream_config("pi05_yam_stream5_firsttry_full", FIRSTTRY_REPO_ID, BAGGING_PROMPT, lora=False),
        # The same four at the wider cadence; `_i20` is the only thing that differs.
        stream_config("pi05_yam_stream5_i20_bagging", BAGGING_REPO_ID, BAGGING_PROMPT, lora=True,
                      hist_interval=HIST_INTERVAL_WIDE),
        stream_config("pi05_yam_stream5_i20_bagging_full", BAGGING_REPO_ID, BAGGING_PROMPT, lora=False,
                      hist_interval=HIST_INTERVAL_WIDE),
        stream_config("pi05_yam_stream5_i20_firsttry", FIRSTTRY_REPO_ID, BAGGING_PROMPT, lora=True,
                      hist_interval=HIST_INTERVAL_WIDE),
        stream_config("pi05_yam_stream5_i20_firsttry_full", FIRSTTRY_REPO_ID, BAGGING_PROMPT, lora=False,
                      hist_interval=HIST_INTERVAL_WIDE),
        # Real (not mirrored) native left-arm teleop, wrist-camera-only, the held right arm out of the
        # loss (weight 0) AND its state hidden from the model (the tokenized state only, at train and
        # serve time, via active_state_dims -- delta targets still use the real state).
        stream_config("pi05_yam_stream5_i20_bagging_left_real", BAGGING_LEFT_REAL_REPO_ID, BAGGING_PROMPT,
                      lora=True, hist_interval=HIST_INTERVAL_WIDE,
                      active_image_keys=frozenset({"left_wrist_0_rgb"}),
                      active_state_dims=LEFT_ARM_DIMS,
                      action_dim_weights=single_arm_weights("left", 32)),
    ]
