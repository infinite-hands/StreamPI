"""Infinite Hands YAM configurations for StreamPI (pi0.5 + streaming temporal KV-cache memory)."""

import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.policies.tempo_history as _tempo_history
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
# 313 REAL native right-arm teleop episodes, --leaders right (2026-09-30 + 2026-10-02); the left arm
# was held, except in four takes where a left policy ran beside the teleop.
BAGGING_RIGHT_REAL_REPO_ID = "local/yam_bagging_right_real_20261002"
BAGGING_PROMPT = "place one part in the bag"
LEFT_ARM_DIMS = tuple(range(7))    # [L j0..5, L grip]
RIGHT_ARM_DIMS = tuple(range(7, 14))  # [R j0..5, R grip]
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
# --- TEMPO on the single-arm recipes (openpi.models.tempo, openpi.policies.tempo_history) ---
# Written by infinite-hands' models.vla.tempo prepare step (paths.tempo_stream_cache): one cache per
# dataset and camera, holding tokens/ and actions/. The two repos must agree on this root.
TEMPO_CACHE_ROOT = "/misc/tempo-caches/stream"
# The single-arm production checkpoints the TEMPO pair (channels on / matched control) warm-starts from.
LEFT_REAL_WARM_START = "/checkpoints/fine-tuned/pi05_yam_stream5_i20_bagging_left_real/left-real-i20-run3/15499/params"
RIGHT_REAL_WARM_START = "/checkpoints/fine-tuned/pi05_yam_stream5_i20_bagging_right_real/right-real-i20-run1/19999/params"
TEMPO_WARM_STEPS = 5_000  # upstream TEMPO's warm-started rung length; the control trains exactly as long
TEMPO_WARMUP_STEPS = 500  # upstream TEMPO's warmup for that rung; peak and floor stay this recipe's own
# A warm start must normalise exactly as the checkpoint it continues: the pair reads the base recipe's stats.
NORM_STATS_ROOT = "/checkpoints/assets"
TEMPO_NEW_MODULES = ".*lora.*|.*(sam2_fusion|action_history_tokens|action_history_cond).*"
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
# The same layout plus the TEMPO inputs LoadTempoHistory attaches before repacking.
YAM_TEMPO_REPACK = _transforms.Group(
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
                "sam2_tokens": "sam2_tokens",
                "action_history": "action_history",
                "action_history_is_pad": "action_history_is_pad",
            }
        )
    ]
)


def get_ih_yam_configs():
    # Deferred: config.py imports this module while building its registry.
    from openpi.training.config import AssetsConfig, DataConfig, LeRobotAgilexDataConfig, TrainConfig

    @dataclasses.dataclass(frozen=True)
    class LeRobotYamStreamDataConfig(LeRobotAgilexDataConfig):
        """The AgileX transforms (14-dim bimanual, three cameras, streaming history on every camera)
        with a served default prompt: the YAM datasets carry no prompt column, so InjectDefaultPrompt
        fills it at train and serve time exactly as the openpi fork's YAM configs do."""

        repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=YAM_REPACK)

        @override
        def create(self, assets_dirs, model_config):
            config = dataclasses.replace(
                super().create(assets_dirs, model_config),
                # pi0.5's own default (the AgileX base config forces z-score). The YAM left gripper never
                # moves in the corpus, so its std is ~1e-3 and z-scoring turns its noise into a loss of
                # tens of thousands; the quantile band is floored by the stats writer instead.
                use_quantile_norm=True,
            )
            dims = config.tempo_action_dims
            if dims is None:
                return config
            normalize = _tempo_history.NormalizeActionHistory((config.norm_stats or {}).get("state"), tuple(dims))
            return dataclasses.replace(config, model_transforms=_transforms.Group(
                inputs=[normalize, *config.model_transforms.inputs], outputs=config.model_transforms.outputs))

    def stream_config(name: str, repo_id: str, prompt: str, *, lora: bool,
                      hist_interval: int = HIST_INTERVAL,
                      active_image_keys: frozenset[str] | None = None,
                      active_state_dims: tuple[int, ...] | None = None,
                      held_action_dims: tuple[int, ...] | None = None,
                      warm_start: str | None = None,
                      norm_stats_from: str | None = None,
                      tempo_camera: str | None = None):
        # tempo_camera (a dataset camera role) turns TEMPO's channels on: that camera's SAM2 cue and
        # the driven arm's (active_state_dims') action history. A warm start continues an existing
        # checkpoint for TEMPO_WARM_STEPS instead of training 20k from pi05_base.
        tempo_image_key = {"cam_high": "base_0_rgb", "cam_left_wrist": "left_wrist_0_rgb",
                           "cam_right_wrist": "right_wrist_0_rgb"}.get(tempo_camera, "base_0_rgb")
        model = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=ACTION_HORIZON,
            hist_horizon=HIST_HORIZON,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
            tempo_sam2=tempo_camera is not None,
            tempo_sam2_image_key=tempo_image_key,
            tempo_action_history=tempo_camera is not None,
            tempo_action_history_dim=len(active_state_dims) if active_state_dims else 14,
        )
        return TrainConfig(
            name=name,
            model=model,
            data=LeRobotYamStreamDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                active_image_keys=active_image_keys,
                active_state_dims=active_state_dims,
                held_action_dims=held_action_dims,
                repack_transforms=YAM_TEMPO_REPACK if tempo_camera else YAM_REPACK,
                assets=AssetsConfig(assets_dir=f"{NORM_STATS_ROOT}/{norm_stats_from}", asset_id=repo_id)
                if norm_stats_from else AssetsConfig(),
                base_config=DataConfig(
                    prompt_from_task=False,
                    hist_horizon=HIST_HORIZON,
                    hist_interval=hist_interval,
                    enable_jitter=True,
                    tempo_cache_dir=f"{TEMPO_CACHE_ROOT}/{repo_id}/{tempo_camera}" if tempo_camera else None,
                    tempo_hist_key=f"observation.images.{tempo_camera}" if tempo_camera else None,
                    tempo_action_dims=active_state_dims if tempo_camera else None,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                warm_start or PI05_BASE_PARAMS, missing_regex=TEMPO_NEW_MODULES if tempo_camera else ".*lora.*"),
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
            num_train_steps=TEMPO_WARM_STEPS if warm_start else 20_000,
            lr_schedule=(_optimizer.CosineDecaySchedule(warmup_steps=TEMPO_WARMUP_STEPS, decay_steps=TEMPO_WARM_STEPS)
                         if warm_start else _optimizer.CosineDecaySchedule(decay_steps=20_000)),
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
        # Real (not mirrored) native left-arm teleop: wrist camera only, the right arm's state hidden
        # from the model, and the right arm trained to hold still -- it was held in gravity comp, so
        # its recorded drift is not behaviour to imitate.
        stream_config("pi05_yam_stream5_i20_bagging_left_real", BAGGING_LEFT_REAL_REPO_ID, BAGGING_PROMPT,
                      lora=True, hist_interval=HIST_INTERVAL_WIDE,
                      active_image_keys=frozenset({"left_wrist_0_rgb"}),
                      active_state_dims=LEFT_ARM_DIMS,
                      held_action_dims=RIGHT_ARM_DIMS),
        # Its mirror for real native right-arm teleop: the same recipe with the arms swapped.
        stream_config("pi05_yam_stream5_i20_bagging_right_real", BAGGING_RIGHT_REAL_REPO_ID, BAGGING_PROMPT,
                      lora=True, hist_interval=HIST_INTERVAL_WIDE,
                      active_image_keys=frozenset({"right_wrist_0_rgb"}),
                      active_state_dims=RIGHT_ARM_DIMS,
                      held_action_dims=LEFT_ARM_DIMS),
        # TEMPO on each single-arm recipe, warm-started from its production checkpoint, and the matched
        # control: the same checkpoint, data and schedule with the channels off.
        *[
            stream_config(f"pi05_yam_stream5_i20_bagging_{arm}_real_{variant}", repo_id, BAGGING_PROMPT,
                          lora=True, hist_interval=HIST_INTERVAL_WIDE,
                          active_image_keys=frozenset({f"{arm}_wrist_0_rgb"}),
                          active_state_dims=driven, held_action_dims=held, warm_start=warm_start,
                          norm_stats_from=f"pi05_yam_stream5_i20_bagging_{arm}_real",
                          tempo_camera=f"cam_{arm}_wrist" if variant == "tempo" else None)
            for arm, repo_id, driven, held, warm_start in (
                ("left", BAGGING_LEFT_REAL_REPO_ID, LEFT_ARM_DIMS, RIGHT_ARM_DIMS, LEFT_REAL_WARM_START),
                ("right", BAGGING_RIGHT_REAL_REPO_ID, RIGHT_ARM_DIMS, LEFT_ARM_DIMS, RIGHT_REAL_WARM_START),
            )
            for variant in ("tempo", "tempo_ctrl")
        ],
    ]
