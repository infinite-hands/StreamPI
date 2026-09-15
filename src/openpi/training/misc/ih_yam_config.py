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
BAGGING_PROMPT = "place one part in the bag"
# T = 5 frames of temporal context, the setting behind every real-robot result in the paper.
HIST_HORIZON = 5
# Control frames between two policy calls at the YAM cell's 30 Hz. The deploy loop MUST call the
# policy every HIST_INTERVAL frames (its chunk_play), or the served KV-cache history is spaced
# differently from the history the model was trained on.
HIST_INTERVAL = 10
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
                model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
                # pi0.5's own default (the AgileX base config forces z-score). The YAM left gripper never
                # moves in the corpus, so its std is ~1e-3 and z-scoring turns its noise into a loss of
                # tens of thousands; the quantile band is floored by the stats writer instead.
                use_quantile_norm=True,
            )

    def stream_config(name: str, repo_id: str, prompt: str, *, lora: bool):
        model = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=ACTION_HORIZON,
            hist_horizon=HIST_HORIZON,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        )
        return TrainConfig(
            name=name,
            model=model,
            data=LeRobotYamStreamDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                base_config=DataConfig(
                    prompt_from_task=False,
                    hist_horizon=HIST_HORIZON,
                    hist_interval=HIST_INTERVAL,
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
    ]
