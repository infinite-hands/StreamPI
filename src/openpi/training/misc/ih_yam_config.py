"""Infinite Hands YAM configurations for StreamPI (pi0.5 + streaming temporal KV-cache memory)."""

import dataclasses

from openpi.models.model import IMAGE_KEYS
import openpi.models.pi0_config as pi0_config
from openpi.policies.agilex_policy import AgilexInputs
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
# The reference implementation's offset range (mit-han-lab/vlash, examples/train/*/async*.yaml:
# max_delay_steps: 8), for a deployment that fixes its lead at this many rows or fewer. The full-window
# range (hist_interval - 1) made the previous command so close to the next target that the left-real
# fine-tune followed the wrist image about a third as much as its non-vlash twin did, offline.
VLASH_REFERENCE_MAX_OFFSET = 8
# The YAM LeRobot layout: three cameras, 14-dim state/action [L j0..5, L grip, R j0..5, R grip].
YAM_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def yam_repack(cameras: tuple[str, ...] = YAM_CAMERAS) -> _transforms.Group:
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {camera: f"observation.images.{camera}" for camera in cameras},
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )


YAM_REPACK = yam_repack()


def get_ih_yam_configs():
    # Deferred: config.py imports this module while building its registry.
    from openpi.training.config import DataConfig, LeRobotAgilexDataConfig, TrainConfig

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
                # pi0.5's own default (the AgileX base config forces z-score). The YAM left gripper never
                # moves in the corpus, so its std is ~1e-3 and z-scoring turns its noise into a loss of
                # tens of thousands; the quantile band is floored by the stats writer instead.
                use_quantile_norm=True,
            )

    def stream_config(name: str, repo_id: str, prompt: str, *, lora: bool,
                      hist_interval: int = HIST_INTERVAL,
                      active_image_keys: frozenset[str] | None = None,
                      active_state_dims: tuple[int, ...] | None = None,
                      held_action_dims: tuple[int, ...] | None = None,
                      vlash_max_offset: int = 0, vlash_branches: int = 0, state_cond: bool = False,
                      vlash_state_source: str = "action",
                      encode_only_active_cameras: bool = False):
        # A masked camera's tokens are padding, so a recipe may skip decoding and encoding them: the same
        # model, one SigLIP pass per frame instead of three and a third of the prefix. Opt-in per config
        # because it changes the model's input layout (not its weights).
        if encode_only_active_cameras and active_image_keys is None:
            raise ValueError(f"{name}: encode_only_active_cameras needs active_image_keys")
        cameras = {image_key: camera for camera, image_key in AgilexInputs.IMAGE_KEY_BY_CAMERA.items()}
        image_keys = tuple(key for key in IMAGE_KEYS if not encode_only_active_cameras or key in active_image_keys)
        kept_cameras = tuple(cameras[key] for key in image_keys)
        model = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=ACTION_HORIZON,
            hist_horizon=HIST_HORIZON,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
            # With the state out of the prompt, the single-arm recipe's state mask moves to the conditioning.
            state_cond=state_cond,
            state_cond_dims=active_state_dims if state_cond else None,
            vlash_branches=vlash_branches,
            image_keys=image_keys,
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
                repack_transforms=yam_repack(kept_cameras),
                hist_sequence_keys=tuple(f"observation.images.{camera}" for camera in kept_cameras),
                base_config=DataConfig(
                    prompt_from_task=False,
                    hist_horizon=HIST_HORIZON,
                    hist_interval=hist_interval,
                    enable_jitter=True,
                    vlash_max_offset=vlash_max_offset,
                    vlash_branches=vlash_branches,
                    vlash_state_source=vlash_state_source,
                    decode_only_hist_cameras=encode_only_active_cameras,
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

    recipes = [
        dict(name="pi05_yam_stream5_bagging", repo_id=BAGGING_REPO_ID, prompt=BAGGING_PROMPT, lora=True),
        dict(name="pi05_yam_stream5_bagging_full", repo_id=BAGGING_REPO_ID, prompt=BAGGING_PROMPT, lora=False),
        dict(name="pi05_yam_stream5_firsttry", repo_id=FIRSTTRY_REPO_ID, prompt=BAGGING_PROMPT, lora=True),
        dict(name="pi05_yam_stream5_firsttry_full", repo_id=FIRSTTRY_REPO_ID, prompt=BAGGING_PROMPT, lora=False),
        # The same four at the wider cadence; `_i20` is the only thing that differs.
        dict(name="pi05_yam_stream5_i20_bagging", repo_id=BAGGING_REPO_ID, prompt=BAGGING_PROMPT, lora=True,
             hist_interval=HIST_INTERVAL_WIDE),
        dict(name="pi05_yam_stream5_i20_bagging_full", repo_id=BAGGING_REPO_ID, prompt=BAGGING_PROMPT, lora=False,
             hist_interval=HIST_INTERVAL_WIDE),
        dict(name="pi05_yam_stream5_i20_firsttry", repo_id=FIRSTTRY_REPO_ID, prompt=BAGGING_PROMPT, lora=True,
             hist_interval=HIST_INTERVAL_WIDE),
        dict(name="pi05_yam_stream5_i20_firsttry_full", repo_id=FIRSTTRY_REPO_ID, prompt=BAGGING_PROMPT, lora=False,
             hist_interval=HIST_INTERVAL_WIDE),
        # Real (not mirrored) native left-arm teleop: wrist camera only, the right arm's state hidden
        # from the model, and the right arm trained to hold still -- it was held in gravity comp, so
        # its recorded drift is not behaviour to imitate.
        dict(name="pi05_yam_stream5_i20_bagging_left_real", repo_id=BAGGING_LEFT_REAL_REPO_ID,
             prompt=BAGGING_PROMPT, lora=True, hist_interval=HIST_INTERVAL_WIDE,
             active_image_keys=frozenset({"left_wrist_0_rgb"}),
             active_state_dims=LEFT_ARM_DIMS,
             held_action_dims=RIGHT_ARM_DIMS),
    ]

    def vlash_twin(recipe: dict) -> dict:
        # The offset covers every lead the deploy loop can ask for at this cadence: it plays
        # hist_interval rows per call, so a chunk is asked for at most hist_interval - 1 rows ahead.
        # A single-camera recipe also stops decoding and encoding the cameras it masks.
        interval = recipe.get("hist_interval", HIST_INTERVAL)
        return {**recipe, "name": recipe["name"] + "_vlash", "vlash_max_offset": interval - 1,
                "encode_only_active_cameras": recipe.get("active_image_keys") is not None}

    def vlash_reference_twin(recipe: dict) -> dict:
        # The `_vlash` twin with the reference's offset range instead of the whole window: deployed at a
        # fixed lead of at most VLASH_REFERENCE_MAX_OFFSET rows (rollout --vlash-lead).
        return {**vlash_twin(recipe), "name": recipe["name"] + "_vlash8", "vlash_max_offset": VLASH_REFERENCE_MAX_OFFSET}

    def vlash_measured_twin(recipe: dict) -> dict:
        # The `_vlash8` twin with the MEASURED state at t + delta instead of the previous command: the
        # command is the next target one row early, a shortcut that let the left-real fine-tunes follow
        # the wrist image a third to a half as much as run3 offline; the measured pose lags it by ~3-4 rows
        # and carries the gripper's real opening (a part in hand reads ~0.23, the command 0).
        return {**vlash_reference_twin(recipe), "name": recipe["name"] + "_vlash8m", "vlash_state_source": "state"}

    def vlash_packed_measured_twin(recipe: dict) -> dict:
        # The `_vlash8m` twin trained the paper's shared-observation way: every offset 0..8 as a branch behind
        # one observation, the measured state as adaRMS conditioning instead of prompt text (the reference's
        # pi0.5 layout, state_cond). Serves one state per call like any other config.
        return {**vlash_measured_twin(recipe), "name": recipe["name"] + "_vlash8mp",
                "vlash_branches": VLASH_REFERENCE_MAX_OFFSET + 1, "state_cond": True}

    def vlash_packed_twin(recipe: dict) -> dict:
        # The paper's shared-observation training: every offset 0..max as one branch behind one
        # observation, the state as adaRMS conditioning instead of prompt text. A different model
        # (state_cond) and prompt from the _vlash twin, so a different checkpoint and serve config.
        return {**vlash_twin(recipe), "name": recipe["name"] + "_vlashp",
                "vlash_branches": recipe.get("hist_interval", HIST_INTERVAL), "state_cond": True}

    return [stream_config(**recipe) for recipe in
            recipes + [vlash_twin(recipe) for recipe in recipes] + [vlash_reference_twin(recipe) for recipe in recipes]
            + [vlash_measured_twin(recipe) for recipe in recipes] + [vlash_packed_measured_twin(recipe) for recipe in recipes]
            + [vlash_packed_twin(recipe) for recipe in recipes]]
