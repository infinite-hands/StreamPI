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
# LIT (Latent Interface Training) on the left-real recipe. The driven arm is the left one: dims 0-6 of the 14-dim
# state, the complement of the right arm the recipe holds still (RIGHT_ARM_DIMS) and exactly the dims the recipe
# leaves visible to the model (LEFT_ARM_DIMS as active_state_dims), so the pose loss and the goal encoder never see the
# held arm's drift. Dims 14-31 of the 32-wide model state are zero padding.
LIT_BASE_CONFIG = "pi05_yam_stream5_i20_bagging_left_real"
LIT_GOAL_DIMS = LEFT_ARM_DIMS
# The four LIT rows read the PRODUCTION base recipe's norm statistics, the file the launcher's stats pass already wrote
# for LIT_BASE_CONFIG (/checkpoints/assets/<config name>/<repo id>/norm_stats.json on the Modal volume), so the control
# and the LIT arms normalise exactly like the production baseline and no row depends on another having run first. The
# pin follows ih/openpi's convention: assets_dir is the volume's /checkpoints/assets/<config>, asset_id the repo.
LIT_NORM_ASSETS_DIR = f"/checkpoints/assets/{LIT_BASE_CONFIG}"
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
    from openpi.training.config import AssetsConfig, DataConfig, LeRobotAgilexDataConfig, TrainConfig

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
                      held_action_dims: tuple[int, ...] | None = None):
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
                active_image_keys=active_image_keys,
                active_state_dims=active_state_dims,
                held_action_dims=held_action_dims,
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

    def lit_config(suffix: str, *, lit: str, lora: bool, weight_loader, ema_decay: float | None):
        """A LIT row: the left-real recipe with the model switched to `lit`. Everything else (dataset, prompt, cameras,
        horizons, cadence, norm statistics) is the recipe's; the full fine-tune rows run batch 32 over four devices and
        the LoRA row batch 16 on one, as the recipe's own full and LoRA rows do.

        The hyperparameters are the BASE recipe's, not LIT's own: the lr schedule (cosine, peak 2.5e-5, 1,000 warmup
        steps, floor 2.5e-6 at step 20,000), AdamW (weight decay 1e-10, gradient clipping 1.0), 20,000 steps and the
        batch, except what a row sets itself (ema_decay, weight loader, freeze filter, the lit fields). LIT's own pi0.5
        release differs (reader reports of its torch launchers on LIBERO, not re-verified here): peak lr 1e-4, warmup
        4,000 steps for stage 1 and 5,000 for stage 2, weight decay 0.01; the README's "backbone 1e-5" holds only for
        its MolmoAct2-LIBERO run (its pi0.5 stage 2 trains the whole model at 1e-4). None of those was adopted: the
        right values for this data are measure-first. The step budget (the control's 20,000 against lit1's 20,000
        plus lit2's 20,000) is a launch-time decision via --num-train-steps; the cosine's decay_steps is its own field
        and does not follow it."""
        row = stream_config(
            f"{LIT_BASE_CONFIG}_{suffix}", BAGGING_LEFT_REAL_REPO_ID, BAGGING_PROMPT, lora=lora,
            hist_interval=HIST_INTERVAL_WIDE,
            active_image_keys=frozenset({"left_wrist_0_rgb"}),
            active_state_dims=LEFT_ARM_DIMS,
            held_action_dims=RIGHT_ARM_DIMS,
        )
        model = dataclasses.replace(row.model, lit=lit, lit_goal_dims=LIT_GOAL_DIMS if lit != "off" else ())
        return dataclasses.replace(
            row,
            model=model,
            data=dataclasses.replace(
                row.data,
                assets=AssetsConfig(assets_dir=LIT_NORM_ASSETS_DIR, asset_id=BAGGING_LEFT_REAL_REPO_ID),
            ),
            weight_loader=weight_loader,
            freeze_filter=model.get_freeze_filter(),
            ema_decay=ema_decay,
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
        # LIT (Latent Interface Training) on that recipe. In every row the lr schedule, warmup, weight decay, clipping,
        # steps and batch are the BASE recipe's (see lit_config), not LIT's own pi0.5 release (peak lr 1e-4, warmup
        # 4,000 stage 1 / 5,000 stage 2, weight decay 0.01 per reader reports; "backbone 1e-5" is MolmoAct2-LIBERO
        # only): measure first. Step budgets are a launch-time decision via --num-train-steps.
        #  _litctl   the matched control: lit off, FULL fine-tune, PaliGemma init with a random expert (LIT's tested
        #            protocol). Base-recipe hyperparameters, batch 32, ema 0.99, 20,000 steps unless the launch sets
        #            the matched budget (the LIT arms spend 20,000 + 20,000).
        #  _lit1     stage 1: no images reach the model, expert + goal encoder train, backbone frozen. The loader still
        #            decodes all three cameras (compute_loss needs the images dict) and the cameras are masked as in
        #            the recipe. No EMA: the backbone is frozen. Base-recipe hyperparameters, batch 32, 20,000 steps
        #            unless the launch sets another.
        #  _lit2     stage 2, full fine-tune, from a stage-1 checkpoint: pass --weight-loader.params-path=<the stage-1
        #            run's params directory>. Base-recipe hyperparameters (LIT's stage 2 uses its own, longer warmup),
        #            batch 32, ema 0.99, 20,000 steps unless the launch sets another.
        #  _litlite  stage 2 with LoRA on the backbone and the expert and full lit_* modules, warm start from pi05_base:
        #            a labelled deviation (LIT's tested protocol is the full fine-tune). Base-recipe hyperparameters
        #            (the recipe's LoRA row: batch 16, no EMA), 20,000 steps unless the launch sets another.
        # All four read the base recipe's norm statistics (LIT_NORM_ASSETS_DIR above): compute_norm_stats is never run
        # for a _lit* name, and the file for LIT_BASE_CONFIG must exist before any of them starts.
        # lit_groups=6 divides the 18 layers (3 per group); lit_goal_dims is the left arm, dims 0-6.
        lit_config("litctl", lit="off", lora=False, weight_loader=weight_loaders.PaliGemmaWeightLoader(),
                   ema_decay=0.99),
        lit_config("lit1", lit="stage1", lora=False, weight_loader=weight_loaders.PaliGemmaWeightLoader(),
                   ema_decay=None),
        lit_config("lit2", lit="stage2", lora=False, weight_loader=weight_loaders.LitStage1WeightLoader(),
                   ema_decay=0.99),
        lit_config("litlite", lit="stage2", lora=True,
                   weight_loader=weight_loaders.CheckpointWeightLoader(
                       PI05_BASE_PARAMS, missing_regex=weight_loaders.LIT_MISSING_REGEX),
                   ema_decay=None),
    ]
