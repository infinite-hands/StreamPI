import dataclasses
import random

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import tempo
from openpi.policies import tempo_history
from openpi.training import config as _config
from openpi.training import weight_loaders

BATCH, UNITS, HORIZON, ACT_DIM = 1, 2, 4, 7


def _config_for(dtype: str = "bfloat16", **tempo_fields) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                action_horizon=HORIZON, hist_horizon=UNITS, max_token_len=8, dtype=dtype,
                                tempo_action_history_dim=ACT_DIM, **tempo_fields)


def _observation(config: pi0_config.Pi0Config, seed: int = 0) -> _model.Observation:
    rng = np.random.default_rng(seed)
    images = {key: jnp.asarray(rng.uniform(-1, 1, (BATCH, UNITS, *_model.IMAGE_RESOLUTION, 3)), jnp.float32)
              for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    pad = np.zeros((BATCH, UNITS, config.tempo_action_history_steps), bool)
    pad[:, 0, :4] = True  # the oldest unit's earliest buckets fall before the episode
    return _model.Observation(
        images=images,
        image_masks={key: jnp.ones((BATCH,), bool) for key in images},
        state=jnp.asarray(rng.normal(size=(BATCH, config.action_dim)), jnp.float32),
        tokenized_prompt=jnp.ones((BATCH, config.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((BATCH, config.max_token_len), bool),
        sam2_tokens=jnp.asarray(rng.normal(size=(BATCH, UNITS, config.tempo_sam2_num_tokens,
                                                 config.tempo_sam2_token_dim)), jnp.float32),
        action_history=jnp.asarray(rng.normal(size=(BATCH, UNITS, config.tempo_action_history_steps, ACT_DIM)),
                                   jnp.float32),
        action_history_is_pad=jnp.asarray(pad),
    )


def _shared_base(with_tempo, without_tempo):
    """Copy every base parameter of `without_tempo` into `with_tempo`, so the two differ only in TEMPO's modules."""
    graph, state = nnx.split(with_tempo)
    flat = flax.traverse_util.flatten_dict(state.to_pure_dict())
    base = flax.traverse_util.flatten_dict(nnx.state(without_tempo).to_pure_dict())
    flat.update({key: value for key, value in base.items() if key in flat})
    state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(flat))
    return nnx.merge(graph, state)


def _with_open_gates(model):
    """pi0.5's adaRMS modulation is zero-initialised, so an untrained action expert gates every residual
    branch shut and its output cannot depend on the prefix. A trained checkpoint (what TEMPO warm-starts
    from) has them open: give every all-zero LLM kernel small random weights to stand in for that."""
    graph, state = nnx.split(model)
    flat = flax.traverse_util.flatten_dict(state.to_pure_dict())
    rng = np.random.default_rng(0)
    for key, value in flat.items():
        names = tuple(map(str, key))
        if "llm" in names and np.ndim(value) >= 2 and not np.any(value):
            flat[key] = jnp.asarray(rng.normal(0, 0.02, np.shape(value)), np.asarray(value).dtype)
        elif names[-1] == "gate" or "pos_emb" in names or "mlp_out" in names:  # TEMPO's zero-init gates, opened
            flat[key] = jnp.asarray(rng.normal(0.5, 0.1, np.shape(value)), np.asarray(value).dtype)
    state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(flat))
    return nnx.merge(graph, state)


def _prefix_pass(model, obs):
    """The prefix forward pass compute_loss makes (without its random unit masking): (out, kv_cache)."""
    tokens, mask, ar_mask, *_ = model.embed_prefix(obs)
    attn_mask = model.hide_history(pi0.make_attn_mask(mask, ar_mask))
    positions = jnp.cumsum(model.position_mask(mask), axis=1) - 1
    (out, _), kv_cache = model.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions, adarms_cond=[None, None])
    return out, kv_cache


def test_sam2_fusion_is_a_no_op_at_init_and_can_learn():
    plain = _config_for().create(jax.random.key(0))
    fused_config = _config_for(tempo_sam2=True, tempo_sam2_image_key="left_wrist_0_rgb")
    fused = _shared_base(fused_config.create(jax.random.key(1)), plain)
    obs = _observation(fused_config)
    with_cue, *_ = fused.embed_prefix(obs)
    without_cue, *_ = plain.embed_prefix(obs)
    assert jnp.array_equal(with_cue, without_cue), "a zero gate must leave every token unchanged"

    block = fused.sam2_fusion
    vis, sam2 = jnp.ones((1, 5, 64)), jnp.ones((1, 7, 256))
    small = tempo.Sam2CrossAttnFusion(64, 256, 7, 8, rngs=nnx.Rngs(0))
    grads = nnx.grad(lambda module: jnp.sum(module(vis, sam2) * 1.0))(small)
    assert float(jnp.abs(grads.gate.value).max()) > 0, "the gate must receive a gradient at init (upstream's does not)"
    assert block.gate.value.shape == (1,)


def test_action_history_residual_is_zero_at_init():
    plain = _config_for().create(jax.random.key(0))
    act_config = _config_for(tempo_action_history=True)
    act = _shared_base(act_config.create(jax.random.key(1)), plain)
    obs = _observation(act_config)
    noisy, time = jnp.ones((BATCH, HORIZON, act_config.action_dim)), jnp.full((BATCH, HORIZON), 0.5)
    *_, with_history = act.embed_suffix(obs, noisy, time)
    *_, without_history = plain.embed_suffix(obs, noisy, time)
    assert jnp.array_equal(with_history, without_history), "the adaRMS residual starts at exactly zero"


def test_history_units_carry_their_own_action_tokens():
    config = _config_for(tempo_sam2=True, tempo_sam2_image_key="left_wrist_0_rgb", tempo_action_history=True)
    model = config.create(jax.random.key(0))
    plain = _config_for().create(jax.random.key(0))
    obs = _observation(config)
    tokens, mask, ar_mask, unit_size, units = model.embed_prefix(obs)
    plain_tokens, _, _, plain_unit, _ = plain.embed_prefix(obs)
    steps = config.tempo_action_history_steps
    assert units == UNITS and unit_size == plain_unit + steps
    assert tokens.shape[1] == plain_tokens.shape[1] + UNITS * steps
    oldest_history = mask[:, unit_size - steps:unit_size]
    assert not bool(oldest_history[:, :4].any()) and bool(oldest_history[:, 4:].all()), "pad buckets are masked out"
    assert not bool(ar_mask[unit_size - steps:unit_size].any()), "history attends bidirectionally within its unit"


def test_loss_and_streaming_sample_run_with_both_channels():
    config = _config_for(tempo_sam2=True, tempo_sam2_image_key="left_wrist_0_rgb", tempo_action_history=True)
    model = _with_open_gates(config.create(jax.random.key(0)))
    obs = _observation(config)
    loss = model.compute_loss(jax.random.key(1), obs, jnp.ones((BATCH, HORIZON, config.action_dim)))
    assert loss.shape == (BATCH, HORIZON) and bool(jnp.isfinite(loss).all())
    moved = dataclasses.replace(obs, sam2_tokens=obs.sam2_tokens * 5, action_history=obs.action_history * 5)
    assert not jnp.allclose(loss, model.compute_loss(jax.random.key(1), moved, jnp.ones((BATCH, HORIZON, config.action_dim)))), \
        "with trained-like gates the loss must depend on the TEMPO inputs"
    grads = nnx.grad(lambda m: jnp.mean(m.compute_loss(jax.random.key(1), obs, jnp.ones((BATCH, HORIZON, config.action_dim)))))(model)
    assert float(jnp.abs(grads.sam2_fusion.gate.value).max()) > 0
    assert float(jnp.abs(grads.action_history_tokens.proj.kernel.value).max()) > 0


    memory = {"memory_tokens": None, "memory_kv_cache": None, "memory_prefix_mask": None}
    one_unit = jax.tree.map(lambda x: x[:, -1:] if x.ndim >= 3 and x.shape[1] == UNITS else x, obs)
    for _ in range(2):  # the second call reads the first call's unit from the KV cache
        actions, memory = model.sample_actions(jax.random.key(2), one_unit, num_steps=2, memory=memory)
        assert actions.shape == (BATCH, HORIZON, config.action_dim) and bool(jnp.isfinite(actions).all())


def test_history_tokens_are_live_at_init():
    config = _config_for(tempo_action_history=True)
    model = config.create(jax.random.key(0))
    obs = _observation(config)
    assert bool(jnp.any(model.action_history_tokens(obs.action_history[:, -1]))), \
        "an all-zero history stream makes RMSNorm's backward 1/sqrt(eps) per layer: NaN on Gemma 2B"


def test_history_is_invisible_to_image_and_prompt_tokens():
    plain = _with_open_gates(_config_for("float32").create(jax.random.key(0)))
    config = _config_for("float32", tempo_action_history=True)
    act = _with_open_gates(_shared_base(config.create(jax.random.key(1)), plain))
    obs = _observation(config)
    with_history, _ = _prefix_pass(act, obs)
    without_history, _ = _prefix_pass(plain, obs)
    slots = act.history_slots(with_history.shape[1])
    assert bool(jnp.any(with_history[:, slots])), "the history tokens themselves are live once the gate opens"
    # float32 throughout; the two sequences differ in length, so reductions may differ in the last bits
    np.testing.assert_allclose(np.asarray(with_history[:, ~slots]), np.asarray(without_history), rtol=1e-5, atol=1e-5)


def test_streaming_memory_matches_the_training_pass():
    config = _config_for("float32", tempo_sam2=True, tempo_sam2_image_key="left_wrist_0_rgb", tempo_action_history=True)
    model = _with_open_gates(config.create(jax.random.key(0)))
    obs = _observation(config)
    _, full_kv = _prefix_pass(model, obs)
    memory = {"memory_tokens": None, "memory_kv_cache": None, "memory_prefix_mask": None}
    for t in range(UNITS):
        unit = jax.tree.map(lambda x, t=t: x[:, t:t + 1] if x.ndim >= 3 and x.shape[1] == UNITS else x, obs)
        _, memory = model.sample_actions(jax.random.key(2), unit, num_steps=1, memory=memory)
    # A padded bucket's row is fully masked, so its output averages whatever keys are present: it
    # differs between the two passes and nothing reads it. Every valid token must match.
    valid = np.asarray(memory["memory_prefix_mask"][0])
    for streamed, full in zip(jax.tree.leaves(memory["memory_kv_cache"]), jax.tree.leaves(full_kv), strict=True):
        assert streamed.shape == full.shape  # (layers, b, tokens, heads, head_dim)
        np.testing.assert_allclose(np.asarray(streamed, np.float32)[:, :, valid], np.asarray(full, np.float32)[:, :, valid],
                                   rtol=1e-5, atol=1e-5)


def test_action_history_is_normalized_with_the_driven_arms_state_stats():
    q01, q99 = np.arange(14, dtype=np.float32), np.arange(14, dtype=np.float32) + 2
    stats = transforms.NormStats(mean=np.zeros(14), std=np.ones(14), q01=q01, q99=q99)
    dims = tuple(range(7, 14))  # the right arm
    history = np.broadcast_to(q01[list(dims)] + 1, (UNITS, 3, 7)).astype(np.float32)  # each dim's band midpoint
    pad = np.zeros((UNITS, 3), bool)
    pad[0, 0] = True
    out = tempo_history.NormalizeActionHistory(stats, dims)({"action_history": history, "action_history_is_pad": pad})
    np.testing.assert_allclose(out["action_history"][~pad], 0.0, atol=1e-5)
    assert np.array_equal(out["action_history"][0, 0], np.zeros(7)), "a bucket wholly before the episode is zero"
    with pytest.raises(ValueError, match="state stats"):
        tempo_history.NormalizeActionHistory(None, dims)({"action_history": history, "action_history_is_pad": pad})


def test_missing_inputs_are_refused():
    config = _config_for(tempo_sam2=True, tempo_sam2_image_key="left_wrist_0_rgb", tempo_action_history=True)
    model = config.create(jax.random.key(0))
    obs = dataclasses.replace(_observation(config), sam2_tokens=None)
    with pytest.raises(ValueError, match="SAM2"):
        model.embed_prefix(obs)


def test_warm_start_from_a_checkpoint_without_tempo_modules():
    plain = nnx.state(_config_for().create(jax.random.key(0))).to_pure_dict()
    config = _config_for(tempo_sam2=True, tempo_action_history=True)
    reference = nnx.state(config.create(jax.random.key(1))).to_pure_dict()
    expected = set(flax.traverse_util.flatten_dict(reference, sep="/"))
    lora_only = weight_loaders._merge_params(plain, reference, missing_regex=r".*lora.*")
    assert set(flax.traverse_util.flatten_dict(lora_only, sep="/")) != expected, \
        "without the TEMPO regex the new modules are absent, which the trainer's tree check refuses"
    merged = weight_loaders._merge_params(
        plain, reference, missing_regex=r".*lora.*|.*(sam2_fusion|action_history_tokens|action_history_cond).*")
    flat_merged = flax.traverse_util.flatten_dict(merged, sep="/")
    assert set(flat_merged) == expected
    flat_plain = flax.traverse_util.flatten_dict(plain, sep="/")
    assert all(np.array_equal(flat_merged[key], value) for key, value in flat_plain.items()), \
        "the checkpoint's own weights load unchanged"


def test_jitter_offsets_and_history_loader_agree(tmp_path):
    hist_key, steps, per_bucket, interval, length = "observation.images.cam_left_wrist", 2, 3, 4, 30
    tokens = np.arange(length, dtype=np.float32)[:, None, None] * np.ones((1, 2, 3), np.float32)
    actions = np.arange(length, dtype=np.float32)[:, None] * np.ones((1, 14), np.float32)
    for sub, array in ((tempo_history.TOKENS_DIR, tokens), (tempo_history.ACTIONS_DIR, actions)):
        (tmp_path / sub).mkdir()
        np.save(tmp_path / sub / tempo_history.EPISODE_FILE.format(5), array)
    random.seed(0)
    jitter = transforms.TemporalJitter((-2, -1, 0, 1, 2), interval, UNITS, [hist_key], True, record_offsets=True)
    loader = tempo_history.LoadTempoHistory(str(tmp_path), hist_key, tuple(range(7)), steps, per_bucket)
    window = (UNITS - 1) * interval + 1
    for frame in (0, 3, 17, length - 1):
        # the window LeRobot returns: frame - (window-1) .. frame, clamped at the episode start
        seq = np.clip(np.arange(frame - window + 1, frame + 1), 0, None).astype(np.float32)
        sample = jitter({hist_key: seq, "episode_index": 5, "frame_index": frame})
        loaded = loader(dict(sample))
        kept_frames = sample[hist_key]  # the frame each kept image came from
        assert np.array_equal(loaded["sam2_tokens"][:, 0, 0], kept_frames), (frame, kept_frames)
        for unit, kept in enumerate(kept_frames.astype(int)):
            history, pad = tempo_history.action_history(actions[:, :7], kept, steps, per_bucket)
            assert np.array_equal(loaded["action_history"][unit], history) and np.array_equal(loaded["action_history_is_pad"][unit], pad)
        assert hist_key + transforms.HIST_OFFSETS_SUFFIX not in loaded


@pytest.mark.parametrize("arm", ["left", "right"])
def test_yam_tempo_pair_differs_only_in_the_channels(arm):
    tempo_config = _config.get_config(f"pi05_yam_stream5_i20_bagging_{arm}_real_tempo")
    control = _config.get_config(f"pi05_yam_stream5_i20_bagging_{arm}_real_tempo_ctrl")
    assert tempo_config.model.tempo_sam2 and tempo_config.model.tempo_action_history
    assert tempo_config.model.tempo_sam2_image_key == f"{arm}_wrist_0_rgb" and tempo_config.model.tempo_action_history_dim == 7
    assert not control.model.tempo_sam2 and not control.model.tempo_action_history
    assert tempo_config.weight_loader.params_path == control.weight_loader.params_path
    assert tempo_config.num_train_steps == control.num_train_steps == 5_000
    assert tempo_config.lr_schedule == control.lr_schedule
    assert tempo_config.data.assets == control.data.assets and tempo_config.data.assets.assets_dir.endswith(f"bagging_{arm}_real")
    assert tempo_config.data.base_config.tempo_hist_key == f"observation.images.cam_{arm}_wrist"
    assert control.data.base_config.tempo_cache_dir is None
