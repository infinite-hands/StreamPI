"""The LIT data pipeline: the goal split, the dataset's delta_timestamps, the forwarding through AgilexInputs, the
normalization alias, the padding, the serve-side stats, and what a stage-1 batch carries.

No model is built here. The real tokenizer (cached under ~/.cache/openpi) and the real transforms of the YAM left-arm
recipe run on synthetic samples shaped like what LeRobotDataset delivers (state (2, 14) at [t, t + action_horizon], its
`_is_pad` flags, three cameras of `hist_horizon` frames, torch float images [0, 1]); the query/pad semantics are the
vendored library's own (`LeRobotDataset._get_query_indices`), not a copy of them.
"""

import dataclasses

import numpy as np
import pytest
import torch

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import agilex_policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

_LeRobotDataset = lerobot_dataset.LeRobotDataset  # the tests below replace the module's attribute with fakes

H = 6  # the action horizon of the synthetic episodes (30 in the real recipe)
HIST, INTERVAL = 2, 3
STATE_DIM = 14
ACTION_DIM = 32
GOAL_DIMS = tuple(range(7))
HELD_DIMS = tuple(range(7, 14))
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
PROMPT = "place one part in the bag"
KEY = _data_loader.LIT_STATE_KEY
EPISODE_ENDS = (10, 25)  # two episodes: frames [0, 10) and [10, 25)


def _model_config(lit: str = "off", **overrides) -> pi0_config.Pi0Config:
    kwargs = {"pi05": True, "action_horizon": H, "hist_horizon": HIST, **overrides}
    if lit != "off":
        kwargs |= {"lit": lit, "lit_goal_dims": GOAL_DIMS, "action_dim": ACTION_DIM}
    return pi0_config.Pi0Config(**kwargs)


def _states_table() -> np.ndarray:
    """One distinct state per frame: frame i is i + 0.01 * dim, so any off-by-one frame is visible."""
    return np.arange(EPISODE_ENDS[-1], dtype=np.float32)[:, None] + 0.01 * np.arange(STATE_DIM, dtype=np.float32)


def _episode_of(frame: int) -> int:
    return 0 if frame < EPISODE_ENDS[0] else 1


def _lerobot_query(frame: int) -> dict:
    """The state fields LeRobotDataset.__getitem__ gives for `frame` when KEY is fetched at [0, H] frames: the
    vendored library's own index/padding code, on a dataset object that has only the episode boundaries."""
    dataset = object.__new__(_LeRobotDataset)
    dataset.episode_data_index = {
        "from": torch.tensor([0, EPISODE_ENDS[0]]),
        "to": torch.tensor(EPISODE_ENDS),
    }
    dataset.delta_indices = {KEY: [0, H]}
    indices, padding = dataset._get_query_indices(frame, _episode_of(frame))  # noqa: SLF001
    return {KEY: torch.as_tensor(_states_table()[indices[KEY]]), **padding}


def _images(seed: int = 0) -> dict:
    """What the hist camera columns hold after TemporalJitter: (hist, channels, height, width) floats in [0, 1]."""
    rng = np.random.default_rng(seed)
    return {
        f"observation.images.{camera}": torch.as_tensor(rng.uniform(size=(HIST, 3, 24, 32)).astype(np.float32))
        for camera in CAMERAS
    }


def _raw_sample(frame: int, *, split: bool = True, goal_rows: bool = True) -> dict:
    """A sample as the dataset hands it to the transform chain: the LIT dataset after SplitLitGoal (`split`), or the
    stock one (`goal_rows` False: the state at the sample's frame only, no pad flag)."""
    sample = {
        **_images(frame),
        "action": torch.as_tensor(np.tile(_states_table()[frame], (H, 1)) + 0.5),
    }
    if goal_rows:
        sample |= _lerobot_query(frame)
        return _transforms.SplitLitGoal(KEY)(sample) if split else sample
    sample[KEY] = torch.as_tensor(_states_table()[frame])
    return sample


def _norm_stats() -> dict:
    lo, hi = np.full(STATE_DIM, -2.0), np.full(STATE_DIM, 40.0)
    state = _normalize.NormStats(mean=np.zeros(STATE_DIM), std=np.ones(STATE_DIM), q01=lo, q99=hi)
    actions = _normalize.NormStats(mean=np.zeros(STATE_DIM), std=np.ones(STATE_DIM), q01=lo - 1, q99=hi + 1)
    return {"state": state, "actions": actions}


def _quantile(x, stats):
    return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


def _data_config(model_config, tmp_path, **factory_overrides) -> _config.DataConfig:
    """The data config of the YAM left-arm recipe on `model_config`, with synthetic norm stats (and the quantile norm
    the YAM factory of ih_yam_config forces over the AgileX base config's z-score)."""
    from openpi.training.misc import ih_yam_config

    factory = _config.LeRobotAgilexDataConfig(
        repo_id="local/lit_synthetic",
        default_prompt=PROMPT,
        repack_transforms=ih_yam_config.YAM_REPACK,
        active_image_keys=frozenset({"left_wrist_0_rgb"}),
        active_state_dims=GOAL_DIMS,
        held_action_dims=HELD_DIMS,
        base_config=_config.DataConfig(hist_horizon=HIST, hist_interval=INTERVAL),
        **factory_overrides,
    )
    return dataclasses.replace(factory.create(tmp_path, model_config), norm_stats=_norm_stats(), use_quantile_norm=True)


def _transformed(data_config, samples) -> list:
    """The samples through the real train-time chain (repack, data transforms, Normalize, model transforms)."""
    dataset = _data_loader.transform_dataset(list(samples), data_config)
    return [dataset[i] for i in range(len(dataset))]


# ---- the split ----


@pytest.mark.parametrize("frame", range(EPISODE_ENDS[-1]))
def test_goal_is_the_state_at_t_plus_h_and_the_tail_is_flagged(frame):
    table = _states_table()
    end = EPISODE_ENDS[_episode_of(frame)]
    item = _transforms.SplitLitGoal(KEY)(_raw_sample(frame, split=False))
    np.testing.assert_array_equal(item[KEY], table[frame])
    assert item[KEY].shape == (STATE_DIM,)
    real = frame + H < end
    assert item[_transforms.LIT_GOAL_MASK_KEY] == np.bool_(real)
    assert item[_transforms.LIT_GOAL_MASK_KEY].dtype == np.bool_
    # a clamped goal is the episode's last state (and flagged); a real one is exactly t + H
    np.testing.assert_array_equal(item[_transforms.LIT_GOAL_KEY], table[frame + H if real else end - 1])
    assert f"{KEY}_is_pad" not in item


def test_the_clamped_tail_of_each_episode_is_exactly_the_last_h_frames():
    flagged = [
        frame
        for frame in range(EPISODE_ENDS[-1])
        if not _raw_sample(frame)[_transforms.LIT_GOAL_MASK_KEY]
    ]
    assert flagged == [*range(EPISODE_ENDS[0] - H, EPISODE_ENDS[0]), *range(EPISODE_ENDS[1] - H, EPISODE_ENDS[1])]


def test_a_sample_that_was_not_fetched_at_two_frames_is_refused():
    split = _transforms.SplitLitGoal(KEY)
    sample = _raw_sample(0, split=False, goal_rows=True)
    with pytest.raises(ValueError, match="is_pad"):
        split({k: v for k, v in sample.items() if k != f"{KEY}_is_pad"})
    with pytest.raises(ValueError, match="two frames"):
        split({**sample, KEY: sample[KEY][0], f"{KEY}_is_pad": sample[f"{KEY}_is_pad"]})


# ---- delta_timestamps ----


class _Meta:
    fps = 30
    tasks = {}  # noqa: RUF012

    def __init__(self, repo_id):
        self.repo_id = repo_id


def _capture(monkeypatch, fps=30):
    seen = {}

    class Meta(_Meta):
        pass

    Meta.fps = fps

    class Dataset(list):
        def __init__(self, repo_id, delta_timestamps=None, **kwargs):
            super().__init__()
            seen["delta_timestamps"] = delta_timestamps
            seen["repo_id"] = repo_id

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", Meta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", Dataset)
    return seen


def _stock_delta_timestamps(data_config, fps) -> dict:
    """What create_torch_dataset fetched before LIT existed (the pin's expression)."""
    stock = {key: [t / fps for t in range(H)] for key in data_config.action_sequence_keys}
    for key in data_config.hist_sequence_keys:
        stock[key] = [-t / fps for t in range((HIST - 1) * INTERVAL, -1, -1)]
    return stock


@pytest.mark.parametrize("fps", [30, 50])
def test_lit_off_fetches_exactly_what_the_stock_pipeline_fetched(monkeypatch, tmp_path, fps):
    seen = _capture(monkeypatch, fps)
    config = _model_config()
    data_config = _data_config(config, tmp_path)
    dataset = _data_loader.create_torch_dataset(data_config, H, config)
    assert seen["delta_timestamps"] == _stock_delta_timestamps(data_config, fps)
    assert KEY not in seen["delta_timestamps"]
    # one jitter stage over the raw dataset and nothing after it: TransformedDataset(TemporalJitter)
    assert isinstance(dataset._dataset, list)  # noqa: SLF001
    assert [type(t).__name__ for t in dataset._transform.transforms] == ["TemporalJitter"]  # noqa: SLF001


@pytest.mark.parametrize("fps", [30, 50])
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_lit_fetches_the_state_now_and_action_horizon_frames_ahead(monkeypatch, tmp_path, fps, stage):
    seen = _capture(monkeypatch, fps)
    config = _model_config(stage)
    data_config = _data_config(config, tmp_path)
    _data_loader.create_torch_dataset(data_config, H, config)
    expected = {**_stock_delta_timestamps(data_config, fps), KEY: [0.0, H / fps]}
    assert seen["delta_timestamps"] == expected
    # at the frame rate the rows land on whole frames: 0 and exactly action_horizon
    assert [round(d * fps) for d in seen["delta_timestamps"][KEY]] == [0, H]


def test_lit_refuses_vlash_temporal_offsets(monkeypatch, tmp_path):
    """ih/vlash shifts the state and the actions of a sample by a random offset; the goal would not follow. The
    fields do not exist at this pin: a DataConfig subclass carries them, as vlash does."""

    @dataclasses.dataclass(frozen=True)
    class WithVlash(_config.DataConfig):
        vlash_max_offset: int = 0

    def with_vlash(base, **fields):
        return WithVlash(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)}, **fields)

    _capture(monkeypatch)
    for lit in ("stage1", "stage2"):
        config = _model_config(lit)
        base = _data_config(config, tmp_path)
        _data_loader.create_torch_dataset(with_vlash(base), H, config)
        with pytest.raises(ValueError, match="vlash"):
            _data_loader.create_torch_dataset(with_vlash(base, vlash_max_offset=4), H, config)
    stock = _model_config()
    _data_loader.create_torch_dataset(with_vlash(_data_config(stock, tmp_path), vlash_max_offset=4), H, stock)


def test_create_torch_dataset_hands_the_transform_chain_a_split_sample(monkeypatch, tmp_path):
    """End to end through create_torch_dataset: the dataset's own two-row state comes out as state + goal."""
    config = _model_config("stage2")
    data_config = _data_config(config, tmp_path)

    class Dataset(list):
        def __init__(self, repo_id, delta_timestamps=None, **kwargs):
            super().__init__()

        def __getitem__(self, index):
            frame = index
            history = {k: v for k, v in _images(frame).items()}
            # TemporalJitter reads the frames of the history window: (T, c, h, w) with T = (hist - 1) * interval + 1
            window = {k: v[:1].repeat(((HIST - 1) * INTERVAL) + 1, 1, 1, 1) for k, v in history.items()}
            return {**window, "action": torch.zeros(H, STATE_DIM), **_lerobot_query(frame)}

        def __len__(self):
            return EPISODE_ENDS[-1]

    class Meta(_Meta):
        pass

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", Meta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", Dataset)
    dataset = _data_loader.create_torch_dataset(data_config, H, config)
    table = _states_table()
    for frame in (0, 3, 8, 12, 24):
        item = dataset[frame]
        np.testing.assert_array_equal(item[KEY], table[frame])
        np.testing.assert_array_equal(
            item[_transforms.LIT_GOAL_KEY], table[min(frame + H, EPISODE_ENDS[_episode_of(frame)] - 1)]
        )
        assert item[_transforms.LIT_GOAL_MASK_KEY] == (frame + H < EPISODE_ENDS[_episode_of(frame)])


# ---- the transform chain ----


def test_agilex_inputs_forwards_the_goal_and_its_mask_and_still_drops_unknown_keys():
    sample = {
        "state": np.arange(STATE_DIM, dtype=np.float32),
        "images": {camera: np.zeros((HIST, 3, 8, 8), np.float32) for camera in CAMERAS},
        _transforms.LIT_GOAL_KEY: np.arange(STATE_DIM, dtype=np.float32) + 100,
        _transforms.LIT_GOAL_MASK_KEY: np.True_,
        "something_else": np.zeros(3),
    }
    out = agilex_policy.AgilexInputs()(dict(sample))
    np.testing.assert_array_equal(out[_transforms.LIT_GOAL_KEY], sample[_transforms.LIT_GOAL_KEY])
    assert out[_transforms.LIT_GOAL_MASK_KEY] == np.True_
    assert "something_else" not in out
    plain = agilex_policy.AgilexInputs()({k: v for k, v in sample.items() if not k.startswith("lit_")})
    assert not any(k.startswith("lit_") for k in plain)


def test_carry_lit_goal_adds_the_keys_to_every_repack_and_nothing_else():
    from openpi.training.misc import ih_yam_config

    carried = _transforms.carry_lit_goal(ih_yam_config.YAM_REPACK)
    (repack,) = carried.inputs
    assert repack.structure == {
        **ih_yam_config.YAM_REPACK.inputs[0].structure,
        "lit_goal": "lit_goal",
        "lit_goal_mask": "lit_goal_mask",
    }
    other = _transforms.Group(
        inputs=[_transforms.InjectDefaultPrompt("x")], outputs=[_transforms.InjectDefaultPrompt("y")]
    )
    assert _transforms.carry_lit_goal(other) == other
    # the original group is untouched
    assert "lit_goal" not in ih_yam_config.YAM_REPACK.inputs[0].structure


def test_lit_ee6d_is_refused(tmp_path):
    with pytest.raises(ValueError, match="ee6d"):
        _data_config(_model_config("stage2"), tmp_path, use_ee6d=True)


def test_the_goal_is_normalized_with_the_state_statistics_and_the_mask_is_untouched(tmp_path):
    config = _model_config("stage2")
    data_config = _data_config(config, tmp_path)
    assert data_config.norm_aliases == {"lit_goal": "state"}
    assert "lit_goal" not in data_config.norm_stats, "the alias is never stored: the saved assets stay the control's"
    stats = data_config.norm_stats["state"]
    table = _states_table()
    frame = 2
    (item,) = _transformed(data_config, [_raw_sample(frame)])
    # state and goal are both quantile-normalized with the state statistics, then padded to the model width
    np.testing.assert_allclose(item["state"][:STATE_DIM], _quantile(table[frame], stats), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(item["lit_goal"][:STATE_DIM], _quantile(table[frame + H], stats), rtol=1e-5, atol=1e-6)
    assert item["state"].shape == item["lit_goal"].shape == (ACTION_DIM,)
    assert not np.any(item["state"][STATE_DIM:]) and not np.any(item["lit_goal"][STATE_DIM:])
    assert item["lit_goal_mask"] == np.True_ and item["lit_goal_mask"].dtype == np.bool_
    # the action statistics are not the state's: the goal under them would differ
    under_actions = _quantile(table[frame + H], data_config.norm_stats["actions"])
    assert not np.allclose(under_actions, _quantile(table[frame + H], stats))
    # a clamped goal keeps its (clamped) value and is flagged
    (tail,) = _transformed(data_config, [_raw_sample(EPISODE_ENDS[0] - 1)])
    assert tail["lit_goal_mask"] == np.False_


def test_normalize_aliases_need_the_stats_they_name(tmp_path):
    config = _model_config("stage2")
    data_config = dataclasses.replace(_data_config(config, tmp_path), norm_stats={"actions": _norm_stats()["actions"]})
    with pytest.raises(ValueError, match="norm_aliases"):
        _transformed(data_config, [_raw_sample(0)])


def test_state_and_prompt_tokens_never_see_the_goal(tmp_path):
    config = _model_config("stage2")
    stock_config = _model_config(action_dim=ACTION_DIM)
    lit_data, stock_data = _data_config(config, tmp_path), _data_config(stock_config, tmp_path)
    frame = 2
    (with_goal,) = _transformed(lit_data, [_raw_sample(frame)])
    (without_goal,) = _transformed(stock_data, [_raw_sample(frame, goal_rows=False)])
    assert "lit_goal" not in without_goal
    for key in ("tokenized_prompt", "tokenized_prompt_mask", "state", "actions"):
        np.testing.assert_array_equal(with_goal[key], without_goal[key], err_msg=key)
    # and moving only the goal (the state at t + H) moves nothing the model reads from the prompt
    moved = _raw_sample(frame)
    moved[_transforms.LIT_GOAL_KEY] = moved[_transforms.LIT_GOAL_KEY] + 7.0
    (moved_out,) = _transformed(lit_data, [moved])
    for key in ("tokenized_prompt", "tokenized_prompt_mask", "state", "actions"):
        np.testing.assert_array_equal(moved_out[key], with_goal[key], err_msg=key)
    assert not np.allclose(moved_out["lit_goal"], with_goal["lit_goal"])
    # the prompt does carry the current state (left arm only: the right arm's dims are masked to the midpoint)
    other_state = _raw_sample(frame + 1)
    (other,) = _transformed(lit_data, [other_state])
    assert not np.array_equal(other["tokenized_prompt"], with_goal["tokenized_prompt"])


def test_a_stock_config_carries_no_lit_key_and_no_alias(tmp_path):
    config = _model_config(action_dim=ACTION_DIM)
    data_config = _data_config(config, tmp_path)
    assert data_config.norm_aliases == {}
    assert "lit_goal" not in data_config.repack_transforms.inputs[0].structure
    (item,) = _transformed(data_config, [_raw_sample(3, goal_rows=False)])
    assert sorted(item) == sorted(
        ["image", "image_mask", "state", "actions", "tokenized_prompt", "tokenized_prompt_mask"]
    )
    # the stats passed to Normalize are the stock ones, the very same object
    stats_for_normalize = _data_loader._with_aliases(data_config.norm_stats, data_config.norm_aliases)  # noqa: SLF001
    assert stats_for_normalize is data_config.norm_stats


def test_pad_states_and_actions_pads_the_goal_like_the_state_and_leaves_the_mask(tmp_path):
    pad = _transforms.PadStatesAndActions(ACTION_DIM)
    data = {
        "state": np.ones(STATE_DIM),
        "actions": np.ones((H, STATE_DIM)),
        "lit_goal": np.full(STATE_DIM, 2.0),
        "lit_goal_mask": np.True_,
    }
    out = pad(dict(data))
    assert out["state"].shape == out["lit_goal"].shape == (ACTION_DIM,)
    np.testing.assert_array_equal(out["lit_goal"][:STATE_DIM], 2.0)
    assert not np.any(out["lit_goal"][STATE_DIM:])
    assert out["lit_goal_mask"] == np.True_
    assert "lit_goal" not in pad({"state": np.ones(STATE_DIM)})


def test_output_norm_stats_drops_the_input_only_goal_key():
    stats = _norm_stats()
    with_goal = {**stats, "lit_goal": stats["state"]}
    outputs = {"state": np.zeros(ACTION_DIM), "actions": np.zeros((H, ACTION_DIM))}
    with pytest.raises(ValueError, match="lit_goal"):
        _transforms.Unnormalize(with_goal, use_quantiles=True)(dict(outputs))
    restored = _transforms.Unnormalize(_transforms.output_norm_stats(with_goal), use_quantiles=True)(dict(outputs))
    assert restored["actions"].shape == (H, ACTION_DIM)
    assert _transforms.output_norm_stats(stats) == stats
    assert _transforms.output_norm_stats(None) is None


# ---- what stage 1 needs from the loader ----


def test_a_stage1_batch_through_the_real_transform_chain_has_every_camera(tmp_path):
    """Stage 1 embeds no image, but preprocess_observation (inside compute_loss) raises on an images dict that lacks a
    camera: the loader must decode and carry all three, the masked ones as zero images with a False mask."""
    config = _model_config("stage1")
    data_config = _data_config(config, tmp_path)
    items = _transformed(data_config, [_raw_sample(frame) for frame in (1, 4)])
    batch = _data_loader._collate_fn([items[0], items[1]])  # noqa: SLF001
    observation = _model.Observation.from_dict(batch)
    assert sorted(observation.images) == sorted(_model.IMAGE_KEYS)
    for name in _model.IMAGE_KEYS:
        assert observation.images[name].shape == (2, HIST, 224, 224, 3), name
    masks = {name: np.asarray(mask).tolist() for name, mask in observation.image_masks.items()}
    assert masks == {
        "base_0_rgb": [False, False],
        "left_wrist_0_rgb": [True, True],
        "right_wrist_0_rgb": [False, False],
    }
    # a masked camera is a constant image (the uint8 zeros of AgilexInputs, as -1.0 after from_dict)
    assert np.all(np.asarray(observation.images["base_0_rgb"]) == -1.0)
    assert observation.lit_goal.shape == (2, ACTION_DIM)
    assert observation.lit_goal_mask.shape == (2,)
    processed = _model.preprocess_observation(None, observation, train=False)
    assert sorted(processed.images) == sorted(_model.IMAGE_KEYS)
    assert processed.lit_goal is not None
    # the same call with a camera missing is what a loader that skipped decoding would hit
    missing = observation.replace(images={k: v for k, v in observation.images.items() if k != "base_0_rgb"},
                                  image_masks={k: v for k, v in observation.image_masks.items() if k != "base_0_rgb"})
    with pytest.raises(ValueError, match="missing keys"):
        _model.preprocess_observation(None, missing, train=False)
