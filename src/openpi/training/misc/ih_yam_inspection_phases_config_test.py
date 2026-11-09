"""The inspection phase recipes: each frame's task text reaches the model, a missing prompt raises, and the sort-both shape."""

import dataclasses

import flax.nnx as nnx
import jax
import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.misc import ih_yam_config

BOTH = "pi05_yam_stream5_i15_h35_sort_both_real"
PHASES = "pi05_yam_stream5_i15_h35_inspection_phases_real"
PHASES_NO_HISTORY = "pi05_yam_stream1_i15_h35_inspection_phases_real"
TASKS = {0: "pick up the roll with the left arm", 1: "show the end of the roll to the overhead camera"}
IMAGE_SHAPE = (3, 8, 8)   # any size: ResizeImages pads it to the model's


def _data_config(name, monkeypatch):
    config = _config.get_config(name)
    monkeypatch.setattr(type(config.data), "_load_norm_stats", lambda self, assets_dir, asset_id: None)
    return config, config.data.create(config.assets_dirs, config.model)


def _lerobot_sample(config, task_index: int) -> dict:
    """One item as LeRobotDataset hands it to the loader: flat keys, a history stack per camera, a task index."""
    frames = config.data.base_config.hist_horizon
    rng = np.random.default_rng(task_index)
    return {
        **{key: np.zeros((frames, *IMAGE_SHAPE), np.float32) for key in config.data.hist_sequence_keys},
        "observation.state": rng.uniform(-1, 1, ih_yam_config.STATE_DIM).astype(np.float32),
        "action": rng.uniform(-1, 1, (config.model.action_horizon, ih_yam_config.STATE_DIM)).astype(np.float32),
        "task_index": np.int64(task_index),
    }


def _trained_on(config, data_config, task_index: int) -> dict:
    """The sample after the training input chain, with the task prompt the loader adds for prompt_from_task."""
    sample = _data_loader.TransformedDataset([_lerobot_sample(config, task_index)],
                                             [_transforms.PromptFromLeRobotTask(TASKS)])
    return _data_loader.transform_dataset(sample, data_config, skip_norm_stats=True)[0]


def _expected_tokens(config, text: str, state: np.ndarray) -> np.ndarray:
    return _tokenizer.PaligemmaTokenizer(config.model.max_token_len).tokenize(text, state)[0]


@pytest.mark.parametrize("name", [PHASES, PHASES_NO_HISTORY])
def test_each_frames_task_text_survives_the_repack_into_the_tokens(name, monkeypatch):
    config, data_config = _data_config(name, monkeypatch)
    assert data_config.prompt_from_task and config.data.default_prompt is None
    for task_index, text in TASKS.items():
        state = _lerobot_sample(config, task_index)["observation.state"]
        tokens = _trained_on(config, data_config, task_index)["tokenized_prompt"]
        np.testing.assert_array_equal(tokens, _expected_tokens(config, text, state))


def test_sort_both_still_drops_the_task_text_and_trains_on_its_default(monkeypatch):
    config, data_config = _data_config(BOTH, monkeypatch)
    assert not data_config.prompt_from_task
    state = _lerobot_sample(config, 1)["observation.state"]
    tokens = _trained_on(config, data_config, 1)["tokenized_prompt"]
    np.testing.assert_array_equal(tokens, _expected_tokens(config, ih_yam_config.SORT_PROMPT, state))


def _served(data_config, request: dict) -> dict:
    """policy_config.create_trained_policy's input chain with no --default-prompt and no repack."""
    chain = _transforms.compose([_transforms.InjectDefaultPrompt(None), *data_config.data_transforms.inputs,
                                 _transforms.Normalize({}, use_quantiles=data_config.use_quantile_norm),
                                 *data_config.model_transforms.inputs])
    return chain(request)


def _request() -> dict:
    return {"images": {camera: np.zeros(IMAGE_SHAPE, np.float32) for camera in ih_yam_config.YAM_CAMERAS},
            "state": np.zeros(ih_yam_config.STATE_DIM, np.float32)}


@pytest.mark.parametrize("name", [PHASES, PHASES_NO_HISTORY])
def test_a_missing_prompt_raises_at_serve_and_in_training(name, monkeypatch):
    config, data_config = _data_config(name, monkeypatch)
    with pytest.raises(ValueError, match="Prompt is required"):
        _served(data_config, _request())
    served = _served(data_config, {**_request(), "prompt": TASKS[1]})
    np.testing.assert_array_equal(served["tokenized_prompt"],
                                  _expected_tokens(config, TASKS[1], np.zeros(ih_yam_config.STATE_DIM)))
    with pytest.raises(KeyError):
        data_config.repack_transforms.inputs[0](_lerobot_sample(config, 0))


def test_sort_both_still_serves_its_default_to_a_request_without_one(monkeypatch):
    config, data_config = _data_config(BOTH, monkeypatch)
    np.testing.assert_array_equal(_served(data_config, _request())["tokenized_prompt"],
                                  _expected_tokens(config, ih_yam_config.SORT_PROMPT, np.zeros(ih_yam_config.STATE_DIM)))


@pytest.mark.parametrize("name", [PHASES, PHASES_NO_HISTORY])
def test_the_phase_recipes_are_the_sort_both_shape(name):
    config, both = _config.get_config(name), _config.get_config(BOTH)
    assert dataclasses.replace(config.model, hist_horizon=both.model.hist_horizon) == both.model
    assert dataclasses.replace(config.data.base_config, prompt_from_task=False,
                               hist_horizon=both.data.base_config.hist_horizon) == both.data.base_config
    assert config.data.hist_sequence_keys == both.data.hist_sequence_keys
    assert (config.data.active_image_keys, config.data.active_state_dims, config.data.held_action_dims) == (None, None, None)
    assert (config.batch_size, config.fsdp_devices, config.ema_decay, config.num_train_steps) == (
        both.batch_size, both.fsdp_devices, both.ema_decay, both.num_train_steps)
    assert config.data.repo_id == ih_yam_config.INSPECTION_PHASES_REAL_REPO_ID


def test_streaming_and_history_off_differ_only_in_history():
    streaming, ablation = _config.get_config(PHASES), _config.get_config(PHASES_NO_HISTORY)
    assert streaming.model == _config.get_config(BOTH).model
    assert (streaming.model.hist_horizon, ablation.model.hist_horizon) == (ih_yam_config.HIST_HORIZON, 1)
    assert ablation.data.base_config.hist_horizon == 1


def test_a_sort_both_checkpoint_loads_into_the_history_off_recipe():
    def param_shapes(model_config):
        model = nnx.eval_shape(model_config.create, jax.random.key(0))
        return jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), nnx.state(model).to_pure_dict())

    assert param_shapes(_config.get_config(PHASES_NO_HISTORY).model) == param_shapes(_config.get_config(BOTH).model)


def test_existing_recipes_are_unchanged_and_the_phase_recipes_have_no_twins():
    configs = {config.name: config for config in ih_yam_config.get_ih_yam_configs()}
    assert {name for name in configs if "inspection" in name} == {PHASES, PHASES_NO_HISTORY}
    for name, config in configs.items():
        if name in (PHASES, PHASES_NO_HISTORY):
            continue
        assert config.data.default_prompt is not None and not config.data.base_config.prompt_from_task, name
        assert "prompt" not in config.data.repack_transforms.inputs[0].structure, name
        assert config.model.hist_horizon == config.data.base_config.hist_horizon == ih_yam_config.HIST_HORIZON, name
