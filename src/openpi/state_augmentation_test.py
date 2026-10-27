import random

import numpy as np
import pytest

import openpi.transforms as transforms

STATE = np.linspace(-0.8, 0.8, 14)


class _Tokenizer:
    def __init__(self):
        self.states = []

    def tokenize(self, prompt, state=None, *, task_only=False):
        self.states.append(None if state is None else np.array(state))
        return np.zeros(4, np.int32), np.ones(4, bool)


def _run(tokenizer, calls=1, **kwargs):
    transform = transforms.TokenizePrompt(tokenizer, discrete_state_input=True, **kwargs)
    return [transform({"prompt": "place one part in the bag", "state": STATE.copy()}) for _ in range(calls)]


def test_defaults_leave_the_tokenized_state_alone():
    tokenizer = _Tokenizer()
    _run(tokenizer, calls=5)
    assert all(np.array_equal(state, STATE) for state in tokenizer.states)


def test_dropout_of_one_hides_the_state_from_the_tokens_only():
    tokenizer = _Tokenizer()
    (out,) = _run(tokenizer, state_dropout=1.0)
    assert not tokenizer.states[0].any()
    assert np.array_equal(out["state"], STATE)


def test_dropout_rate_follows_the_probability():
    random.seed(0)
    tokenizer = _Tokenizer()
    _run(tokenizer, calls=2000, state_dropout=0.3)
    hidden = sum(not state.any() for state in tokenizer.states)
    assert 0.26 < hidden / 2000 < 0.34


def test_noise_touches_only_the_visible_dims_and_stays_in_range():
    random.seed(1)
    visible = tuple(range(7, 14))
    tokenizer = _Tokenizer()
    _run(tokenizer, calls=200, active_state_dims=visible, state_noise_std=0.2)
    states = np.stack(tokenizer.states)
    assert not states[:, :7].any()
    assert np.abs(states).max() <= 1.0
    spread = states[:, 7:].std(axis=0)
    assert (spread > 0.1).all()
    assert np.allclose(states[:, 7:].mean(axis=0), STATE[7:], atol=0.08)


def test_with_state_augmentation_switches_it_on_in_the_tokenizer_only():
    tokenizer = _Tokenizer()
    chain = [transforms.InjectDefaultPrompt("p"), transforms.TokenizePrompt(tokenizer, discrete_state_input=True)]
    same = transforms.with_state_augmentation(chain, 0.0, 0.0)
    assert all(a is b for a, b in zip(same, chain, strict=True))

    out = transforms.with_state_augmentation(chain, 0.4, 0.05)
    assert out[0] is chain[0]
    assert (out[1].state_dropout, out[1].state_noise_std) == (0.4, 0.05)
    assert (chain[1].state_dropout, chain[1].state_noise_std) == (0.0, 0.0)


def test_with_state_augmentation_refuses_configs_it_cannot_act_on():
    tokenizer = _Tokenizer()
    task_only = [transforms.TokenizePrompt(tokenizer, task_only=True)]
    with pytest.raises(ValueError, match="carries the state as text"):
        transforms.with_state_augmentation(task_only, 0.3, 0.0)
    with pytest.raises(ValueError, match="need a TokenizePrompt"):
        transforms.with_state_augmentation([transforms.InjectDefaultPrompt("p")], 0.3, 0.0)
