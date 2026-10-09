from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
import random
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

# RACE (Pi0Config.race). The sidecar a recipe's DataConfig.race_targets_dir names holds meta.json and these
# arrays, each indexed by the LeRobot dataset's global frame index (its "index" column).
RACE_STORE_ARRAYS = {"episode_index": np.int32, "frame_index": np.int32, "transition": np.float32,
                     "weight": np.float32}
# The training targets RaceTargets adds to a sample (Observation.transition_window / transition_window_mask).
RACE_TARGET_KEYS = ("transition_window", "transition_window_mask")
# A RACE model's served output: its head's transition score for each returned action row.
TRANSITION_SCORES_KEY = "transition_scores"
_RACE_STORES: dict[str, dict[str, np.ndarray]] = {}


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class TemporalJitter(DataTransformFn):
    jitter_range: None
    hist_interval: int
    hist_horizon: int
    hist_sequence_keys: list
    enable_jitter: bool

    def __call__(self, data: DataDict) -> DataDict:
        for key in self.hist_sequence_keys:
            full_seq = data[key]
            T = full_seq.shape[0]
            
            sampled_frames = []
            if self.enable_jitter:
                base_offset = random.choice(self.jitter_range)
            else:
                base_offset = 0

            for i in range(self.hist_horizon):
                frame_idx = (T - 1) - (i * self.hist_interval) + base_offset

                frame_idx = np.clip(frame_idx, 0, T - 1)
                sampled_frames.append(full_seq[frame_idx])

            data[key] = np.stack(sampled_frames[::-1], axis=0)

        # import cv2
        # T, H, W, _ = data[self.hist_sequence_keys[0]].shape
        # frames_list = [(data[self.hist_sequence_keys[0]][i] * 255).astype(np.uint8).transpose((1, 2, 0)) for i in range(T)]

        # concat_image = cv2.hconcat(frames_list)
        # success = cv2.imwrite("/workspace/jhhou/projects/StreamingVLA/imgs.png", concat_image)
        # import pdb;pdb.set_trace()
        return data


NOW_STATE_KEY = "state_now"   # the measured state at the image's frame, beside the state at t + delta
DELTA_KEY = "vlash_delta"     # delta / max_offset: how far the state runs ahead of the image


def output_norm_stats(norm_stats):
    """The norm stats Unnormalize gets: without the input-only NOW_STATE_KEY, which no model output carries
    (Unnormalize is strict, so leaving it in failed every serve call of a vlash_cond_now config)."""
    if norm_stats is None:
        return None
    return {key: value for key, value in norm_stats.items() if key != NOW_STATE_KEY}


@dataclasses.dataclass(frozen=True)
class ConcatVlashCond(DataTransformFn):
    """Join NOW_STATE_KEY and DELTA_KEY to the (normalized) state: [state at t+delta, state at t, delta],
    the input of a state_cond model's conditioning MLP. Runs after Normalize and before the state is padded
    to the model's width. A request without them is the offset-0 case -- the state is now, delta 0 -- which
    is exactly what a plain (non-vlash) call is."""

    def __call__(self, data: DataDict) -> DataDict:
        state = np.asarray(data["state"], dtype=np.float32)
        now = np.asarray(data.pop(NOW_STATE_KEY, state), dtype=np.float32)
        delta = np.asarray(data.pop(DELTA_KEY, np.zeros(state.shape[:-1], dtype=np.float32)), dtype=np.float32)
        delta = delta.reshape(state.shape[:-1] + (1,))   # one per sample, or one per branch
        data["state"] = np.concatenate([state, np.broadcast_to(now, state.shape), delta], axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class TemporalOffset(DataTransformFn):
    """VLASH's temporal-offset augmentation (arXiv 2512.01031): the images stay at frame t while the
    state and the action window move `delta` frames ahead, `delta` drawn uniformly from 0..max_offset
    per sample. The deploy loop can then send the state the arm will be at when the chunk starts.

    The dataset must have fetched action_horizon + max_offset action rows. The state at t+delta is
    the previous commanded action a[t+delta-1] (state_source "action": the reference
    implementation's proxy, and the one thing the deploy loop knows ahead of time) or the recorded
    state s[t+delta] (state_source "state", which needs max_offset + 1 fetched state rows). Runs
    before repack so DeltaActions is relative to the shifted state. A no-op at max_offset 0.

    With `branches` > 0 (the paper's shared-observation training) the sample keeps the one observation
    and carries that many distinct offsets at once: the state becomes (branches, D) and each action key
    (branches, action_horizon, A), in ascending offset order.

    With `cond_now` the sample also carries what the state at t+delta alone leaves out: NOW_STATE_KEY, the
    measured state at frame t (where the arm -- and a wrist camera -- was when the image was taken), and
    DELTA_KEY, delta / max_offset (how far the state runs ahead of the image). ConcatVlashCond joins them to
    the state for the adaRMS conditioning."""

    max_offset: int
    action_horizon: int
    action_keys: Sequence[str]
    state_key: str = "observation.state"
    state_source: str = "action"
    branches: int = 0
    cond_now: bool = False

    def __post_init__(self):
        if self.max_offset < 0:
            raise ValueError(f"max_offset must be >= 0, got {self.max_offset}")
        if self.state_source not in ("action", "state"):
            raise ValueError(f"state_source must be 'action' or 'state', got {self.state_source!r}")
        if not self.action_keys:
            raise ValueError("TemporalOffset needs at least one action key")
        if self.branches < 0 or self.branches > self.max_offset + 1:
            raise ValueError(f"branches must be 0..{self.max_offset + 1} (one per distinct offset), got {self.branches}")

    def __call__(self, data: DataDict) -> DataDict:
        if self.max_offset == 0 and not self.branches:
            return data
        needed = self.action_horizon + self.max_offset
        for key in self.action_keys:
            if data[key].shape[0] < needed:
                raise ValueError(f"{key} has {data[key].shape[0]} rows; TemporalOffset needs {needed} "
                                 f"(action_horizon {self.action_horizon} + max_offset {self.max_offset})")
        states = data[self.state_key]
        if self.state_source == "state" and states.shape[0] < self.max_offset + 1:
            raise ValueError(f"{self.state_key} has {states.shape[0]} rows; TemporalOffset needs "
                             f"{self.max_offset + 1} for state_source 'state'")
        first_actions = data[self.action_keys[0]]
        now = np.asarray(states[0] if self.state_source == "state" else states)
        if self.branches:
            deltas = sorted(random.sample(range(self.max_offset + 1), self.branches))
            data[self.state_key] = np.stack([self._state_at(delta, states, first_actions) for delta in deltas])
            for key in self.action_keys:
                rows = data[key]
                data[key] = np.stack([rows[delta:delta + self.action_horizon] for delta in deltas])
            if self.cond_now:
                data[NOW_STATE_KEY] = np.stack([now] * len(deltas))
                data[DELTA_KEY] = np.asarray(deltas, dtype=np.float32) / max(self.max_offset, 1)
            return data
        delta = random.randint(0, self.max_offset)
        data[self.state_key] = self._state_at(delta, states, first_actions)
        for key in self.action_keys:
            data[key] = data[key][delta:delta + self.action_horizon]
        if self.cond_now:
            data[NOW_STATE_KEY] = now
            data[DELTA_KEY] = np.float32(delta / max(self.max_offset, 1))
        return data

    def _state_at(self, delta: int, states, actions):
        if self.state_source == "state":
            return states[delta]
        return states if delta == 0 else actions[delta - 1]


def race_store(directory: str) -> dict[str, np.ndarray]:
    """A RACE sidecar's arrays, memory-mapped once per process (each data-loader worker maps its own)."""
    if directory not in _RACE_STORES:
        _RACE_STORES[directory] = {name: np.load(f"{directory}/{name}.npy", mmap_mode="r")
                                   for name in RACE_STORE_ARRAYS}
    return _RACE_STORES[directory]


@dataclasses.dataclass(frozen=True)
class RaceTargets(DataTransformFn):
    """RACE's training targets for a LeRobot sample anchored at global frame i (its "index"), whose action rows
    are frames i .. i+H-1: the soft transition target of frames i-1 .. i+H ("transition_window", the rows plus
    one neighbour each side for the conditioning jitter) and which of those frames lie inside the anchor's
    episode ("transition_window_mask"). A frame outside the episode reads 0. Runs on the raw LeRobot sample."""

    directory: str
    action_horizon: int

    def __call__(self, data: DataDict) -> DataDict:
        store = race_store(self.directory)
        episodes = store["episode_index"]
        index = int(data["index"])
        if not 0 <= index < episodes.shape[0]:
            raise ValueError(f"frame {index} is outside the race targets at {self.directory} "
                             f"({episodes.shape[0]} frames)")
        episode = int(episodes[index])
        if int(data["episode_index"]) != episode or int(data["frame_index"]) != int(store["frame_index"][index]):
            raise ValueError(
                f"race targets at {self.directory} disagree with the dataset at frame {index}: they say episode "
                f"{episode} frame {int(store['frame_index'][index])}, the dataset episode "
                f"{int(data['episode_index'])} frame {int(data['frame_index'])}")
        frames = np.arange(index - 1, index + self.action_horizon + 1)
        held = np.clip(frames, 0, episodes.shape[0] - 1)
        inside = (frames >= 0) & (frames < episodes.shape[0]) & (episodes[held] == episode)
        data["transition_window"] = np.where(inside, store["transition"][held], 0.0).astype(np.float32)
        data["transition_window_mask"] = inside
        return data


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        for k, v in data["image"].items():
            if v.ndim == 3:
                data["image"][k] = image_tools.resize_with_pad(v, self.height, self.width)
            elif v.ndim == 4:
                images = [image_tools.resize_with_pad(vi, self.height, self.width) for vi in v]
                data["image"][k] = np.stack(images)
            else:
                raise ValueError
        # import cv2
        # T, H, W, _ = data["image"]["base_0_rgb"].shape
        # frames_list = [data["image"]["base_0_rgb"][i] for i in range(T)]
        # concat_image = cv2.hconcat(frames_list)
        # success = cv2.imwrite("/data1/zliu/projects/OpenPI/openpi/imgs.png", concat_image)
        # import pdb;pdb.set_trace()
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class HoldActions(DataTransformFn):
    """Replaces the target of `dims` with holding still: 0 where `delta_mask` marks a delta dim, the
    current state where the dim is absolute. Runs after DeltaActions. A no-op when `dims` is None."""

    dims: Sequence[int] | None
    delta_mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.dims is None:
            return data

        state, actions = np.asarray(data["state"]), np.array(data["actions"], copy=True)
        delta = np.zeros(actions.shape[-1], dtype=bool)
        if self.delta_mask is not None:
            mask = np.asarray(self.delta_mask)
            delta[: mask.shape[-1]] = mask
        for dim in self.dims:
            actions[..., dim] = 0.0 if delta[dim] else state[..., dim, None]   # one state per chunk, over its rows
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    # None (default): exact no-op. Else: state dims the model may see; every other dim is set to the
    # normalized midpoint (0) in the TOKENIZED copy only. data["state"] is left real on purpose:
    # DeltaActions (train) and AbsoluteActions (serve) both need the true state, and pi0.5 reads
    # state nowhere but these tokens, so this is the one place a mask hides it from the model.
    active_state_dims: tuple[int, ...] | None = None
    # pi0.5 with the state out of the prompt (Pi0Config.state_cond): "Task: ...;\nAction: ".
    task_only: bool = False

    def __post_init__(self):
        if self.active_state_dims is not None and not self.discrete_state_input:
            raise ValueError("active_state_dims masks the discrete state tokens; it needs discrete_state_input")
        if self.task_only and self.discrete_state_input:
            raise ValueError("task_only leaves the state out of the prompt; it excludes discrete_state_input")

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
            if self.active_state_dims is not None:
                state = np.asarray(state)
                keep = np.zeros(state.shape[-1], dtype=bool)
                keep[list(self.active_state_dims)] = True
                state = np.where(keep, state, 0.0)
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state, task_only=self.task_only)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
