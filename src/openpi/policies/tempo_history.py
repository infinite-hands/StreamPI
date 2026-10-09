"""TEMPO inputs for streaming history units: per unit, one camera's SAM2 tokens and the bucket-mean
past actions at that unit's frame.

`action_history` is TEMPO's LoadActionHistory (tempo-robot/TEMPO, yam_dual_policy.py at c88312cf):
the window ending at t-1 split into `steps` equal buckets, each averaged, frames outside the episode
counted as zero, and a bucket flagged pad when all of it lies outside. Training (LoadTempoHistory)
and serving (the policy server, over the actions the box executed) both call it, so the two compute
the same thing."""

import dataclasses
import pathlib

import numpy as np

from openpi import transforms

TOKENS_DIR = "tokens"
ACTIONS_DIR = "actions"
EPISODE_FILE = "episode_{:06d}.npy"
_OPEN: dict[str, np.ndarray] = {}  # path -> its memory map (tokens) or array (actions), per worker process


def action_history(actions: np.ndarray, frame: int, steps: int, frames_per_bucket: int) -> tuple[np.ndarray, np.ndarray]:
    """(steps, A) bucket means over actions[frame - steps * frames_per_bucket : frame], and (steps,) pad flags."""
    window = steps * frames_per_bucket
    index = np.arange(frame - window, frame)
    valid = (index >= 0) & (index < len(actions))
    gathered = np.zeros((window, actions.shape[1]), dtype=np.float32)
    gathered[valid] = actions[index[valid]]
    return (gathered.reshape(steps, frames_per_bucket, -1).mean(axis=1),
            (~valid).reshape(steps, frames_per_bucket).all(axis=1))


def _load(path: str, mmap: bool) -> np.ndarray:
    if path not in _OPEN:
        if not pathlib.Path(path).is_file():
            raise FileNotFoundError(f"{path} is missing: TEMPO trains on no stand-in for an absent cache entry")
        _OPEN[path] = np.load(path, mmap_mode="r" if mmap else None)
    return _OPEN[path]


@dataclasses.dataclass(frozen=True)
class LoadTempoHistory(transforms.DataTransformFn):
    """Attach `sam2_tokens` (H, n, d), `action_history` (H, steps, A) and `action_history_is_pad`
    (H, steps) at the frames TemporalJitter kept for `hist_key`. Runs right after TemporalJitter, while
    episode_index and frame_index are still present."""

    cache_dir: str
    hist_key: str
    action_dims: tuple[int, ...]
    steps: int
    frames_per_bucket: int

    def __call__(self, data: dict) -> dict:
        episode = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        offsets = np.asarray(data.pop(self.hist_key + transforms.HIST_OFFSETS_SUFFIX))
        name = EPISODE_FILE.format(episode)
        tokens = _load(f"{self.cache_dir}/{TOKENS_DIR}/{name}", mmap=True)
        actions = _load(f"{self.cache_dir}/{ACTIONS_DIR}/{name}", mmap=False)[:, list(self.action_dims)]
        frames = np.clip(frame + offsets, 0, len(tokens) - 1)  # LeRobot clamps a history query at the episode start
        histories = [action_history(actions, int(f), self.steps, self.frames_per_bucket) for f in frames]
        return {
            **data,
            "sam2_tokens": np.array(tokens[frames], dtype=np.float32),
            "action_history": np.stack([history for history, _ in histories]),
            "action_history_is_pad": np.stack([pad for _, pad in histories]),
        }


@dataclasses.dataclass(frozen=True)
class NormalizeActionHistory(transforms.DataTransformFn):
    """Quantile-normalize `action_history` with the STATE stats at `dims`: the history is absolute joint
    and gripper commands, the space `state` is in (the action stats describe deltas). A bucket that lies
    wholly before the episode becomes exactly zero, so the adaRMS residual, which reads pad buckets too,
    sees no offset there. Runs after Normalize, at train and serve time alike."""

    state_stats: transforms.NormStats | None
    dims: tuple[int, ...]

    def __call__(self, data: dict) -> dict:
        if "action_history" not in data:
            return data
        if self.state_stats is None or self.state_stats.q01 is None:
            raise ValueError("TEMPO's action history needs the recipe's quantile state stats, and none were loaded")
        q01 = np.asarray(self.state_stats.q01)[list(self.dims)]
        q99 = np.asarray(self.state_stats.q99)[list(self.dims)]
        history = (np.asarray(data["action_history"]) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        pad = np.asarray(data["action_history_is_pad"])[..., None]
        return {**data, "action_history": np.where(pad, 0.0, history).astype(np.float32)}
