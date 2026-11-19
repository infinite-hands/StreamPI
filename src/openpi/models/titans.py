"""Titans' neural long-term memory (arXiv 2501.00663, "Titans: Learning to Memorize at Test Time"): a small network
whose WEIGHTS are learned while the model runs. A write is one gradient step on an associative loss (eq. 12) with
momentum (eq. 14) and a forgetting gate (eq. 13); a read is a forward pass (eq. 15). Ordinary training is the outer
loop: it learns everything a write and a read are made of by differentiating through the writes.

Equation and section numbers are the arXiv v1's. Every rule below is the paper's unless its comment says "our choice".
This is the paper's memory module (sec. 3.1-3.2) alone: no persistent tokens (sec. 3.3), none of the architectures of
sec. 4 (the caller decides how a read is used) and no convolutions around the projections (sec. 4.4)."""

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at

# Our choice: the initial logits of the three gates, [theta (step size), eta (surprise decay), alpha (forgetting)].
# Their kernels start at zero, so a new memory's gates are these constants until training makes them depend on the
# frame (sec. 3.2: "we make these parameters as the functions of tokens"). sigmoid(0) = 0.5; sigmoid(-3) = 0.047, a
# half-life of about 14 writes, near the frame count a training sample unrolls.
GATE_INITIAL_LOGITS = (0.0, 0.0, -3.0)
# Our choice (the paper has no bound): a write's momentary surprise is clipped to this global norm per sample, so one
# write moves the weights by a bounded amount whatever training has made of the parameters. Measured on a near-static
# scene over 2,000 writes it never engaged (median surprise norm 0.07 to 0.18); unclipped, a two-layer memory at
# extreme gates reached inf within 8 to 23 writes of one frame.
MAX_SURPRISE_NORM = 1.0
# Inside every norm below, so a zero vector has a finite gradient.
_NORM_EPSILON = 1e-6

# The state of one memory per sample: what its weights have moved by since the run began and the momentum of its
# surprise (one array per layer, each (b, in, out)), and whether anything has been written.
State = dict


def grid_pool(tokens: at.Float[at.Array, "*b n w"], grid: int) -> at.Float[at.Array, "*b cells w"]:
    """A frame's patch tokens (a square grid, row-major) averaged over a grid x grid partition of the picture."""
    count = tokens.shape[-2]
    side = math.isqrt(count)
    if side * side != count or side % grid:
        raise ValueError(f"{count} image tokens are not a square grid that {grid} x {grid} cells divide")
    cell = side // grid
    cells = tokens.reshape(*tokens.shape[:-2], grid, cell, grid, cell, tokens.shape[-1])
    return cells.mean(axis=(-4, -2)).reshape(*tokens.shape[:-2], grid * grid, tokens.shape[-1])


def _unit(x):
    return x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + _NORM_EPSILON)


def _rms(x):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + _NORM_EPSILON)


def recall(weights: tuple, keys: at.Float[at.Array, "p d"]) -> at.Float[at.Array, "p d"]:
    """M(k) for one sample: sec. 3.1's "simple MLPs with L_M >= 1 layers", without biases. One layer is the linear
    (matrix) memory. The paper names no activation; SiLU is our choice, smooth because the outer loop differentiates
    through the gradient of this function."""
    hidden = keys
    for index, weight in enumerate(weights):
        hidden = hidden @ weight
        if index < len(weights) - 1:
            hidden = jax.nn.silu(hidden)
    return hidden


def token_weights(keys: at.Float[at.Array, "p d"]) -> at.Float[at.Array, " p"]:
    """How much each token of a frame counts in its write. Sec. 3.2 SUMS a chunk's gradients (the frame is our chunk);
    that overshoots when the tokens share their key, as the cells of one camera picture largely do. Our choice: token p
    counts 1 / sum_q |k_p . k_q| -- the paper's sum when the keys are orthogonal, the mean when they are identical --
    floored at 1, so near-zero keys cannot magnify a write."""
    return 1.0 / jnp.maximum(jnp.sum(jnp.abs(keys @ keys.T), axis=-1), 1.0)


def associative_loss(weights: tuple, keys, values, step_sizes) -> at.Float[at.Array, ""]:
    """Eq. 12, ||M(k) - v||^2, summed over one frame's tokens, each weighted by its own step size (eq. 14's theta_t,
    times token_weights in a write): the gradient of this is the frame's momentary surprise. Every token's gradient is
    taken at the frame's first weights, as sec. 3.2 takes a chunk's."""
    error = jnp.sum(jnp.square(recall(weights, keys) - values), axis=-1)
    return jnp.sum(step_sizes * error)


class TitansMemory(nnx.Module):
    """The memory, its projections and its gates. Stateless: a memory's State is passed in and returned."""

    def __init__(self, token_width: int, dim: int, hidden_dim: int, depth: int, grid: int, rngs: nnx.Rngs):
        if depth < 1 or min(token_width, dim, hidden_dim, grid) < 1:
            raise ValueError("a Titans memory needs depth >= 1 and positive widths and grid")
        self.dim = dim
        self.grid = grid
        self.layers = tuple(f"layer_{index}" for index in range(depth))
        normal = nnx.initializers.lecun_normal()
        # Our choice: what the memory is given. A frame's grid-pooled image tokens, projected to the memory's width, plus
        # a learned position per grid cell as large as the content, so a key says where in the picture as much as what:
        # a read can then say what a cell showed before it changed (measured; with positions 50x smaller it cannot).
        self.in_proj = nnx.Linear(token_width, dim, rngs=rngs)
        self.position = nnx.Param(jax.random.normal(rngs.params(), (grid * grid, dim)))
        # Eq. 11 and sec. 3.1: k = x W_K, v = x W_V, q = x W_Q. Our choice: W_Q starts as a copy of W_K (then trains on its
        # own), so a read addresses what was written from the first step; with an independent random W_Q, reads after
        # writing two unrelated scenes were 0.99 alike.
        self.key_proj = nnx.Param(normal(rngs.params(), (dim, dim)))
        self.value_proj = nnx.Param(normal(rngs.params(), (dim, dim)))
        self.query_proj = nnx.Param(jnp.array(self.key_proj.value))
        self.gate_kernel = nnx.Param(jnp.zeros((dim, len(GATE_INITIAL_LOGITS))))
        self.gate_bias = nnx.Param(jnp.asarray(GATE_INITIAL_LOGITS, dtype=jnp.float32))
        # M_0, the weights a run starts from. The paper does not say where they come from; learning them in the outer
        # loop is our choice.
        widths = (dim, *([hidden_dim] * (depth - 1)), dim)
        self.initial_weights = nnx.Dict({
            name: nnx.Param(normal(rngs.params(), (fan_in, fan_out)))
            for name, fan_in, fan_out in zip(self.layers, widths[:-1], widths[1:], strict=True)})

    def tokens(self, image_tokens: at.Float[at.Array, "*b n w"]) -> at.Float[at.Array, "*b cells d"]:
        """One frame's image tokens as the memory's input x, float32."""
        content = _rms(self.in_proj(grid_pool(image_tokens.astype(jnp.float32), self.grid)))
        return _rms(content + self.position.value)

    def initial_state(self, batch_size: int) -> State:
        """A run's first state: nothing moved, no momentum, nothing written. It does not depend on the parameters."""
        zeros = tuple(jnp.zeros((batch_size, *self.initial_weights[name].value.shape), jnp.float32)
                      for name in self.layers)
        return {"deviation": zeros, "momentum": zeros, "written": jnp.zeros((batch_size,), dtype=jnp.bool_)}

    def _weights(self, state: State) -> tuple:
        return tuple(self.initial_weights[name].value[None] + deviation
                     for name, deviation in zip(self.layers, state["deviation"], strict=True))

    def _pairs(self, x):
        # Sec. 4.4 normalises keys and queries; the values too is our choice, so the error and the step stay O(1).
        return _unit(x @ self.key_proj.value), _unit(x @ self.value_proj.value)

    def surprise(self, state: State, x: at.Float[at.Array, "b cells d"]) -> at.Float[at.Array, " b"]:
        """Eq. 12 for each sample's memory on one frame, averaged over its tokens: how badly the memory recalls it."""
        keys, values = self._pairs(x)
        return jax.vmap(associative_loss)(
            self._weights(state), keys, values, jnp.full(x.shape[:2], 1.0 / x.shape[1], jnp.float32))

    def write(self, state: State, x: at.Float[at.Array, "b cells d"], valid: at.Bool[at.Array, " b"]) -> State:
        """Eq. 13-14 for one frame: S <- eta S - theta grad l(M; x), then the forgetting step. A sample whose `valid`
        is False keeps its state.

        Two things differ from the paper, both our choice. The paper's forgetting decays the weights themselves,
        M <- (1 - alpha) M + S; here it decays what the run has written, D <- (1 - alpha) D + S with M = M_0 + D, so a
        memory that forgets returns to its learned starting weights instead of to zero (a deeper memory whose weights
        have decayed to zero passes no signal and no gradient). And the surprise is clipped (MAX_SURPRISE_NORM)."""
        keys, values = self._pairs(x)
        kernel, bias = self.gate_kernel.value, self.gate_bias.value
        # theta per token; eta and alpha per frame, from its mean token (sec. 3.2 allows one value per chunk).
        theta = jax.nn.sigmoid(x @ kernel[:, 0] + bias[0]) * jax.vmap(token_weights)(keys)
        frame = jnp.mean(x, axis=-2)
        eta = jax.nn.sigmoid(frame @ kernel[:, 1] + bias[1])[:, None, None]
        alpha = jax.nn.sigmoid(frame @ kernel[:, 2] + bias[2])[:, None, None]

        surprise = jax.vmap(jax.grad(associative_loss))(self._weights(state), keys, values, theta)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(layer), axis=(-2, -1)) for layer in surprise) + _NORM_EPSILON)
        clip = jnp.minimum(1.0, MAX_SURPRISE_NORM / norm)[:, None, None]
        momentum = tuple(eta * previous - clip * layer
                         for previous, layer in zip(state["momentum"], surprise, strict=True))
        deviation = tuple((1.0 - alpha) * previous + step
                          for previous, step in zip(state["deviation"], momentum, strict=True))
        keep = valid[:, None, None]
        return {
            "deviation": tuple(jnp.where(keep, new, old)
                               for new, old in zip(deviation, state["deviation"], strict=True)),
            "momentum": tuple(jnp.where(keep, new, old) for new, old in zip(momentum, state["momentum"], strict=True)),
            "written": state["written"] | valid,
        }

    def read(self, state: State, x: at.Float[at.Array, "b cells d"]
             ) -> tuple[at.Float[at.Array, "b cells d"], at.Bool[at.Array, "b cells"]]:
        """Eq. 15, y = M*(q): the forward pass, no weight update. Returns (tokens at unit RMS, which of them count).
        Nothing counts for a sample whose memory is still empty, or whose read is not finite (our choice: a memory
        that has gone wrong reads as an empty one, it does not reach whoever reads it)."""
        queries = _unit(x @ self.query_proj.value)
        retrieved = jax.vmap(recall)(self._weights(state), queries)
        finite = jnp.all(jnp.isfinite(retrieved), axis=(-2, -1))
        tokens = _rms(jnp.where(finite[:, None, None], retrieved, 0.0))
        return tokens, jnp.broadcast_to((state["written"] & finite)[:, None], tokens.shape[:2])

    def unroll(self, frames: at.Float[at.Array, "b k cells d"], valid: at.Bool[at.Array, "b k"]) -> State:
        """The state after writing `frames` in order (oldest first) from an empty memory: training's inner loop."""
        state, _ = jax.lax.scan(
            lambda state, frame: (self.write(state, *frame), None),
            self.initial_state(frames.shape[0]),
            (jnp.swapaxes(frames, 0, 1), jnp.swapaxes(valid, 0, 1)))
        return state
