"""A test-time-training layer (Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden
States", arXiv 2407.04620), the TTT-Linear variant of the reference implementation
(test-time-training/ttt-lm-pytorch, ttt.py).

The layer's hidden state is the weights (W, b) of a linear model f(x) = x W + b. Every mini-batch of tokens
takes one gradient step on W and b for the self-supervised loss || LN(f(theta_K x)) - (theta_V x - theta_K x) ||^2,
and the layer's output is the updated model applied to the query view, theta_Q x + LN(f(theta_Q x)). The
projections theta, the starting weights (W_0, b_0), the layer norm and the per-token learning-rate gate are
the outer loop's parameters: they are trained by backpropagating through the gradient steps.

What differs from the reference, each our choice:
- A mini-batch is one streaming frame (its image and prompt tokens), not 16 consecutive tokens: StreamPI's
  prefix attends within a frame and to earlier frames, so every token of a frame may read the weights after
  that frame's step. One policy call adds one frame, so one call is one gradient step.
- The reference scales token k of a mini-batch by 1/k plus a learned offset (a running mean over the tokens
  so far). A frame's step is the mean over its valid tokens, with no learned offset.
- Padding tokens (prompt padding, a masked camera, a dropped history frame) take no part in the step.
- The output projection starts at zero, so a checkpoint without the layer starts as the model it was.
"""

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

# The reference's TTTConfig.ttt_base_lr for TTT-Linear.
TTT_BASE_LR = 1.0
# The reference's head width: hidden_size 2048 over num_attention_heads 32.
TTT_HEAD_DIM = 64
# The reference's init of W_0 and of the learning-rate gate's weight (normal, std 0.02), and its
# initializer_range for the projections.
TTT_INIT_STD = 0.02
# The reference's epsilon in the inner layer norm and in post_norm.
TTT_LN_EPS = 1e-6

# (W, b): float32 (b, heads, head_dim, head_dim) and (b, heads, 1, head_dim).
TttState = tuple[jax.Array, jax.Array]


def _layer_norm(x, scale, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + TTT_LN_EPS)
    normed = (x - mean) / std
    return scale * normed + bias, normed, std


def _layer_norm_l2_grad(x, target, scale, bias):
    """d/dx of 0.5 || LN(x) - target ||^2 summed over the last axis: the reference's ln_fused_l2_bwd."""
    out, normed, std = _layer_norm(x, scale, bias)
    grad_normed = (out - target) * scale
    width = x.shape[-1]
    return (width * grad_normed
            - jnp.sum(grad_normed, axis=-1, keepdims=True)
            - normed * jnp.sum(grad_normed * normed, axis=-1, keepdims=True)) / (width * std)


class TTTLinear(nn.Module):
    """One TTT-Linear layer over a stream of frames. `width` is the inner model's total width, split into
    heads of TTT_HEAD_DIM that each learn their own linear model."""

    width: int

    @nn.compact
    def __call__(self, x, mask, state: TttState | None):
        """x: (b, frames * frame_len, d) tokens, oldest frame first. mask: bool (b, frames, frame_len), False
        where a token is padding. state: the weights the first frame starts from; None starts from the
        learned (W_0, b_0). Returns the layer's output in x's dtype and the weights after the last frame."""
        if self.width % TTT_HEAD_DIM:
            raise ValueError(f"TTT width {self.width} is not a multiple of the head width {TTT_HEAD_DIM}")
        heads = self.width // TTT_HEAD_DIM
        batch, frames, _frame_len = mask.shape
        dtype = x.dtype
        x = x.astype(jnp.float32)

        def view(name):
            projected = nn.Dense(self.width, use_bias=False, kernel_init=nn.initializers.normal(TTT_INIT_STD),
                                 name=name)(x)
            return einops.rearrange(projected, "b (t f) (h d) -> t b h f d", t=frames, h=heads)

        query, key, value = view("q_proj"), view("k_proj"), view("v_proj")

        lr_weight = self.param("lr_weight", nn.initializers.normal(TTT_INIT_STD), (heads, x.shape[-1]))
        lr_bias = self.param("lr_bias", nn.initializers.zeros_init(), (heads,))
        gate = jax.nn.sigmoid(jnp.einsum("bsd,hd->bhs", x, lr_weight) + lr_bias[None, :, None])
        gate = einops.rearrange(gate, "b h (t f) -> t b h f 1", t=frames)
        valid = einops.rearrange(mask.astype(jnp.float32), "b t f -> t b 1 f 1")
        # The mean over the frame's valid tokens; a frame with none leaves the weights as they were.
        eta = TTT_BASE_LR / TTT_HEAD_DIM * gate * valid / jnp.maximum(jnp.sum(valid, axis=-2, keepdims=True), 1.0)

        w_init = self.param("w_init", nn.initializers.normal(TTT_INIT_STD), (heads, TTT_HEAD_DIM, TTT_HEAD_DIM))
        b_init = self.param("b_init", nn.initializers.zeros_init(), (heads, 1, TTT_HEAD_DIM))
        ln_scale = self.param("ln_scale", nn.initializers.ones_init(), (heads, 1, TTT_HEAD_DIM))
        ln_bias = self.param("ln_bias", nn.initializers.zeros_init(), (heads, 1, TTT_HEAD_DIM))
        if state is None:
            weights = jnp.broadcast_to(w_init, (batch, *w_init.shape))
            biases = jnp.broadcast_to(b_init, (batch, *b_init.shape))
        else:
            weights, biases = state

        outputs = []
        for frame in range(frames):
            grad = _layer_norm_l2_grad(key[frame] @ weights + biases, value[frame] - key[frame], ln_scale, ln_bias)
            weights = weights - jnp.einsum("bhfd,bhfe->bhde", eta[frame] * key[frame], grad)
            biases = biases - jnp.sum(eta[frame] * grad, axis=-2, keepdims=True)
            outputs.append(query[frame] + _layer_norm(query[frame] @ weights + biases, ln_scale, ln_bias)[0])

        out = einops.rearrange(jnp.stack(outputs), "t b h f d -> b (t f) (h d)")
        out = nn.LayerNorm(epsilon=TTT_LN_EPS, name="post_norm")(out)
        out = nn.Dense(x.shape[-1], use_bias=False, kernel_init=nn.initializers.zeros_init(), name="out_proj")(out)
        return out.astype(dtype), (weights, biases)
