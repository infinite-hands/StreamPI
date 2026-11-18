"""The TTT-Linear layer alone: its step is a gradient step on the inner loss, a stream of single-frame calls
equals one call over the frames, and padding takes no part. Float32, CPU."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import ttt as _ttt

BATCH, FRAMES, FRAME_LEN, DIM, WIDTH = 2, 3, 5, 24, 128
HEADS = WIDTH // _ttt.TTT_HEAD_DIM


def _layer():
    return _ttt.TTTLinear(width=WIDTH)


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(rng.normal(size=(BATCH, FRAMES * FRAME_LEN, DIM)), dtype=jnp.float32)
    mask = jnp.ones((BATCH, FRAMES, FRAME_LEN), dtype=bool)
    return tokens, mask


def _params(tokens, mask, *, live_output: bool = True):
    params = _layer().init(jax.random.key(0), tokens, mask, None)
    if live_output:
        kernel = params["params"]["out_proj"]["kernel"]
        params["params"]["out_proj"]["kernel"] = 0.1 * jax.random.normal(jax.random.key(1), kernel.shape)
    return params


def test_layer_norm_grad_is_the_gradient_of_the_inner_loss():
    rng = np.random.default_rng(0)
    x, target, scale, bias = (jnp.asarray(rng.normal(size=(4, _ttt.TTT_HEAD_DIM)), dtype=jnp.float32) for _ in range(4))

    def loss(x):
        return 0.5 * jnp.sum(jnp.square(_ttt._layer_norm(x, scale, bias)[0] - target))

    np.testing.assert_allclose(_ttt._layer_norm_l2_grad(x, target, scale, bias), jax.grad(loss)(x), rtol=1e-4, atol=1e-5)


def test_output_starts_at_zero_and_weights_start_at_the_learned_start():
    tokens, mask = _inputs()
    params = _params(tokens, mask, live_output=False)
    out, (weights, biases) = _layer().apply(params, tokens, mask, None)
    assert out.shape == tokens.shape and not np.asarray(out).any()
    assert weights.shape == (BATCH, HEADS, _ttt.TTT_HEAD_DIM, _ttt.TTT_HEAD_DIM)
    assert biases.shape == (BATCH, HEADS, 1, _ttt.TTT_HEAD_DIM)
    assert np.abs(np.asarray(weights) - np.asarray(params["params"]["w_init"])).max() > 0, "the frames stepped it"


def test_one_frame_is_one_gradient_step_on_the_inner_loss():
    tokens, mask = _inputs()
    tokens, mask = tokens[:, :FRAME_LEN], mask[:, :1]
    params = _params(tokens, mask)
    p = params["params"]
    _, (weights, biases) = _layer().apply(params, tokens, mask, None)

    def heads(name):
        return (tokens @ p[name]["kernel"]).reshape(BATCH, FRAME_LEN, HEADS, -1).transpose(0, 2, 1, 3)

    key, value = heads("k_proj"), heads("v_proj")
    gate = jax.nn.sigmoid(jnp.einsum("bsd,hd->bhs", tokens, p["lr_weight"]) + p["lr_bias"][None, :, None])
    eta = _ttt.TTT_BASE_LR / _ttt.TTT_HEAD_DIM * gate / FRAME_LEN

    def inner_loss(model):
        w, b = model
        recon = _ttt._layer_norm(key @ w + b, p["ln_scale"], p["ln_bias"])[0]
        return jnp.sum(eta * 0.5 * jnp.sum(jnp.square(recon - (value - key)), axis=-1))

    start = (jnp.broadcast_to(p["w_init"], weights.shape), jnp.broadcast_to(p["b_init"], biases.shape))
    grad_w, grad_b = jax.grad(inner_loss)(start)
    np.testing.assert_allclose(weights, start[0] - grad_w, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(biases, start[1] - grad_b, rtol=1e-4, atol=1e-6)


def test_a_stream_of_single_frame_calls_equals_one_call_over_the_frames():
    tokens, mask = _inputs()
    params = _params(tokens, mask)
    whole, whole_state = _layer().apply(params, tokens, mask, None)

    state, pieces = None, []
    for frame in range(FRAMES):
        rows = slice(frame * FRAME_LEN, (frame + 1) * FRAME_LEN)
        out, state = _layer().apply(params, tokens[:, rows], mask[:, frame : frame + 1], state)
        pieces.append(out)
    np.testing.assert_allclose(jnp.concatenate(pieces, axis=1), whole, rtol=1e-5, atol=1e-6)
    for streamed, joint in zip(state, whole_state, strict=True):
        np.testing.assert_allclose(streamed, joint, rtol=1e-5, atol=1e-6)
    assert np.abs(np.asarray(pieces[-1]) - np.asarray(
        _layer().apply(params, tokens[:, -FRAME_LEN:], mask[:, -1:], None)[0])).max() > 1e-4, \
        "the last frame's output depends on the frames before it"


def test_padding_takes_no_part_in_the_step():
    tokens, mask = _inputs()
    params = _params(tokens, mask)
    # Sample 0: the first frame is all padding (a dropped history frame), and one token of the second is.
    mask = mask.at[0, 0].set(False).at[0, 1, 2].set(False)
    _, state = _layer().apply(params, tokens, mask, None)
    scrambled = tokens.at[0, :FRAME_LEN].set(7.0).at[0, FRAME_LEN + 2].set(-3.0)
    _, scrambled_state = _layer().apply(params, scrambled, mask, None)
    for a, b in zip(state, scrambled_state, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    # A frame of padding leaves the weights where they were: skipping it gives the same state.
    _, skipped_state = _layer().apply(params, tokens[:1, FRAME_LEN:], mask[:1, 1:], None)
    for a, b in zip(state, skipped_state, strict=True):
        np.testing.assert_allclose(a[:1], b, rtol=1e-6, atol=1e-7)


def test_the_outer_gradient_reaches_every_parameter_and_is_finite():
    tokens, mask = _inputs()
    params = _params(tokens, mask)

    def outer_loss(params):
        return jnp.sum(jnp.square(_layer().apply(params, tokens, mask, None)[0]))

    grads = jax.grad(outer_loss)(params)
    for path, grad in jax.tree_util.tree_leaves_with_path(grads):
        assert np.isfinite(np.asarray(grad)).all(), path
        assert np.abs(np.asarray(grad)).max() > 0, path


def test_half_precision_tokens_come_back_in_their_dtype():
    tokens, mask = _inputs()
    params = _params(tokens, mask)
    out, state = _layer().apply(params, tokens.astype(jnp.bfloat16), mask, None)
    assert out.dtype == jnp.bfloat16 and all(part.dtype == jnp.float32 for part in state)
    assert isinstance(_layer(), nn.Module)
