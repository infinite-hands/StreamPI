"""TEMPO's two conditioning channels (arXiv:2609.16864), ported to NNX from tempo-robot/TEMPO's
PyTorch pi0 (src/openpi/models_pytorch/pi0_pytorch.py at c88312cf): the SAM2 cross-attention fusion
(TEMPO-MOT's motion cue) and the action-history tokens + adaRMS residual (TEMPO-ACT).

One deliberate difference: upstream zero-initialises BOTH the fusion gate and its output projection,
and since the block's output is their product, every parameter's gradient is exactly zero -- measured
on upstream's own module, 200 AdamW steps leave it at zero, so its fusion can never learn. Here only
the gate starts at zero (tanh(0) = 0 keeps the block an exact no-op at init) and the output projection
takes the default init, so the gate's gradient is the projection's output and the block can open."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

# torch.nn.LayerNorm's default epsilon, which the upstream modules use.
LAYER_NORM_EPS = 1e-5


class Sam2CrossAttnFusion(nnx.Module):
    """Zero-gated cross-attention: one camera's visual tokens (queries) read the SAM2 spatial tokens
    (keys/values). A no-op at init (zero gate). Computed in float32."""

    def __init__(self, vis_dim: int, sam2_dim: int, num_tokens: int, num_heads: int, *, rngs: nnx.Rngs):
        if vis_dim % num_heads:
            raise ValueError(f"vis_dim {vis_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = vis_dim // num_heads
        self.norm_q = nnx.LayerNorm(vis_dim, epsilon=LAYER_NORM_EPS, rngs=rngs)
        self.kv_in = nnx.Linear(sam2_dim, vis_dim, rngs=rngs)
        self.sam2_pos = nnx.Param(jnp.zeros((num_tokens, vis_dim)))
        self.norm_kv = nnx.LayerNorm(vis_dim, epsilon=LAYER_NORM_EPS, rngs=rngs)
        self.q_proj = nnx.Linear(vis_dim, vis_dim, rngs=rngs)
        self.k_proj = nnx.Linear(vis_dim, vis_dim, rngs=rngs)
        self.v_proj = nnx.Linear(vis_dim, vis_dim, rngs=rngs)
        self.out_proj = nnx.Linear(vis_dim, vis_dim, rngs=rngs)
        self.gate = nnx.Param(jnp.zeros((1,)))

    def __call__(self, vis_tokens: jax.Array, sam2_tokens: jax.Array) -> jax.Array:
        # vis_tokens: (b, nv, vis_dim); sam2_tokens: (b, ns, sam2_dim)
        vis = vis_tokens.astype(jnp.float32)
        b, nv, d = vis.shape
        ns = sam2_tokens.shape[1]
        q = self.q_proj(self.norm_q(vis))
        kv = self.norm_kv(self.kv_in(sam2_tokens.astype(jnp.float32)) + self.sam2_pos.value[None])
        k, v = self.k_proj(kv), self.v_proj(kv)
        q = q.reshape(b, nv, self.num_heads, self.head_dim)
        k = k.reshape(b, ns, self.num_heads, self.head_dim)
        v = v.reshape(b, ns, self.num_heads, self.head_dim)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.float32(self.head_dim))
        attn = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), v).reshape(b, nv, d)
        out = jnp.tanh(self.gate.value) * self.out_proj(attn)
        return vis_tokens + out.astype(vis_tokens.dtype)


class ActionHistoryTokens(nnx.Module):
    """Each bucket-mean past action projected to one VLM token: a shared linear plus a per-bucket
    position embedding (zero-init).

    Upstream trains from pi0.5 base; here the configs warm-start a fine-tuned checkpoint, where these
    tokens in plain prefix attention moved the step-0 loss from 0.0047 to 0.54 (left real arm). So Pi0
    hides them from the image and prompt tokens and leaves them out of the positions: only the action
    expert reads them. There is deliberately no zero-init gate on them: an all-zero token stream sits
    where RMSNorm's backward is 1/sqrt(eps) per layer, and with the gate at zero, d loss / d gate
    measured 1e8 on the four-layer test model and NaN on Gemma 2B's eighteen."""

    def __init__(self, in_dim: int, num_tokens: int, vlm_dim: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Linear(in_dim, vlm_dim, rngs=rngs)
        self.pos_emb = nnx.Param(jnp.zeros((num_tokens, vlm_dim)))

    def __call__(self, history: jax.Array) -> jax.Array:
        # (b, k, in_dim) -> (b, k, vlm_dim)
        return self.proj(history.astype(jnp.float32)) + self.pos_emb.value[None]


class ActionHistoryCondMLP(nnx.Module):
    """in -> hidden -> silu -> hidden -> silu, the last layer zero-init: a zero residual on the
    action expert's adaRMS conditioning at init."""

    def __init__(self, in_dim: int, hidden: int, *, rngs: nnx.Rngs):
        self.in_proj = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.mlp_in = nnx.Linear(hidden, hidden, rngs=rngs)
        self.mlp_out = nnx.Linear(hidden, hidden, kernel_init=nnx.initializers.zeros,
                                  bias_init=nnx.initializers.zeros, rngs=rngs)

    def __call__(self, flat_history: jax.Array) -> jax.Array:
        h = self.in_proj(flat_history.astype(jnp.float32))
        h = nnx.silu(self.mlp_in(h))
        return nnx.silu(self.mlp_out(h))
