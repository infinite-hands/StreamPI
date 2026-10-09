"""TEMPO's two conditioning channels (arXiv:2609.16864), ported to NNX from tempo-robot/TEMPO's
PyTorch pi0 (src/openpi/models_pytorch/pi0_pytorch.py at c88312cf): the SAM2 cross-attention fusion
(TEMPO-MOT's motion cue) and the action history + adaRMS residual (TEMPO-ACT).

Two deliberate differences, both because these configs warm-start a fine-tuned checkpoint rather than
train from pi0.5 base. The action history reaches the action expert through a zero-gated
cross-attention on its input tokens instead of as prefix tokens (GatedCrossAttn says why). And upstream zero-initialises BOTH the fusion gate and its output projection,
and since the block's output is their product, every parameter's gradient is exactly zero -- measured
on upstream's own module, 200 AdamW steps leave it at zero, so its fusion can never learn. Here only
the gate starts at zero (tanh(0) = 0 keeps the block an exact no-op at init) and the output projection
takes the default init, so the gate's gradient is the projection's output and the block can open."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

# torch.nn.LayerNorm's default epsilon, which the upstream modules use.
LAYER_NORM_EPS = 1e-5


class GatedCrossAttn(nnx.Module):
    """Zero-gated cross-attention: query tokens read a set of context tokens (projected to the query
    width, plus a per-token position embedding). A no-op at init (zero gate). Computed in float32.

    Two uses: TEMPO-MOT's SAM2 fusion (one camera's visual tokens read the SAM2 spatial tokens) and
    TEMPO-ACT here (the action expert's input tokens read the current unit's history buckets). Upstream
    puts the action history in the prefix; on a warm-started checkpoint that moved the step-0 loss from
    0.0047 to 0.54 (left real arm), and even hidden from the image and prompt tokens it cost 0.33,
    while an input gate on prefix tokens cannot help (RMSNorm rescales any non-zero gate, and a zero
    one NaN'd Gemma 2B's backward). Gating the read instead is exact at init and gradient-safe."""

    def __init__(self, query_dim: int, context_dim: int, num_tokens: int, num_heads: int, *, rngs: nnx.Rngs):
        if query_dim % num_heads:
            raise ValueError(f"query_dim {query_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.norm_q = nnx.LayerNorm(query_dim, epsilon=LAYER_NORM_EPS, rngs=rngs)
        self.kv_in = nnx.Linear(context_dim, query_dim, rngs=rngs)
        self.sam2_pos = nnx.Param(jnp.zeros((num_tokens, query_dim)))  # the name upstream's SAM2 fusion uses
        self.norm_kv = nnx.LayerNorm(query_dim, epsilon=LAYER_NORM_EPS, rngs=rngs)
        self.q_proj = nnx.Linear(query_dim, query_dim, rngs=rngs)
        self.k_proj = nnx.Linear(query_dim, query_dim, rngs=rngs)
        self.v_proj = nnx.Linear(query_dim, query_dim, rngs=rngs)
        self.out_proj = nnx.Linear(query_dim, query_dim, rngs=rngs)
        self.gate = nnx.Param(jnp.zeros((1,)))

    def __call__(self, queries: jax.Array, context: jax.Array, kv_mask: jax.Array | None = None) -> jax.Array:
        # queries: (b, nq, query_dim); context: (b, nk, context_dim); kv_mask: (b, nk), True = attend
        x = queries.astype(jnp.float32)
        b, nq, d = x.shape
        nk = context.shape[1]
        q = self.q_proj(self.norm_q(x))
        kv = self.norm_kv(self.kv_in(context.astype(jnp.float32)) + self.sam2_pos.value[None])
        k, v = self.k_proj(kv), self.v_proj(kv)
        q = q.reshape(b, nq, self.num_heads, self.head_dim)
        k = k.reshape(b, nk, self.num_heads, self.head_dim)
        v = v.reshape(b, nk, self.num_heads, self.head_dim)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.float32(self.head_dim))
        if kv_mask is not None:
            logits = jnp.where(kv_mask[:, None, None, :], logits, jnp.finfo(jnp.float32).min)
        attn = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), v).reshape(b, nq, d)
        out = jnp.tanh(self.gate.value) * self.out_proj(attn)
        if kv_mask is not None:
            out = out * jnp.any(kv_mask, axis=-1)[:, None, None]  # nothing to read: add nothing
        return queries + out.astype(queries.dtype)



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
