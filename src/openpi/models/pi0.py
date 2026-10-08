import dataclasses
import functools
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# RACE (arXiv 2610.05719, "When to Switch: Reliable Action-Chunk Extension"). Every number below is the paper's
# unless its comment says "our choice".
# Sec. 4.1, "with K = 10 denoising steps": one learnable gate alpha^k per step k (eq. 1). A flow time is mapped
# to the gate of the step whose interval [(k-1)/K, k/K) contains it (sec. 3.1), whatever step count a sampler
# runs.
RACE_NUM_GATES = 10
# Eq. 1, "alpha^k in (0, 1) is a learnable per-step gate with sigmoid activation"; sec. 4.1, "we initialize
# every gate alpha^k to 0.5": a logit of 0.
RACE_GATE_INIT_LOGIT = 0.0
# Our choice: the paper gives the head's attention no head count, so it uses one head of width d_z.
RACE_HEAD_NUM_HEADS = 1
# App. C.4, "we randomly shift the target by one action step with probability 0.5" (alg. 1: "+-1 step").
RACE_JITTER_PROB = 0.5
# Our choice: a +1 and a -1 shift are equally likely (the paper says only "+-1").
RACE_JITTER_FORWARD_PROB = 0.5
# Folded into the step's rng for the jitter draws, so the base model's noise, time and history-drop draws are
# the same with and without RACE.
_RACE_JITTER_STREAM = 0x5ACE


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def branch_isolation_mask(prefix_len: int, branches: int, branch_len: int) -> jax.Array:
    """bool[N, N] over a prefix followed by `branches` suffix blocks of `branch_len` tokens: False only
    where one suffix block would look at another. ANDed onto make_attn_mask's block-causal mask it gives
    VLASH's shared-observation attention (arXiv 2512.01031, fig. 4): every branch sees the prefix and
    itself, never a sibling; the prefix never sees a suffix."""
    total = prefix_len + branches * branch_len
    branch_of = jnp.concatenate([jnp.full((prefix_len,), -1), jnp.repeat(jnp.arange(branches), branch_len)])
    suffix = jnp.arange(total) >= prefix_len
    cross_branch = suffix[:, None] & suffix[None, :] & (branch_of[:, None] != branch_of[None, :])
    return ~cross_branch


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False],
                      race=config.race)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.state_cond = config.state_cond
        self.state_cond_dims = config.state_cond_dims
        self.vlash_branches = config.vlash_branches
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            if config.state_cond:
                # Zero-initialised output: at the first step the conditioning is the time MLP's alone, so a
                # pi0.5 checkpoint without these weights keeps its behaviour until fine-tuning moves them.
                self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                self.state_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                self.state_mlp_out = nnx.Linear(
                    action_expert_config.width, action_expert_config.width, rngs=rngs,
                    kernel_init=nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.race = config.race
        if config.race:
            self.race_aux_weight = config.race_aux_weight
            self.race_timing_weight = config.race_timing_weight
            width = action_expert_config.width                                   # d_a
            prefix_width = paligemma_config.num_kv_heads * paligemma_config.head_dim  # d_z: a cached value's width
            # Eq. 5's e_trans, "a randomly initialized learnable transition embedding"; N(0, 1) is our choice.
            self.race_embedding = nnx.Param(jax.random.normal(rngs.params(), (width,)))
            self.race_gate_logits = nnx.Param(jnp.full((RACE_NUM_GATES,), RACE_GATE_INIT_LOGIT))
            # Eq. 3's head: W_f, a cross-attention, a self-attention and w_p. Our choices where the paper is
            # silent: flax's default initialisers, biases on every projection, float32 throughout.
            self.race_head_proj = nnx.Linear(width, prefix_width, rngs=rngs)
            self.race_head_cross = nnx.MultiHeadAttention(RACE_HEAD_NUM_HEADS, prefix_width, decode=False, rngs=rngs)
            self.race_head_self = nnx.MultiHeadAttention(RACE_HEAD_NUM_HEADS, prefix_width, decode=False, rngs=rngs)
            self.race_head_out = nnx.Linear(prefix_width, 1, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        self.hist_horizon = config.hist_horizon
        self.image_keys = tuple(config.image_keys)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int, int]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        all_image_tokens = dict()

        for name in self.image_keys:
            image = obs.images[name]
            bs, T = image.shape[0], image.shape[1]
            image = image.reshape(bs*T, image.shape[2], image.shape[3], image.shape[4])
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            image_tokens = image_tokens.reshape(bs, T, image_tokens.shape[1], image_tokens.shape[2])
    
            all_image_tokens[name] = image_tokens

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            num_text_tokens = tokenized_inputs.shape[1]
        else:
            num_text_tokens = 0

        for t in range(T):
            visual_tokens = list()
            visual_input_mask = list()
            for name in self.image_keys:
                visual_tokens.append(all_image_tokens[name][:, t])

                visual_input_mask.append(einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=visual_tokens[0].shape[1],
                ))

            visual_tokens = jnp.concatenate(visual_tokens, axis=1)
            visual_input_mask = jnp.concatenate(visual_input_mask, axis=1)

            multimodal_tokens = jnp.concatenate([visual_tokens, tokenized_inputs], axis=1)
            multimodal_input_mask = jnp.concatenate([visual_input_mask, obs.tokenized_prompt_mask], axis=1)

            # image tokens attend to each other
            ar_mask += [True] + [False] * (visual_tokens.shape[1] - 1)

            # full attention between image and language inputs
            ar_mask += [False] * (tokenized_inputs.shape[1])

            tokens.append(multimodal_tokens)
            input_mask.append(multimodal_input_mask)

        num_visual_tokens = tokens[0].shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1).reshape(bs, -1)
    
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, num_visual_tokens, T

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b ah"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = jax.vmap(
            functools.partial(
                posemb_sincos, embedding_dim=self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
        )(timestep)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
            if self.state_cond:
                adarms_cond = adarms_cond + self._state_cond_emb(obs.state)[:, None, :]
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _state_cond_emb(self, state: at.Float[at.Array, "b s"]) -> at.Float[at.Array, "b emb"]:
        if self.state_cond_dims is not None:
            keep = jnp.zeros((state.shape[-1],), dtype=bool).at[jnp.array(self.state_cond_dims)].set(True)
            state = jnp.where(keep, state, 0.0)
        emb = self.state_mlp_in(self.state_proj(state))
        emb = nnx.swish(emb)
        emb = self.state_mlp_out(emb)
        return nnx.swish(emb)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """The flow-matching loss per action row (RACE's L_full for a RACE model)."""
        if actions.ndim == 4:
            return self._compute_loss_branches(rng, observation, actions, train=train)
        return self._compute_losses(rng, observation, actions, train=train)[0]

    @override
    def training_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        if not self.race:
            return super().training_loss(rng, observation, actions)
        full, aux, timing = self._compute_losses(rng, observation, actions, train=True)
        flow_loss, aux_loss = jnp.mean(full), jnp.mean(aux)
        # Sec. 3.3: L = L_full + lambda_aux L_aux + lambda_timing L_timing.
        loss = flow_loss + self.race_aux_weight * aux_loss + self.race_timing_weight * timing
        return loss, {"flow_loss": flow_loss, "aux_loss": aux_loss, "timing_loss": timing}

    def _compute_losses(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool
    ):
        """(L_full per row, L_aux per row, L_timing); the last two are None for a model without RACE."""
        preprocess_rng, noise_rng, time_rng, mask_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, image_keys=self.image_keys)

        b, ah, ad = actions.shape
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (b, 1)) * 0.999 + 0.001
        time = jnp.broadcast_to(time, (b, ah))

        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, num_img_tokens, T = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)

        mask_num = jax.random.randint(mask_rng, (), 0, T)
        mask_tokens = mask_num * num_img_tokens
        seq_len = input_mask.shape[1]
        indices = jnp.arange(seq_len)
        
        to_mask = ~(indices < mask_tokens)
        input_mask = input_mask & to_mask[None, :]

        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        def safe_print(fmt, *args):
            if jax.process_index() == 0:
                jax.debug.print(fmt, *args)
        
        # safe_print("tokens_num {} {}", num_img_tokens, attn_mask.shape)
        # safe_print("{} {}", input_mask.shape, ar_mask.shape)
        # safe_print("x1, 1 = {}", attn_mask[0, 0:num_img_tokens, 0:num_img_tokens].sum())
        # safe_print("x2, 0 = {}", attn_mask[0, 0:num_img_tokens, num_img_tokens:num_img_tokens*2].sum())
        # safe_print("mask_num {} {}", mask_num, attn_mask.shape)
        # safe_print("x1, x0 = {}", attn_mask[0, 768:768*2, 0:768].sum())
        # safe_print("x2, x1 = {}", attn_mask[0, 768*2:768*3, 768:768*2].sum())
        # # safe_print("y2, 1 = {}", attn_mask[0, 768:768*2, 768:768*2].sum())
        # # safe_print("y3, 0 = {}", attn_mask[0, 768:768*2, 768*2:].sum())
        # # safe_print("x1, x2 = {}", attn_mask[0, 768:768*2, 768*2:768*3].sum())
        # # safe_print("x2, x0 = {}", attn_mask[0, 768*2:768*3, 0:768].sum())
        # # safe_print("x2, x3 = {}", attn_mask[0, 768*2:768*3, 768*3:768*3 + 10].sum())
        # safe_print("x3, x2 = {}", attn_mask[0, 768*3:768*3 + 10, 768*2:768*3].sum())

        race_cond = None
        if self.race:
            # Sec. 3.3 / alg. 1: the full pass is conditioned on the jittered target (teacher forcing), with
            # the gate of the denoising step whose interval contains the sampled flow time.
            race_cond = self.race_conditioning(self._race_teacher_prior(rng, observation, train=train),
                                               self._race_gate_for_training(time[:, 0]))

        (prefix_out, suffix_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond],
            race_cond=race_cond,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        full = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if not self.race:
            return full, None, None

        # Sec. 3.2 / alg. 1 lines 5-10: the auxiliary pass, one unmodulated step from the same noise at the
        # paper's s = 0 (t = 1 here), over the prefix this pass just cached -- the inference path's own shape.
        # Prefix tokens never attend to the suffix, so their cached keys and values are the prefix's alone.
        prefix_len = prefix_tokens.shape[1]
        prefix_mask = input_mask[:, :prefix_len]
        prefix_kv = jax.tree.map(lambda cache: cache[:, :, :prefix_len], kv_cache)
        v_aux, hidden = self._suffix_pass(observation, prefix_mask, prefix_kv, noise, jnp.ones_like(time))
        aux = jnp.mean(jnp.square(v_aux - u_t), axis=-1)
        # The head reads the current frame only: the last of the T frame blocks, as a serve call's prefix is.
        current = (T - 1) * num_img_tokens
        logits = self._race_head_logits(hidden, self._race_prefix_values(prefix_kv, current, num_img_tokens),
                                        prefix_mask[:, current:])
        return full, aux, self._race_timing_loss(logits, observation)

    def _compute_loss_branches(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: at.Float[at.Array, "b n ah ad"],
        *, train: bool,
    ) -> at.Float[at.Array, "b n ah"]:
        """VLASH's shared-observation loss: one prefix (images + prompt) for `n` (state, action chunk)
        branches at different temporal offsets. Each branch's action tokens attend to the prefix and to
        themselves, never to a sibling branch, and every branch's positions restart at the prefix's end,
        so the pass equals `n` separate samples that share the observation -- encoded once."""
        if not self.state_cond:
            raise ValueError("branched samples need state_cond: the prompt carries no per-branch state")
        preprocess_rng, noise_rng, time_rng, mask_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, image_keys=self.image_keys)

        b, n, ah, ad = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (b, n, 1)) * 0.999 + 0.001
        time = jnp.broadcast_to(time, (b, n, ah))

        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, num_img_tokens, T = self.embed_prefix(observation)
        flat = dataclasses.replace(observation, state=observation.state.reshape(b * n, -1))
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            flat, x_t.reshape(b * n, ah, ad), time.reshape(b * n, ah))
        suffix_len = suffix_tokens.shape[1]
        suffix_tokens = suffix_tokens.reshape(b, n * suffix_len, -1)
        suffix_mask = suffix_mask.reshape(b, n * suffix_len)
        adarms_cond = adarms_cond.reshape(b, n * suffix_len, -1)

        # The same history drop as the plain loss, on the prefix alone.
        mask_num = jax.random.randint(mask_rng, (), 0, T)
        prefix_len = prefix_mask.shape[1]
        prefix_mask = prefix_mask & ~(jnp.arange(prefix_len) < mask_num * num_img_tokens)[None, :]

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.tile(suffix_ar_mask, n)], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask) & branch_isolation_mask(prefix_len, n, suffix_len)[None]
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        suffix_positions = jnp.sum(prefix_mask, axis=1)[:, None] + jnp.tile(jnp.arange(suffix_len), n)[None, :]
        positions = jnp.concatenate([prefix_positions, suffix_positions], axis=1)

        (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out).reshape(b, n, suffix_len, ad)[:, :, -ah:]
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @at.typecheck
    def embed_prefix_infer(
        self, obs: _model.Observation, memory_tokens
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int, int]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        all_image_tokens = dict()

        for name in self.image_keys:
            image = obs.images[name]
            bs, T = image.shape[0], image.shape[1]
            image = image.reshape(bs*T, image.shape[2], image.shape[3], image.shape[4])
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            image_tokens = image_tokens.reshape(bs, T, image_tokens.shape[1], image_tokens.shape[2])
    
            all_image_tokens[name] = image_tokens

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            num_text_tokens = tokenized_inputs.shape[1]
        else:
            num_text_tokens = 0

        for t in range(T):
            visual_tokens = list()
            visual_input_mask = list()
            for name in self.image_keys:
                visual_tokens.append(all_image_tokens[name][:, t])

                visual_input_mask.append(einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=visual_tokens[0].shape[1],
                ))

            visual_tokens = jnp.concatenate(visual_tokens, axis=1)
            visual_input_mask = jnp.concatenate(visual_input_mask, axis=1)

            multimodal_tokens = jnp.concatenate([visual_tokens, tokenized_inputs], axis=1)
            multimodal_input_mask = jnp.concatenate([visual_input_mask, obs.tokenized_prompt_mask], axis=1)

            # image tokens attend to each other
            ar_mask += [True] + [False] * (visual_tokens.shape[1] - 1)

            # full attention between image and language inputs
            ar_mask += [False] * (tokenized_inputs.shape[1])

            tokens.append(multimodal_tokens)
            input_mask.append(multimodal_input_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1).reshape(bs, -1)
    
        num_visual_tokens = tokens.shape[1]

        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, num_visual_tokens, num_text_tokens

    def reset_memory(self, memory):
        memory["memory_tokens"] = None
        memory["memory_kv_cache"] = None
        memory["memory_prefix_mask"] = None

    @at.typecheck
    def save_memory(
        self, prefix_tokens: at.Float[at.Array, "b s1 emb"], prefix_mask: at.Bool[at.Array, "b s2"], num_visual_tokens: int, num_text_tokens: int, kv_cache, memory: dict
    ):
        memory["memory_tokens"] = prefix_tokens

        memory_kv_cache = list()
        for cache in kv_cache:
            memory_kv_cache.append(cache)

        memory["memory_kv_cache"] = tuple(memory_kv_cache)
        memory["memory_prefix_mask"] = prefix_mask

    def fetch_memory(self, memory):
        memory_tokens = memory["memory_tokens"]
        memory_kv_cache = memory["memory_kv_cache"]
        memory_prefix_mask = memory["memory_prefix_mask"]
        return None, memory_kv_cache, memory_prefix_mask

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        step: at.Int[at.Array, ""] = None,
        memory: dict = None
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self.image_keys)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
 
        memory_tokens, memory_kv_cache, memory_prefix_mask = self.fetch_memory(memory)

        # if jax.process_index() == 0:
        #     jax.debug.print("x1, {}, {}", memory_kv_cache is None, memory_prefix_mask is None)

        prefix_tokens, prefix_mask, prefix_ar_mask, num_visual_tokens, num_text_tokens = self.embed_prefix_infer(observation, memory_tokens)

        # prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        if memory_prefix_mask is not None:
            memory_prefix_attn_mask = einops.repeat(memory_prefix_mask, "b p -> b s p", s=prefix_mask.shape[1])
            prefix_attn_mask = jnp.concatenate([memory_prefix_attn_mask, prefix_attn_mask], axis=-1)
            prefix_mask = jnp.concatenate([memory_prefix_mask, prefix_mask], axis=-1)
        # memory_kv_cache = None
            # import pdb;pdb.set_trace()
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            positions = positions[..., memory_prefix_mask.shape[-1]:] 
        else:
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions, kv_cache=memory_kv_cache)
        self.save_memory(prefix_tokens, prefix_mask, num_visual_tokens, num_text_tokens, kv_cache, memory)

        # RACE (alg. 2): the prior, once per call, from this call's prefix and the noise the loop starts from.
        prior = None
        if self.race:
            prior = self.race_prior(observation, prefix_mask, kv_cache, noise, current_prefix_len=prefix_tokens.shape[1])

        def step(carry):
            x_t, time = carry
            time_ = jnp.broadcast_to(time, batch_size)  # (b, )
            race_cond = None if prior is None else self.race_step_conditioning(prior, time, num_steps)
            v_t, _ = self._suffix_pass(observation, prefix_mask, kv_cache, x_t, time_[:, None], race_cond)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        with at.disable_typechecking():
            x_0, _ = jax.lax.while_loop(cond, step, (noise, jnp.float32(1.0)))
        if prior is None:
            return x_0, memory
        # A RACE model also returns the head's scores for the rows of x_0 (Policy.infer: "transition_scores").
        return x_0, memory, prior

    def _suffix_pass(self, observation, prefix_mask, kv_cache, x_t, timestep, race_cond=None):
        """One action-expert pass over a cached prefix: (velocity, the final-layer hidden states of the action
        rows). `observation` is preprocessed; `prefix_mask` (b, cached tokens) covers every cached token."""
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep)
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            race_cond=race_cond,
        )
        assert prefix_out is None
        hidden = suffix_out[:, -self.action_horizon :]
        return self.action_out_proj(hidden), hidden

    # --- RACE. The public methods are the API an external denoising loop (e.g. a guided RTC sampler) uses:
    #   prior = model.race_prior(obs, prefix_mask, kv_cache, noise, current_prefix_len=...)   # once per call
    #   race_cond = model.race_step_conditioning(prior, time, num_steps)                     # every step
    #   model.PaliGemma.llm([None, suffix_tokens], ..., race_cond=race_cond)

    def race_prior(
        self,
        observation: _model.Observation,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache,
        noise: at.Float[at.Array, "b ah ad"],
        *,
        current_prefix_len: int,
    ) -> at.Float[at.Array, "b ah"]:
        """Alg. 2 lines 5-7: the transition-timing prior p in [0, 1]^H for one policy call. `observation` is
        preprocessed; `prefix_mask` and `kv_cache` are the prefix pass's (streaming memory first, then this
        call's prefix, which is the last `current_prefix_len` cached tokens -- the head reads only those);
        `noise` is the A^0 the denoising loop starts from."""
        _v_aux, hidden = self._suffix_pass(
            observation, prefix_mask, kv_cache, noise, jnp.ones(noise.shape[:2], dtype=jnp.float32))
        current = prefix_mask.shape[1] - current_prefix_len
        logits = self._race_head_logits(
            hidden, self._race_prefix_values(kv_cache, current, current_prefix_len), prefix_mask[:, current:])
        return jax.nn.sigmoid(logits)

    def race_step_conditioning(
        self, prior: at.Float[at.Array, "b ah"], time, num_steps
    ) -> at.Float[at.Array, "b ah emb"]:
        """Eq. 5's U^k for the Euler step of a `num_steps`-step loop that starts at flow time `time` (this
        repo's convention, 1 = noise): pass it to the action expert as `race_cond`."""
        return self.race_conditioning(prior, race_gate_index(time, num_steps))

    def race_conditioning(self, prior: at.Float[at.Array, "b ah"], gate) -> at.Float[at.Array, "b ah emb"]:
        """Eq. 5: U^k = alpha^k p (outer) e_trans, for gate index `gate` (a scalar or one per sample)."""
        alpha = jnp.broadcast_to(jax.nn.sigmoid(self.race_gate_logits.value)[gate], prior.shape[:1])
        return (alpha[:, None] * prior)[..., None] * self.race_embedding.value

    @staticmethod
    def _race_gate_for_training(time: at.Float[at.Array, " b"]) -> at.Int[at.Array, " b"]:
        """Alg. 1 line 13, k = floor(K s) + 1 with s = 1 - t (the paper's flow time runs noise 0 -> data 1)."""
        return jnp.clip(jnp.floor(RACE_NUM_GATES * (1.0 - time)).astype(jnp.int32), 0, RACE_NUM_GATES - 1)

    @staticmethod
    def _race_prefix_values(kv_cache, start: int, length: int) -> at.Float[at.Array, "b s d"]:
        """z_hat: the VLM's final-layer cached values of tokens [start, start + length), (b, length, d_z)."""
        values = kv_cache[1][-1][:, start : start + length]
        return values.reshape(*values.shape[:2], -1)

    def _race_head_logits(self, action_hidden, prefix_values, prefix_values_mask) -> at.Float[at.Array, "b ah"]:
        """Eq. 3 up to the sigmoid: F_hat = F W_f; F_bar = F_hat + CrossAttn(F_hat, z, z); F_tilde = F_bar +
        SelfAttn(F_bar); logits = F_tilde w_p."""
        features = self.race_head_proj(action_hidden.astype(jnp.float32))
        prefix_values = prefix_values.astype(jnp.float32)
        features = features + self.race_head_cross(
            features, prefix_values, prefix_values, mask=prefix_values_mask[:, None, None, :])
        features = features + self.race_head_self(features)
        return self.race_head_out(features)[..., 0]

    def _race_window(self, observation: _model.Observation) -> tuple[at.Float[at.Array, "b r"], at.Bool[at.Array, "b r"]]:
        if observation.transition_window is None or observation.transition_window_mask is None:
            raise ValueError("a RACE model trains on transition targets: set the data config's race_targets_dir")
        if observation.transition_window.shape[-1] != self.action_horizon + 2:
            raise ValueError(f"transition_window has {observation.transition_window.shape[-1]} rows; a RACE model "
                             f"of horizon {self.action_horizon} needs {self.action_horizon + 2} (rows i-1 .. i+H)")
        return observation.transition_window.astype(jnp.float32), observation.transition_window_mask

    def _race_teacher_prior(self, rng, observation: _model.Observation, *, train: bool) -> at.Float[at.Array, "b ah"]:
        """Sec. 3.3 / app. C.4: training conditions on the target (teacher forcing), shifted by one row with
        probability 0.5 when `train` (jittering). A frame outside the anchor's episode conditions as 0 (app. A.4:
        steps beyond the end of an episode "receive a score of zero")."""
        window, inside = self._race_window(observation)
        window = jnp.where(inside, window, 0.0)
        batch = window.shape[0]
        shift = jnp.zeros((batch,), dtype=jnp.int32)
        if train:
            jitter_rng, direction_rng = jax.random.split(jax.random.fold_in(rng, _RACE_JITTER_STREAM))
            forward = jax.random.bernoulli(direction_rng, RACE_JITTER_FORWARD_PROB, (batch,))
            shift = jnp.where(jax.random.bernoulli(jitter_rng, RACE_JITTER_PROB, (batch,)),
                              jnp.where(forward, 1, -1), 0)
        # Row h of the conditioning is target row h - shift; the window's row h + 1 is target row h.
        rows = jnp.arange(self.action_horizon)[None, :] + 1 - shift[:, None]
        return jnp.take_along_axis(window, rows, axis=1)

    def _race_timing_loss(self, logits: at.Float[at.Array, "b ah"], observation: _model.Observation):
        """Eq. 4: the binary cross-entropy of the prior against the unjittered soft target, averaged over the
        chunk rows inside the anchor's episode. Our contract leaves rows past the episode end out of the loss;
        the paper instead scores them 0 (app. A.4)."""
        window, inside = self._race_window(observation)
        target = window[:, 1 : self.action_horizon + 1]
        counted = inside[:, 1 : self.action_horizon + 1]
        bce = -(target * jax.nn.log_sigmoid(logits) + (1.0 - target) * jax.nn.log_sigmoid(-logits))
        return jnp.sum(jnp.where(counted, bce, 0.0)) / jnp.maximum(jnp.sum(counted), 1)


def race_gate_index(time, num_steps):
    """The gate of the Euler step that starts at flow time `time` (1 = noise) in a `num_steps`-step loop: that
    step starts at the paper's s = (k-1)/num_steps, and its gate is the one whose [(j-1)/K, j/K) contains s
    (sec. 3.1). The step index is recovered by rounding, so the loop's accumulated float error in `time` can
    never move a step across a gate boundary."""
    step = jnp.round((1.0 - jnp.asarray(time, dtype=jnp.float32)) * num_steps).astype(jnp.int32)
    return jnp.clip((RACE_NUM_GATES * step) // num_steps, 0, RACE_NUM_GATES - 1)