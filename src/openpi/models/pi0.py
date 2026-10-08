import functools
import logging
from typing import NamedTuple

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import lit as _lit
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


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


class LitPrefix(NamedTuple):
    """What the action rows read in a LIT step, from `Pi0._lit_prefix_pass`."""

    # Stacked (k, v) of the prefix pass, each (layers, b, s, kv_heads, head_dim).
    kv_cache: tuple
    # bool (b, s): the prefix columns the action rows may attend (history drop and the LIT masks applied).
    visible: jax.Array
    # int (b,): valid prefix tokens; the suffix positions continue from here.
    count: jax.Array
    # (k, v, visible) of the columns appended after the prefix, k and v (layers, b, e, kv_heads, head_dim): the latents
    # (stage 2) or the goal tokens (stage 1).
    extra: tuple
    # Final latents (b, num_latents, lit_dim) for the pose decoder; None in stage 1.
    latents: jax.Array | None


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
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
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
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        self.hist_horizon = config.hist_horizon

        # LIT modules come last so that lit="off" keeps the stock parameter tree and rng stream.
        self.lit = config.lit
        if config.lit != "off":
            self.lit_goal_dims = tuple(config.lit_goal_dims)
            self.lit_mask_image = config.lit_mask_image
            self.lit_mask_language = config.lit_mask_language
            self.lit_pose_weight = config.lit_pose_weight
            self.lit_kv_heads = paligemma_config.num_kv_heads
            self.lit_head_dim = paligemma_config.head_dim
            if config.lit == "stage1":
                self.lit_goal_encoder = _lit.LitGoalEncoder(
                    pose_dim=len(config.lit_goal_dims),
                    num_tokens=config.lit_goal_tokens,
                    dim=config.lit_dim,
                    kv_dim=config.lit_kv_dim,
                    rngs=rngs,
                )
            else:
                self.lit_aggregator = _lit.LitAggregator(
                    num_latents=config.lit_num_latents,
                    dim=config.lit_dim,
                    context_dim=paligemma_config.width,
                    kv_dim=config.lit_kv_dim,
                    num_heads=config.lit_heads,
                    num_groups=config.lit_groups,
                    rngs=rngs,
                    dtype=jnp.dtype(config.dtype),
                    remat=True,
                )
                self.lit_pose_decoder = _lit.LitPoseDecoder(
                    pose_dim=len(config.lit_goal_dims),
                    num_tokens=config.lit_pose_tokens,
                    dim=config.lit_dim,
                    rngs=rngs,
                )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int, int]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        all_image_tokens = dict()

        for name in obs.images:
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
            for name in obs.images:
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

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.lit != "off":
            return self._compute_loss_lit(rng, observation, actions, train=train)[0]
        preprocess_rng, noise_rng, time_rng, mask_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

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

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def compute_loss_and_aux(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict]:
        """`compute_loss` plus scalar diagnostics: for lit != "off" {"pose_loss" (stage 2 only), "pose_copy_baseline"},
        both unweighted means over the samples whose goal is real (weight: config.lit_pose_weight); {} for lit="off"."""
        if self.lit == "off":
            return self.compute_loss(rng, observation, actions, train=train), {}
        return self._compute_loss_lit(rng, observation, actions, train=train, with_aux=True)

    @at.typecheck
    def embed_prefix_text(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int, int]:
        """Stage 1 prefix: the language/state tokens as one block, no images. Same 5-tuple as `embed_prefix`."""
        tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        ar_mask = jnp.array([True] + [False] * (tokens.shape[1] - 1))
        return tokens, obs.tokenized_prompt_mask, ar_mask, tokens.shape[1], 1

    @at.typecheck
    def prefix_roles(
        self, obs: _model.Observation, num_block_tokens: int, num_blocks: int, *, ignore_validity: bool = False
    ) -> at.Int[at.Array, "b s"]:
        """Role of every prefix token (lit.ROLE_PAD / ROLE_IMAGE / ROLE_SEMANTIC) for `num_blocks` blocks of
        [camera 0 | camera 1 | ... | language+state] of `num_block_tokens` tokens, in the order embed_prefix lays them
        out. Image tokens are valid where their camera's mask is set, language/state tokens where the prompt mask is;
        `ignore_validity` returns the layout's roles with every token counted valid (what a column is, whatever
        its observation had masked)."""
        if obs.tokenized_prompt_mask is None:
            raise ValueError("LIT needs the tokenized prompt.")
        b = obs.state.shape[0]
        text_len = obs.tokenized_prompt_mask.shape[1]
        per_camera, rest = divmod(num_block_tokens - text_len, len(obs.images))
        if rest or per_camera < 0:
            raise ValueError(
                f"{num_block_tokens} tokens per block do not fit {len(obs.images)} cameras + {text_len} text."
            )
        block = []
        if per_camera:
            for name in obs.images:
                valid = jnp.broadcast_to(obs.image_masks[name][..., None] | ignore_validity, (b, per_camera))
                block.append(jnp.where(valid, _lit.ROLE_IMAGE, _lit.ROLE_PAD))
        text_valid = jnp.broadcast_to(obs.tokenized_prompt_mask | ignore_validity, (b, text_len))
        block.append(jnp.where(text_valid, _lit.ROLE_SEMANTIC, _lit.ROLE_PAD))
        return jnp.tile(jnp.concatenate(block, axis=1), (1, num_blocks)).astype(jnp.int32)

    def _lit_goal(self, obs: _model.Observation):
        """(goal [b, len(lit_goal_dims)] float32 with padded rows zeroed, valid bool[b])."""
        if obs.lit_goal is None:
            raise ValueError("lit != 'off' needs observation.lit_goal.")
        if obs.lit_goal.shape[-1] != self.action_dim:
            raise ValueError(f"lit_goal must be {self.action_dim} wide like the state, got {obs.lit_goal.shape}.")
        b = obs.state.shape[0]
        valid = jnp.ones((b,), bool) if obs.lit_goal_mask is None else jnp.broadcast_to(obs.lit_goal_mask, (b,))
        goal = jnp.broadcast_to(obs.lit_goal, (b, self.action_dim))[:, jnp.asarray(self.lit_goal_dims)]
        return jnp.where(valid[:, None], goal.astype(jnp.float32), 0.0), valid

    def _lit_kv_columns(self, k, v):
        """[layers, b, e, lit_kv_dim] -> the kv-cache layout [layers, b, e, kv_heads, head_dim]."""
        shape = (*k.shape[:3], self.lit_kv_heads, self.lit_head_dim)
        return k.reshape(shape), v.reshape(shape)

    def _lit_prefix_pass(self, observation: _model.Observation, mask_rng=None, *, mask_num=None) -> LitPrefix:
        """Prefix pass, then the aggregator (stage 2) or the goal encoder (stage 1).

        The first `mask_num` history blocks are dropped from the prefix mask, as in the stock loss: mask_num is drawn
        from `mask_rng` as there, or 0 when neither is given. Only the current (last) block feeds the aggregator."""
        stage1 = self.lit == "stage1"
        embed = self.embed_prefix_text if stage1 else self.embed_prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, block_len, blocks = embed(observation)
        roles = self.prefix_roles(observation, block_len, blocks)
        if mask_num is None:
            mask_num = 0 if mask_rng is None else jax.random.randint(mask_rng, (), 0, blocks)
        prefix_mask = prefix_mask & (jnp.arange(prefix_mask.shape[1]) >= mask_num * block_len)[None, :]
        roles = jnp.where(prefix_mask, roles, _lit.ROLE_PAD)

        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        count = jnp.sum(prefix_mask, axis=1)
        if stage1:
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
            goal, valid = self._lit_goal(observation)
            goal_k, goal_v = self.lit_goal_encoder.project_kv(self.lit_goal_encoder(goal))
            layers = kv_cache[0].shape[0]
            k, v = (jnp.broadcast_to(x[None], (layers, *x.shape)) for x in (goal_k, goal_v))
            extra = (*self._lit_kv_columns(k, v), jnp.broadcast_to(valid[:, None], goal_k.shape[:2]))
            return LitPrefix(kv_cache, prefix_mask, count, extra, None)

        start = (blocks - 1) * block_len
        _, kv_cache, layer_inputs = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attn_mask,
            positions=positions,
            return_layer_inputs=True,
            layer_input_slice=(start, block_len),
        )
        semantic_mask, image_mask = _lit.role_masks(roles[:, start:])
        latents, keys, values = self.lit_aggregator(layer_inputs, semantic_mask, image_mask)
        extra = (*self._lit_kv_columns(keys, values), jnp.ones(keys.shape[1:3], bool))
        return LitPrefix(kv_cache, self._lit_visible(prefix_mask, roles), count, extra, latents)

    def _lit_visible(self, prefix_mask, roles):
        """The valid prefix columns the action rows may attend: the image and language/state ones are hidden per the
        stage-2 switches. `roles` has the columns' roles (validity does not matter, `prefix_mask` carries it)."""
        visible = prefix_mask
        if self.lit_mask_image:
            visible = visible & (roles != _lit.ROLE_IMAGE)
        if self.lit_mask_language:
            visible = visible & (roles != _lit.ROLE_SEMANTIC)
        return visible

    @staticmethod
    def _lit_extend_cache(kv_cache, visible, extra):
        """Append `extra` = (k, v, visible) after the cache columns, k and v cast to the cache dtype."""
        extra_k, extra_v, extra_visible = extra
        kv_cache = tuple(
            jnp.concatenate([cache, e.astype(cache.dtype)], axis=2)
            for cache, e in zip(kv_cache, (extra_k, extra_v), strict=True)
        )
        return kv_cache, jnp.concatenate([visible, extra_visible], axis=1)

    def _suffix_velocity(self, observation, x_t, time, kv_cache, visible, prefix_count, extra=None):
        """The action expert's velocity for noisy actions `x_t` at `time` over a finished prefix pass: the one routine
        shared by the training loss and sampling.

        kv_cache: stacked prefix (k, v), (layers, b, s, kv_heads, head_dim). visible: bool (b, s), the prefix columns
        every action row may attend. prefix_count: int (b,), the valid prefix tokens (RoPE positions of the suffix
        continue from it). extra: optional (k, v, visible) to append after the prefix columns, k and v (layers, b, e,
        kv_heads, head_dim), visible bool (b, e); they are cast to the cache dtype and carry no RoPE."""
        if extra is not None:
            kv_cache, visible = self._lit_extend_cache(kv_cache, visible, extra)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_len = suffix_tokens.shape[1]
        columns = jnp.broadcast_to(visible[:, None, :], (visible.shape[0], suffix_len, visible.shape[1]))
        full_mask = jnp.concatenate([columns, make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1)
        positions = prefix_count[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def _lit_pose_losses(self, observation: _model.Observation, latents) -> dict:
        goal, valid = self._lit_goal(observation)
        weights = valid.astype(jnp.float32)
        denominator = jnp.maximum(jnp.sum(weights), 1.0)

        def masked_mean(squared):
            return jnp.sum(jnp.where(valid, jnp.mean(squared, axis=-1), 0.0)) / denominator

        state = observation.state[:, jnp.asarray(self.lit_goal_dims)].astype(jnp.float32)
        aux = {"pose_copy_baseline": masked_mean(jnp.square(state - goal))}
        if latents is not None:
            aux["pose_loss"] = masked_mean(jnp.square(self.lit_pose_decoder(latents) - goal))
        return aux

    def _compute_loss_lit(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        with_aux: bool = False,
    ):
        """The LIT training loss: (action loss (b, ah), aux). Two passes instead of the stock joint one: the prefix
        pass (and the aggregator), then the action rows over [prefix cache; latent or goal K/V; own block]. The rng
        stream is the stock 4-way split; nothing here consumes randomness of its own."""
        preprocess_rng, noise_rng, time_rng, mask_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b, ah, _ = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (b, 1)) * 0.999 + 0.001
        time = jnp.broadcast_to(time, (b, ah))

        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        prefix = self._lit_prefix_pass(observation, mask_rng)
        v_t = self._suffix_velocity(
            observation, x_t, time, prefix.kv_cache, prefix.visible, prefix.count, prefix.extra
        )
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return action_loss, (self._lit_pose_losses(observation, prefix.latents) if with_aux else {})

    @at.typecheck
    def embed_prefix_infer(
        self, obs: _model.Observation, memory_tokens
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int, int]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        all_image_tokens = dict()

        for name in obs.images:
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
            for name in obs.images:
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

    def _prefix_attention(self, prefix_mask, prefix_ar_mask, memory_prefix_mask):
        """(attention mask, prefix mask, positions) of the current prefix tokens over [memory; current] columns: the
        mask and positions cover the current rows, the prefix mask the whole [memory; current] width."""
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        if memory_prefix_mask is not None:
            memory_prefix_attn_mask = einops.repeat(memory_prefix_mask, "b p -> b s p", s=prefix_mask.shape[1])
            prefix_attn_mask = jnp.concatenate([memory_prefix_attn_mask, prefix_attn_mask], axis=-1)
            prefix_mask = jnp.concatenate([memory_prefix_mask, prefix_mask], axis=-1)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            positions = positions[..., memory_prefix_mask.shape[-1]:]
        else:
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
        return prefix_attn_mask, prefix_mask, positions

    def lit_prefix(self, observation: _model.Observation, memory: dict):
        """The stage-2 prefix pass for sampling, with the memory carried across calls.

        Takes the observation as `sample_actions` passes it on (preprocessed) and the memory dict. Returns
        (visible, kv_cache, offset, new_memory):
          visible   bool (b, s + e): the columns of kv_cache an action row may attend; the memory and current
                    prefix columns (image and language/state ones hidden per the LIT switches, invalid ones
                    hidden) followed by the e = lit_num_latents latent columns, which are always visible.
          kv_cache  stacked (k, v), each (layers, b, s + e, kv_heads, head_dim): [memory; current prefix; latents],
                    the latent K/V after RoPE in the cache dtype.
          offset    int (b,): the valid prefix tokens (memory included); suffix RoPE positions continue from it.
          new_memory  a copy of `memory` holding this call's prefix, saved before the latents are appended: it
                    has the stock keys only and never a latent, so the memory contract is unchanged.
        The velocity for noisy actions x_t at time t is then
        `_suffix_velocity(observation, x_t, t, kv_cache, visible, offset)`, with t of shape (b, action_horizon)
        as in training, the same routine as the loss. Only lit="stage2" can sample."""
        self._check_lit_sampling()
        memory_tokens, memory_kv_cache, memory_prefix_mask = self.fetch_memory(memory)
        prefix_tokens, prefix_mask, prefix_ar_mask, num_visual_tokens, num_text_tokens = self.embed_prefix_infer(
            observation, memory_tokens
        )
        blocks = next(iter(observation.images.values())).shape[1]
        block_len = prefix_tokens.shape[1] // blocks
        # roles of the current tokens with validity (the aggregator's masks) and of every column without it (the
        # visibility: memory columns carry their own validity in the memory prefix mask)
        roles = jnp.where(prefix_mask, self.prefix_roles(observation, block_len, blocks), _lit.ROLE_PAD)
        layout = self.prefix_roles(observation, block_len, blocks, ignore_validity=True)
        if memory_prefix_mask is not None:
            memory_len = memory_prefix_mask.shape[-1]
            if memory_len % block_len:
                raise ValueError(f"the memory holds {memory_len} columns, not whole {block_len}-token blocks.")
            layout = jnp.concatenate([jnp.tile(layout[:, -block_len:], (1, memory_len // block_len)), layout], axis=1)

        prefix_attn_mask, prefix_mask, positions = self._prefix_attention(
            prefix_mask, prefix_ar_mask, memory_prefix_mask
        )
        start = (blocks - 1) * block_len
        _, kv_cache, layer_inputs = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            kv_cache=memory_kv_cache,
            return_layer_inputs=True,
            layer_input_slice=(start, block_len),
        )
        new_memory = dict(memory)
        self.save_memory(prefix_tokens, prefix_mask, num_visual_tokens, num_text_tokens, kv_cache, new_memory)

        semantic_mask, image_mask = _lit.role_masks(roles[:, start:])
        _, keys, values = self.lit_aggregator(layer_inputs, semantic_mask, image_mask)
        extra = (*self._lit_kv_columns(keys, values), jnp.ones(keys.shape[1:3], bool))
        kv_cache, visible = self._lit_extend_cache(kv_cache, self._lit_visible(prefix_mask, layout), extra)
        return visible, kv_cache, jnp.sum(prefix_mask, axis=-1), new_memory

    def _check_lit_sampling(self):
        if self.lit == "stage1":
            raise ValueError(
                "lit='stage1' is a training-only configuration (no images, goal K/V); it cannot sample. "
                "Sample from a lit='stage2' model."
            )
        if self.lit == "off":
            raise ValueError("lit_prefix is for lit='stage2'; lit='off' samples through the stock path.")

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
        if self.lit == "stage1":
            self._check_lit_sampling()
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if self.lit != "off":
            return self._sample_actions_lit(observation, noise, dt, memory)

        # first fill KV cache with a forward pass of the prefix
 
        memory_tokens, memory_kv_cache, memory_prefix_mask = self.fetch_memory(memory)

        # if jax.process_index() == 0:
        #     jax.debug.print("x1, {}, {}", memory_kv_cache is None, memory_prefix_mask is None)

        prefix_tokens, prefix_mask, prefix_ar_mask, num_visual_tokens, num_text_tokens = self.embed_prefix_infer(observation, memory_tokens)

        # prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        prefix_attn_mask, prefix_mask, positions = self._prefix_attention(
            prefix_mask, prefix_ar_mask, memory_prefix_mask
        )
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions, kv_cache=memory_kv_cache)
        self.save_memory(prefix_tokens, prefix_mask, num_visual_tokens, num_text_tokens, kv_cache, memory)

        def step(carry):
            x_t, time = carry
            time_ = jnp.broadcast_to(time, batch_size)  # (b, )
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time_[:, None]
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            # assert full_attn_mask.shape == (
            #     batch_size,
            #     suffix_tokens.shape[1],
            #     prefix_tokens.shape[1] + suffix_tokens.shape[1],
            # )
            # print(full_attn_mask.shape)
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        with at.disable_typechecking():
            x_0, _ = jax.lax.while_loop(cond, step, (noise, jnp.float32(1.0)))
        return x_0, memory

    def _sample_actions_lit(self, observation, noise, dt, memory):
        visible, kv_cache, offset, new_memory = self.lit_prefix(observation, memory)
        x_0 = self._denoise(observation, noise, dt, visible, kv_cache, offset)
        memory.update(new_memory)
        return x_0, memory

    def _denoise(self, observation, noise, dt, visible, kv_cache, offset):
        """Euler steps of `dt` (< 0) from `noise` at t=1 to 0 over a finished prefix (see `lit_prefix`)."""
        batch_size = noise.shape[0]

        def step(carry):
            x_t, time = carry
            time_ = jnp.broadcast_to(time, (batch_size, self.action_horizon))
            v_t = self._suffix_velocity(observation, x_t, time_, kv_cache, visible, offset)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        with at.disable_typechecking():
            x_0, _ = jax.lax.while_loop(cond, step, (noise, jnp.float32(1.0)))
        return x_0
