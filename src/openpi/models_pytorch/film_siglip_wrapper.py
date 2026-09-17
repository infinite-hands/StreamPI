"""FiLM language conditioning for StreamPI's SigLIP vision tower.

Ported from OpenVLA-OFT's FiLM ViT wrapper (prismatic/models/film_vit_wrapper.py),
retargeted from timm.models.vision_transformer.VisionTransformer blocks onto this
repo's HF-style SiglipEncoderLayer / SiglipEncoder (transformers_replace/models/siglip).

Conditions each frame's visual features on that same frame's re-anchored instruction
embedding, per StreamPI's instruction-anchored temporal unit design (each (image,
instruction) pair is embedded independently via embed_prefix before the cross-frame
causal KV-cache attention is applied) -- so this sits strictly *within* one frame's
embed_prefix call and does not touch cross-frame attention, the KV cache, or the
action expert.
"""

import torch
import torch.nn as nn

# NOT from openpi.models_pytorch.transformers_replace.models.siglip.modeling_siglip: that tree is a
# patch staged to be `cp -r`'d over the installed transformers package (see PI0Pytorch.__init__'s own
# check), and its relative imports (`from ...activations import ACT2FN`) only resolve once it is
# physically sitting inside transformers/ -- importing it from its source location here breaks at
# that relative import. transformers.models.siglip.modeling_siglip is the same file post-patch.
from transformers.models.siglip.modeling_siglip import SiglipEncoder, SiglipEncoderLayer


class FiLMedSiglipEncoderLayer(nn.Module):
    """Wraps a SiglipEncoderLayer to modulate its output via FiLM:

        x = (1 + gamma) * x + beta

    gamma/beta are learned projections of the average language embedding for the
    current frame's instruction. Matches OFT's placement (modulate after the
    attention sub-block, before the MLP sub-block) and its (1 + gamma) parameterization
    so gamma/beta near zero at init leaves the pretrained SigLIP weights undisturbed.
    """

    def __init__(self, layer: SiglipEncoderLayer, vision_dim: int, lang_dim: int):
        super().__init__()
        self.layer = layer
        self.scale = nn.Linear(lang_dim, vision_dim)
        self.shift = nn.Linear(lang_dim, vision_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, hidden_states, attention_mask, average_language_embedding, output_attentions=False):
        gamma = self.scale(average_language_embedding)  # (batch, vision_dim)
        beta = self.shift(average_language_embedding)  # (batch, vision_dim)

        residual = hidden_states
        hidden_states = self.layer.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * (1 + gamma[:, None, :]) + beta[:, None, :]

        residual = hidden_states
        hidden_states = self.layer.layer_norm2(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class FiLMedSiglipEncoder(nn.Module):
    """Wraps a SiglipEncoder, threading average_language_embedding to every wrapped layer."""

    def __init__(self, encoder: SiglipEncoder, lang_dim: int):
        super().__init__()
        vision_dim = encoder.config.hidden_size
        self.config = encoder.config
        self.gradient_checkpointing = encoder.gradient_checkpointing
        self.layers = nn.ModuleList(
            [FiLMedSiglipEncoderLayer(layer, vision_dim=vision_dim, lang_dim=lang_dim) for layer in encoder.layers]
        )

    def forward(
        self,
        inputs_embeds,
        average_language_embedding,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                average_language_embedding,
                output_attentions=bool(output_attentions),
            )
            hidden_states = layer_outputs[0]
        return hidden_states


def apply_film_to_siglip(vision_model, lang_dim: int) -> None:
    """In-place: replaces vision_model.encoder with a FiLM-conditioned encoder.

    `vision_model` is a SiglipVisionTransformer (accessible as
    `paligemma_with_expert.paligemma.vision_tower.vision_model`).
    """
    vision_model.encoder = FiLMedSiglipEncoder(vision_model.encoder, lang_dim=lang_dim)
