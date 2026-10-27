import openpi.models.gemma as gemma


def test_expert_rank_128_variant_keeps_the_lora_scale_and_the_shape():
    small = gemma.get_config("gemma_300m_lora")
    large = gemma.get_config("gemma_300m_lora_r128")
    for part in ("attn", "ffn"):
        assert large.lora_configs[part].rank == 128
        assert large.lora_configs[part].scaling_value == small.lora_configs[part].scaling_value == 1.0
    assert (large.width, large.depth, large.mlp_dim) == (small.width, small.depth, small.mlp_dim)
