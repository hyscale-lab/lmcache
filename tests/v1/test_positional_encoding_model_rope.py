import pytest
import torch

from lmcache.v1.compute.positional_encoding import get_fused_rope_from_vllm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA fused op")
def test_fused_relocation_uses_dynamic_ntk_model_cache():
    from vllm.model_executor.layers.rotary_embedding import get_rope

    rope = get_rope(
        head_size=128,
        rotary_dim=128,
        max_position=32768,
        base=1_000_000,
        rope_scaling={"factor": 2.0, "rope_type": "dynamic"},
        dtype=torch.bfloat16,
    ).cuda()
    fused = get_fused_rope_from_vllm(rope)
    assert fused is not None
    fused.rope_cache_to_device(torch.device("cuda"))

    old_positions = torch.arange(5000, 5016, device="cuda")
    new_positions = torch.arange(100, 116, device="cuda")
    raw = torch.randn(16, 8 * 128, device="cuda", dtype=torch.bfloat16)
    query = raw.clone()
    _, old_k = rope(old_positions, query, raw.clone())
    _, expected = rope(new_positions, query, raw.clone())
    actual = fused(old_positions, new_positions, old_k.clone())

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.02
