# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl


def test_cached_text_prefix_is_an_explicit_no_visual_noop():
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    result = connector._reconstruct_inputs_embeds(
        token_ids=list(range(100)),
        mm_hashes=["frame-0"],
        mm_positions=[SimpleNamespace(offset=1000, length=256)],
        num_tokens=43,
    )

    assert result.no_visual_prefix
    assert result.status == "no_visual_prefix"
    assert result.inputs_embeds is None
    assert "43 tokens" in result.detail


def test_text_only_request_is_an_explicit_no_visual_noop():
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    result = connector._reconstruct_inputs_embeds(
        token_ids=list(range(20)),
        mm_hashes=None,
        mm_positions=None,
        num_tokens=20,
    )

    assert result.no_visual_prefix
