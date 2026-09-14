# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.compute.models.utils import VLLMModelTracker
from vllm.multimodal.evs import recompute_mrope_positions


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


def test_qwen_cache_positions_follow_pruned_encoder_metadata(monkeypatch):
    vision_start = 100
    image_token = 101
    video_token = 102

    class FakeQwen:
        visual_dim = 4
        multiscale_dim = 0

        def recompute_mrope_positions(
            self, input_ids, multimodal_embeddings, positions,
            num_computed_tokens,
        ):
            mm_positions = [
                embed[:, -4:].permute(1, 0).long()
                for embed in multimodal_embeddings
            ]
            updated, delta = recompute_mrope_positions(
                torch.as_tensor(input_ids, device=positions.device),
                mm_positions,
                positions,
                num_computed_tokens,
                vision_start,
                image_token,
                video_token,
            )
            return multimodal_embeddings, updated, delta

    encoder_output = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 1, 2],
    ], dtype=torch.float32)
    monkeypatch.setitem(
        VLLMModelTracker._vllm_models, "vllm-instance", FakeQwen(),
    )
    monkeypatch.setitem(
        VLLMModelTracker._encoder_caches,
        "vllm-instance",
        {"frame-0": encoder_output},
    )

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.blender = SimpleNamespace(is_mrope=True)
    connector._ensure_blender_initialized = lambda: None
    request = SimpleNamespace(
        model_token_ids=[9, vision_start, image_token, image_token, 8],
        mm_hashes=["frame-0"],
        mm_positions=[SimpleNamespace(offset=2, length=2)],
        image_grid_thw=[[1, 2, 2]],
    )

    positions = connector._compute_request_cache_positions(
        request, 5, torch.device("cpu"),
    )

    expected = torch.tensor([
        [0, 1, 2, 2, 4],
        [0, 1, 2, 2, 4],
        [0, 1, 2, 3, 4],
    ])
    torch.testing.assert_close(positions, expected)


def test_qwen_cache_positions_survive_full_encoder_cache_release(monkeypatch):
    vision_start = 100
    image_token = 101
    video_token = 102

    class FakeQwen:
        visual_dim = 4
        multiscale_dim = 0

        def recompute_mrope_positions(self, *_args):
            raise AssertionError("full encoder path should be unavailable")

    monkeypatch.setitem(
        VLLMModelTracker._vllm_models, "vllm-instance", FakeQwen(),
    )
    monkeypatch.setitem(
        VLLMModelTracker._encoder_caches, "vllm-instance", {},
    )
    monkeypatch.setitem(
        VLLMModelTracker._encoder_position_caches,
        "vllm-instance",
        {"frame-0": torch.tensor([[0, 0, 0, 2], [0, 0, 1, 2]])},
    )

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.blender = SimpleNamespace(
        is_mrope=True,
        _mrope_model_config={
            "vision_start_token_id": vision_start,
            "image_token_id": image_token,
            "video_token_id": video_token,
        },
    )
    connector._ensure_blender_initialized = lambda: None
    request = SimpleNamespace(
        model_token_ids=[9, vision_start, image_token, image_token, 8],
        mm_hashes=["frame-0"],
        mm_positions=[SimpleNamespace(offset=2, length=2)],
        image_grid_thw=[[1, 2, 2]],
    )

    positions = connector._compute_request_cache_positions(
        request, 5, torch.device("cpu"),
    )
    expected = torch.tensor([
        [0, 1, 2, 2, 4],
        [0, 1, 2, 2, 4],
        [0, 1, 2, 3, 4],
    ])
    torch.testing.assert_close(positions, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reconstruction_uses_model_refresh_protocol(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def prepare_kv_refresh_inputs(
            self, input_ids, multimodal_embeddings,
        ):
            self.calls.append((input_ids.clone(), multimodal_embeddings))
            return torch.ones(5, 3, device=input_ids.device), None

    model = FakeModel()
    encoder_output = torch.arange(
        24, dtype=torch.float32, device="cuda"
    ).reshape(3, 8)
    monkeypatch.setitem(
        VLLMModelTracker._vllm_models, "vllm-instance", model,
    )
    monkeypatch.setitem(
        VLLMModelTracker._encoder_caches,
        "vllm-instance",
        {"frame-0": encoder_output},
    )

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    result = connector._reconstruct_inputs_embeds(
        token_ids=[1, 2, 3, 4, 5],
        mm_hashes=["frame-0"],
        mm_positions=[SimpleNamespace(
            offset=1,
            length=3,
            is_embed=torch.tensor([True, False, True]),
        )],
        num_tokens=5,
    )

    assert result.ready
    assert len(model.calls) == 1
    assert model.calls[0][1][0].shape == (2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reconstruction_transiently_reencodes_evicted_visual_items(
    monkeypatch,
):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def prepare_kv_refresh_inputs(
            self, input_ids, multimodal_embeddings,
        ):
            self.calls.append((input_ids.clone(), multimodal_embeddings))
            return torch.ones(5, 3, device=input_ids.device), None

    model = FakeModel()
    encoder_output = torch.arange(
        24, dtype=torch.float32, device="cuda"
    ).reshape(3, 8)
    recompute_calls = []

    def recompute(request_id, mm_hashes, mm_positions, num_tokens):
        recompute_calls.append(
            (request_id, list(mm_hashes), list(mm_positions), num_tokens)
        )
        return {"frame-0": encoder_output}

    monkeypatch.setitem(
        VLLMModelTracker._vllm_models, "vllm-instance", model,
    )
    monkeypatch.setitem(
        VLLMModelTracker._encoder_caches, "vllm-instance", {},
    )
    monkeypatch.setitem(
        VLLMModelTracker._encoder_recompute_callbacks,
        "vllm-instance",
        recompute,
    )

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    position = SimpleNamespace(
        offset=1,
        length=3,
        is_embed=torch.tensor([True, False, True]),
    )
    result = connector._reconstruct_inputs_embeds(
        token_ids=[1, 2, 3, 4, 5],
        mm_hashes=["frame-0"],
        mm_positions=[position],
        num_tokens=5,
        request_id="request-0",
    )

    assert result.ready
    assert recompute_calls == [
        ("request-0", ["frame-0"], [position], 5)
    ]
    assert len(model.calls) == 1
    assert model.calls[0][1][0].shape == (2, 8)
    # The worker callback owns the transient tensor; reconstruction must not
    # mutate vLLM's scheduler-accounted encoder cache.
    assert VLLMModelTracker._encoder_caches["vllm-instance"] == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reconstruction_reports_transient_reencode_failure(monkeypatch):
    monkeypatch.setitem(
        VLLMModelTracker._vllm_models, "vllm-instance", object(),
    )
    monkeypatch.setitem(
        VLLMModelTracker._encoder_caches, "vllm-instance", {},
    )

    def fail(*_args):
        raise RuntimeError("vision encoder failed")

    monkeypatch.setitem(
        VLLMModelTracker._encoder_recompute_callbacks,
        "vllm-instance",
        fail,
    )
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    result = connector._reconstruct_inputs_embeds(
        token_ids=[1, 2],
        mm_hashes=["frame-0"],
        mm_positions=[SimpleNamespace(offset=0, length=1, is_embed=None)],
        num_tokens=2,
        request_id="request-0",
    )

    assert result.status == "encoder_recompute_failure"
    assert "vision encoder failed" in result.detail
