# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from lmcache.v1.compute.blend.blender import LMCBlender


class MissingKVConnector:
    def get_kv(self, layer_id):
        raise ValueError(f"missing layer {layer_id}")


class RecordingMetadata:
    def __init__(self):
        self.query_lengths = []
        self.key_lengths = []

    def update_from_top_indices(self, indices):
        self.query_lengths.append(len(indices))

    def truncate_keys(self, key_len):
        self.key_lengths.append(key_len)


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def forward_contiguous(self, q, k, v, output, attn_metadata):
        self.calls.append((len(q), len(k), len(v)))
        output.fill_(len(self.calls))
        return output


class RecordingSharedCacheBackend(RecordingBackend):
    def forward_scattered_shared_cache(
        self, q, k, v, output, cache_seqlens, cache_batch_idx
    ):
        self.calls.append((
            len(q), len(k), len(v), cache_seqlens.tolist(),
            cache_batch_idx.tolist(),
        ))
        output.fill_(7)
        return output


class PrefixAwareConnector:
    def __init__(self, exact):
        self.current_exact_prefix_match = False
        self.exact = exact


class RetrievalOnlyCacheEngine:
    def __init__(self, connector, num_layers):
        self.connector = connector
        self.num_layers = num_layers

    def retrieve_layer(self, tokens, mask, **kwargs):
        yield torch.sum(mask) if mask is not None else len(tokens)
        for layer_id in range(self.num_layers):
            if layer_id == 0:
                self.connector.current_exact_prefix_match = self.connector.exact
            yield None
        yield torch.ones(len(tokens), dtype=torch.bool)


class RecordingLayerwiseModel:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.calls = 0

    def compute_layer(self, *args, **kwargs):
        self.calls += 1
        for _ in range(self.num_layers):
            yield None


class CleanableMetadata(SimpleNamespace):
    def clean(self):
        self.cleaned = True


def test_missing_layer_fails_closed():
    blender = object.__new__(LMCBlender)
    blender.gpu_connector = MissingKVConnector()

    tensor = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="layer 3 was not loaded"):
        blender.process_qkv(
            tensor,
            tensor,
            tensor,
            tensor,
            3,
            None,
            SimpleNamespace(),
        )


def test_mrope_eager_positions_require_exact_token_layout():
    blender = object.__new__(LMCBlender)
    blender.is_mrope = True
    positions = torch.arange(15).reshape(3, 5)

    validated = blender._validate_request_positions(
        positions, 5, torch.device("cpu")
    )

    assert torch.equal(validated, positions)
    with pytest.raises(RuntimeError, match="tokens=4"):
        blender._validate_request_positions(
            positions, 4, torch.device("cpu")
        )
    with pytest.raises(RuntimeError, match=r"exact \[3, num_tokens\]"):
        blender._validate_request_positions(
            torch.arange(5), 5, torch.device("cpu")
        )


def make_mrope_blender(input_ids, grids, placeholders):
    blender = object.__new__(LMCBlender)
    blender._mrope_model_config = {
        "image_token_id": 101,
        "video_token_id": 102,
        "vision_start_token_id": 100,
        "spatial_merge_size": 2,
    }
    blender._active_metadata = SimpleNamespace(
        input_ids=input_ids,
        image_grid_thw=grids,
        mm_positions=placeholders,
    )
    return blender


def test_mrope_grid_fallback_handles_chunk_inside_visual_span():
    image_len = 220
    prefix = [7, 100]
    full_ids = prefix + [101] * image_len + [8, 9]
    placeholder = SimpleNamespace(offset=2, length=image_len)
    blender = make_mrope_blender(
        full_ids, [[1, 22, 40]], [placeholder]
    )

    full = blender._compute_mrope_positions(
        len(full_ids), torch.device("cpu")
    )
    chunk_len = 2 + 24
    chunk = blender._compute_mrope_positions(
        chunk_len, torch.device("cpu")
    )

    assert chunk.shape == (3, chunk_len)
    torch.testing.assert_close(chunk, full[:, :chunk_len])


def test_mrope_grid_fallback_rejects_pruned_visual_span():
    blender = make_mrope_blender(
        [7, 100] + [101] * 24,
        [[1, 22, 40]],
        [SimpleNamespace(offset=2, length=24)],
    )

    with pytest.raises(RuntimeError, match="exact encoder positions"):
        blender._compute_mrope_positions(26, torch.device("cpu"))


def test_causal_blocks_preserve_absolute_positions():
    indices = torch.tensor([10, 11, 12, 26, 27, 42, 43, 44])

    blocks = LMCBlender._causal_blocks(indices)

    assert blocks == [(0, 3, 13), (3, 5, 28), (5, 8, 45)]
    for query_start, query_end, key_end in blocks:
        query_len = query_end - query_start
        assert key_end - query_len == int(indices[query_start])


def test_scattered_attention_runs_each_causal_block():
    blender = object.__new__(LMCBlender)
    blender.blend_mode = "topk"
    blender.cacheblend_batched_partial_attention = False
    indices = torch.tensor([10, 11, 12, 26, 27])
    blender._active_metadata = SimpleNamespace(
        imp_indices=indices,
        causal_blocks=LMCBlender._causal_blocks(indices),
    )
    backend = RecordingBackend()
    metadata = RecordingMetadata()
    q = torch.zeros(5, 1, 2)
    k = torch.zeros(28, 1, 2)
    v = torch.zeros_like(k)
    output = torch.zeros_like(q)

    result = blender.forward_attention(backend, q, k, v, output, metadata)

    assert result is output
    assert backend.calls == [(3, 13, 13), (2, 28, 28)]
    assert metadata.query_lengths == [3, 2]
    assert metadata.key_lengths == [13, 28]
    assert output[:, 0, 0].tolist() == [1, 1, 1, 2, 2]


def test_cacheblend_batched_attention_uses_one_shared_cache_call():
    blender = object.__new__(LMCBlender)
    blender.blend_mode = "topk"
    blender.cacheblend_batched_partial_attention = True
    blender._last_selection_stats = {}
    indices = torch.tensor([10, 11, 26, 42])
    blender._active_metadata = SimpleNamespace(
        imp_indices=indices,
        causal_blocks=LMCBlender._causal_blocks(indices),
        scattered_cache_seqlens=(indices + 1).to(torch.int32),
        scattered_cache_batch_idx=torch.zeros(4, dtype=torch.int32),
        selection_stats={},
    )
    backend = RecordingSharedCacheBackend()
    q = torch.zeros(4, 1, 2)
    k = torch.zeros(43, 1, 2)
    v = torch.zeros_like(k)
    output = torch.zeros_like(q)

    result = blender.forward_attention(
        backend, q, k, v, output, RecordingMetadata()
    )

    assert result is output
    assert backend.calls == [(4, 43, 43, [11, 12, 27, 43], [0, 0, 0, 0])]
    assert output[:, 0, 0].tolist() == [7, 7, 7, 7]
    assert blender._last_selection_stats == {
        "cacheblend_partial_attention_path": "shared_kv_batch",
        "cacheblend_partial_attention_calls_per_layer": 1,
    }


def test_vlcache_batched_attention_uses_one_shared_cache_call():
    blender = object.__new__(LMCBlender)
    blender.blend_mode = "vlcache"
    blender.cacheblend_batched_partial_attention = False
    blender.vlcache_batched_partial_attention = True
    blender._last_selection_stats = {}
    indices = torch.tensor([10, 26, 42, 58])
    blender._active_metadata = SimpleNamespace(
        imp_indices=indices,
        causal_blocks=LMCBlender._causal_blocks(indices),
        scattered_cache_seqlens=(indices + 1).to(torch.int32),
        scattered_cache_batch_idx=torch.zeros(4, dtype=torch.int32),
        selection_stats={},
    )
    backend = RecordingSharedCacheBackend()
    q = torch.zeros(4, 1, 2)
    k = torch.zeros(59, 1, 2)
    v = torch.zeros_like(k)
    output = torch.zeros_like(q)

    result = blender.forward_attention(
        backend, q, k, v, output, RecordingMetadata()
    )

    assert result is output
    assert backend.calls == [(4, 59, 59, [11, 27, 43, 59], [0, 0, 0, 0])]
    assert output[:, 0, 0].tolist() == [7, 7, 7, 7]
    assert blender._last_selection_stats == {
        "vlcache_partial_attention_path": "shared_kv_batch",
        "vlcache_partial_attention_calls_per_layer": 1,
    }


def test_recompute_budget_counts_only_visual_candidate_tokens():
    blender = object.__new__(LMCBlender)
    blender._active_metadata = SimpleNamespace(
        mm_positions=[
            SimpleNamespace(offset=2, length=3),
            SimpleNamespace(offset=8, length=5),
        ],
        selection_stats=None,
    )
    blender._last_selection_stats = {}

    blender._record_selection_stats(
        selected_indices=torch.tensor([0, 2, 4, 8]),
        candidate_indices=torch.arange(10),
        effective_len=10,
        layer_id=1,
        mode="topk",
    )

    assert blender._last_selection_stats == {
        "recompute_selection_mode": "topk",
        "recompute_selection_layer": 1,
        "recompute_candidate_tokens": 10,
        "recompute_selected_tokens": 4,
        "recompute_candidate_visual_tokens": 5,
        "recompute_selected_visual_tokens": 3,
        "recompute_visual_ratio": 0.6,
    }
    assert (blender._active_metadata.selection_stats
            == blender._last_selection_stats)


@pytest.mark.parametrize(
    ("exact_prefix", "fast_path", "provider_calls", "model_calls"),
    [(True, True, 0, 0), (True, False, 1, 1), (False, True, 1, 1)],
)
def test_exact_prefix_skips_embedding_and_decoder_recompute(
    exact_prefix, fast_path, provider_calls, model_calls,
):
    num_layers = 2
    connector = PrefixAwareConnector(exact_prefix)
    layerwise_model = RecordingLayerwiseModel(num_layers)
    blender = object.__new__(LMCBlender)
    blender.gpu_connector = connector
    blender.cache_engine = RetrievalOnlyCacheEngine(connector, num_layers)
    blender.layerwise_model = layerwise_model
    blender.num_layers = num_layers
    blender.blend_mode = "codecsight"
    blender.direct_reuse_retrieve_only = True
    blender.exact_prefix_fast_path = fast_path
    blender.common_metadata = SimpleNamespace(check_layers=[])
    blender._active_metadata = CleanableMetadata(cleaned=False)

    calls = 0

    def provide_embeddings():
        nonlocal calls
        calls += 1
        return torch.zeros(4, 2), None

    outputs = list(blender.blend_layer(
        torch.arange(4),
        torch.ones(4, dtype=torch.bool),
        embedding_provider=provide_embeddings,
        model_input_ids=torch.arange(4),
        req_id="test-request",
    ))

    assert len(outputs) == num_layers + 2
    assert calls == provider_calls
    assert layerwise_model.calls == model_calls
    assert blender._active_metadata.cleaned
