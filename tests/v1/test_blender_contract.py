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


def test_causal_blocks_preserve_absolute_positions():
    indices = torch.tensor([10, 11, 12, 26, 27, 42, 43, 44])

    blocks = LMCBlender._causal_blocks(indices)

    assert blocks == [(0, 3, 13), (3, 5, 28), (5, 8, 45)]
    for query_start, query_end, key_end in blocks:
        query_len = query_end - query_start
        assert key_end - query_len == int(indices[query_start])


def test_scattered_attention_runs_each_causal_block():
    blender = object.__new__(LMCBlender)
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
