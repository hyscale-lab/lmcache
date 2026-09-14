import pytest
import torch

from lmcache.v1.cache_engine import (
    LayerwiseRetrievalBatchInfo,
    LayerwiseRetrievalRequest,
)
from lmcache.v1.gpu_connector import VLLMBufferLayerwiseGPUConnector
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat


def test_retrieval_request_validates_parallel_inputs():
    with pytest.raises(ValueError, match="slot_mapping"):
        LayerwiseRetrievalRequest(
            request_id="request-0",
            tokens=[1, 2],
            mask=torch.ones(2, dtype=torch.bool),
            slot_mapping=torch.arange(1),
        )

    with pytest.raises(ValueError, match=r"\[3, T\]"):
        LayerwiseRetrievalRequest(
            request_id="request-0",
            tokens=[1, 2],
            mask=None,
            slot_mapping=torch.arange(2),
            cache_positions=torch.zeros((2, 2), dtype=torch.long),
        )

    with pytest.raises(ValueError, match="protected prefix"):
        LayerwiseRetrievalRequest(
            request_id="request-0",
            tokens=[1, 2],
            mask=None,
            slot_mapping=torch.arange(2),
            protected_prefix_tokens=3,
        )


def test_retrieval_batch_info_exposes_exact_masks_and_counts():
    info = LayerwiseRetrievalBatchInfo((
        torch.tensor([True, False, True]),
        torch.tensor([False, True]),
    ))

    assert info.counts == (2, 1)


def test_coalesced_layout_preserves_request_and_chunk_order():
    layout = VLLMBufferLayerwiseGPUConnector._coalesced_layout(
        [[8, 16], [0]],
        [[12, 20], [6]],
    )

    assert layout == (
        18,
        [(0, 8, 12), (12, 0, 6)],
        [(0, 4), (8, 12), (12, 18)],
        [(4, 8)],
    )


@pytest.mark.parametrize(
    "starts,ends,error",
    [
        ([], [], "parallel"),
        ([[0]], [[], []], "parallel"),
        ([[0]], [[0]], "non-empty"),
        ([[4, 2]], [[6, 4]], "ordered"),
        ([[0, 2]], [[3, 4]], "overlap"),
    ],
)
def test_coalesced_layout_rejects_invalid_ranges(starts, ends, error):
    with pytest.raises(ValueError, match=error):
        VLLMBufferLayerwiseGPUConnector._coalesced_layout(starts, ends)


def test_unprotected_ranges_preserve_multiple_apc_prefixes():
    assert VLLMBufferLayerwiseGPUConnector._unprotected_ranges(
        18, [(0, 3), (12, 14)],
    ) == [(3, 12), (14, 18)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_coalesced_gpu_transfer_matches_individual_requests():
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    layers, blocks, block_size = 2, 8, 16
    heads, head_size = 2, 8
    hidden_size = heads * head_size
    cache_shape = (2, blocks, block_size, heads, head_size)
    batch_cache = [
        torch.full(cache_shape, -3, dtype=dtype, device=device)
        for _ in range(layers)
    ]
    individual_cache = [tensor.clone() for tensor in batch_cache]
    request_starts = [[0, 6], [1]]
    request_ends = [[4, 8], [5]]
    slot_mappings = [
        torch.arange(0, 8, dtype=torch.long, device=device),
        torch.arange(16, 21, dtype=torch.long, device=device),
    ]

    allocator = GPUMemoryAllocator(1 << 20, device=device)
    memory_objects = []
    for _ in range(layers):
        layer_objects = []
        for length in (4, 2, 4):
            memory_obj = allocator.allocate(
                torch.Size([2, length, hidden_size]),
                dtype,
                MemoryFormat.KV_2TD,
            )
            assert memory_obj is not None and memory_obj.tensor is not None
            memory_obj.tensor.copy_(torch.randn_like(memory_obj.tensor))
            layer_objects.append(memory_obj)
        memory_objects.append(layer_objects)

    batch_connector = VLLMBufferLayerwiseGPUConnector(
        hidden_size, layers, use_gpu=True, dtype=dtype, device=device,
    )
    batch_connector.cache_positions = False
    consumer = batch_connector.batched_to_gpu_multi(
        request_starts,
        request_ends,
        slot_mappings,
        [None, None],
        [[b"a", b"b"], [b"c"]],
        protected_prefix_tokens=[3, 3],
        kvcaches=batch_cache,
    )
    next(consumer)
    for layer_objects in memory_objects:
        consumer.send(layer_objects)
    next(consumer)

    individual_connector = VLLMBufferLayerwiseGPUConnector(
        hidden_size, layers, use_gpu=True, dtype=dtype, device=device,
    )
    individual_connector.cache_positions = False
    for request_index, (starts, ends) in enumerate(
        zip(request_starts, request_ends, strict=True)
    ):
        request_objects = [
            objects[:2] if request_index == 0 else objects[2:]
            for objects in memory_objects
        ]
        consumer = individual_connector.batched_to_gpu(
            starts,
            ends,
            kvcaches=individual_cache,
            slot_mapping=slot_mappings[request_index],
            protected_prefix_tokens=3,
        )
        next(consumer)
        for layer_objects in request_objects:
            consumer.send(layer_objects)
        next(consumer)

    torch.cuda.synchronize()
    for actual, expected in zip(batch_cache, individual_cache, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        flat = actual.view(2, -1, heads, head_size)
        assert torch.all(flat[:, [0, 1, 2, 17, 18]] == -3)

    for layer_objects in memory_objects:
        for memory_obj in layer_objects:
            memory_obj.ref_count_down()
    assert allocator.memcheck()
    assert batch_connector.gpu_buffer_allocator.memcheck()
    assert individual_connector.gpu_buffer_allocator.memcheck()
