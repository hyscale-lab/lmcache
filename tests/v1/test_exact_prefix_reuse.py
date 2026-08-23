# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from lmcache.v1.cache_engine import _prefix_context_hashes
from lmcache.v1.gpu_connector import _context_hashes_match
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata


def _memory_obj(context_hash):
    return SimpleNamespace(
        metadata=SimpleNamespace(cached_context_hash=context_hash),
    )


def test_appending_tokens_preserves_existing_prefix_hashes():
    old = _prefix_context_hashes([10, 11, 12, 13], [2, 4])
    extended = _prefix_context_hashes([10, 11, 12, 13, 14, 15], [2, 4, 6])

    assert extended[:2] == old


def test_relocated_segment_has_a_different_context_hash():
    original = _prefix_context_hashes([10, 11, 20, 21], [2, 4])
    relocated = _prefix_context_hashes([10, 11, 30, 31, 20, 21], [2, 4, 6])

    assert relocated[-1] != original[-1]


def test_context_match_requires_every_chunk_and_hash():
    expected = _prefix_context_hashes(torch.arange(6), [2, 4, 6])
    memory_objs = [_memory_obj(value) for value in expected]

    assert _context_hashes_match(memory_objs, expected)
    assert not _context_hashes_match(memory_objs[:-1], expected)
    assert not _context_hashes_match(
        [memory_objs[0], _memory_obj(b"wrong"), memory_objs[2]], expected,
    )
    assert not _context_hashes_match(
        [memory_objs[0], None, memory_objs[2]], expected,
    )


def test_context_hash_survives_memory_metadata_serialization():
    metadata = MemoryObjMetadata(
        shape=torch.Size([2, 4, 8]),
        dtype=torch.float16,
        address=123,
        phy_size=128,
        ref_count=1,
        fmt=MemoryFormat.KV_2TD,
        cached_context_hash=b"prefix-context",
    )

    restored = MemoryObjMetadata.from_dict(metadata.to_dict())

    assert restored.cached_context_hash == metadata.cached_context_hash
