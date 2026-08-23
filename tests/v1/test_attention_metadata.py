# SPDX-License-Identifier: Apache-2.0

import torch

from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata


def test_truncate_keys_aligns_bottom_right_mask_to_contiguous_query_block():
    """A contiguous block [start, end) must see its true causal prefixes.

    FlashAttention right-aligns the causal triangle when q_len != k_len.  If
    k_len is truncated to ``end``, query i sees ``end - q_len + i``, which is
    exactly its absolute position ``start + i``.
    """
    start, end = 36, 40
    query_len = end - start
    metadata = LMCFlashAttnMetadata(
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=torch.tensor([100], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 100], dtype=torch.int32),
        max_query_len=query_len,
        max_seq_len=100,
    )

    metadata.truncate_keys(end)

    assert metadata.cu_seqlens_k.tolist() == [0, end]
    assert metadata.seq_lens.tolist() == [end]
    assert metadata.max_seq_len == end
    visible_last_key = [end - query_len + i for i in range(query_len)]
    assert visible_last_key == list(range(start, end))
