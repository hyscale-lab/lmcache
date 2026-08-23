# SPDX-License-Identifier: Apache-2.0

import torch

from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_ids,
    hex_hash_to_token_sentinel,
)


class _Placeholder:
    def __init__(self, offset: int, length: int):
        self.offset = offset
        self.length = length


def test_multimodal_hash_sentinels_keep_more_than_low_16_bits():
    # These differ outside the old low-16-bit truncation and used to collide.
    left = "1234567890abcdef1234567890abcdef"
    right = "ffff567890abcdef1234567890abcdef"

    left_id = hex_hash_to_token_sentinel(left)
    right_id = hex_hash_to_token_sentinel(right)

    assert left_id < 0
    assert right_id < 0
    assert left_id != right_id

    token_ids = torch.arange(8, dtype=torch.long)
    apply_mm_hashes_to_token_ids(
        token_ids,
        [left, right],
        [_Placeholder(1, 2), _Placeholder(5, 2)],
    )
    assert token_ids.tolist() == [
        0, left_id, left_id, 3, 4, right_id, right_id, 7,
    ]
