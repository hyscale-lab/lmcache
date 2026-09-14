# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase, SegmentTokenDatabase

# Local
from .utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens


def hf_credentials_available() -> bool:
    token_env = os.getenv("HF_TOKEN")
    hf_home = os.getenv("HF_HOME")
    default_token_file = os.path.expanduser("~/.cache/huggingface/token")
    token_file = os.path.join(hf_home, "token") if hf_home else ""
    return bool(
        token_env or os.path.exists(default_token_file) or os.path.exists(token_file)
    )


@pytest.mark.parametrize("chunk_length", [16, 64, 256])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_chunked_token_database(chunk_length, save_unfull_chunk):
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_length, backend="cpu", save_unfull_chunk=save_unfull_chunk
    )
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [i * chunk_length for i in range(0, test_length // chunk_length)]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens=tokens))
    end = (
        test_length if save_unfull_chunk else (test_length - test_length % chunk_length)
    )
    for i in range(0, end, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        if save_unfull_chunk:
            assert ed == min(i + chunk_length, test_length)
        else:
            assert ed == i + chunk_length

    for i in range(0, test_length // chunk_length):
        mask[: num_falses[i]] = False
        new_results = list(db.process_tokens(tokens=tokens, mask=mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


@pytest.mark.parametrize("prefix_length", [0, 16, 64, 256])
@pytest.mark.parametrize("chunk_lengths", [[256, 512, 256], [1024, 512, 256]])
@pytest.mark.skipif(
    not hf_credentials_available(), reason="No Hugging Face credentials found"
)
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    frame_tokens = []
    for chunk_length in chunk_lengths:
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)
        frame_tokens.append(token_chunk)

    # Separator-inclusive chunks must tile the reported cache-hit prefix.
    chunks = [
        torch.cat([sys_tokens, sep_tokens]),
        *(torch.cat([chunk, sep_tokens]) for chunk in frame_tokens),
        query_tokens,
    ]
    tokens = torch.cat(chunks)
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    expected = []
    start = 0
    for chunk in chunks:
        end = start + len(chunk)
        if end > prefix_length:
            expected.append((
                start,
                end,
                hash((None, tuple(chunk.cpu().tolist()), None)),
            ))
        start = end

    original_results = list(db.process_tokens(tokens=tokens, mask=mask))
    assert len(original_results) == len(expected)
    for (st, ed, key), (expected_st, expected_ed, expected_hash) in zip(
        original_results, expected, strict=True,
    ):
        assert st == expected_st
        assert ed == expected_ed
        assert key.chunk_hash == expected_hash


def test_segment_token_database_ranges_are_contiguous_without_tokenizer():
    """Regression: separator holes must never be reported as cached prefix."""
    db = SegmentTokenDatabase.__new__(SegmentTokenDatabase)
    db.sep_tokens = torch.tensor([90, 91], dtype=torch.long)
    db.sep_len = 2
    db.hash_func = hash
    db.metadata = None

    tokens = torch.tensor([
        1, 2, 90, 91,       # system + separator
        10, 11, 90, 91,     # frame 0 + separator
        20, 21, 90, 91,     # frame 1 + separator
        30, 31,              # query (not stored)
    ])

    results = list(db.process_tokens(
        tokens=tokens, make_key=False, skip_last_segment=True,
    ))
    assert [(start, end) for start, end, _ in results] == [
        (0, 4), (4, 8), (8, 12),
    ]
    assert all(
        left_end == right_start
        for (_, left_end, _), (right_start, _, _) in zip(
            results, results[1:], strict=False,
        )
    )


def test_segment_mask_keeps_the_chunk_crossing_the_prefix_boundary():
    db = SegmentTokenDatabase.__new__(SegmentTokenDatabase)
    db.sep_tokens = torch.tensor([90, 91], dtype=torch.long)
    db.sep_len = 2
    db.hash_func = hash
    db.metadata = None
    tokens = torch.tensor([
        1, 2, 90, 91,
        10, 11, 90, 91,
        20, 21, 90, 91,
    ])
    mask = torch.ones(len(tokens), dtype=torch.bool)
    mask[:6] = False

    results = list(db.process_tokens(
        tokens=tokens, mask=mask, make_key=False,
    ))

    assert [(start, end) for start, end, _ in results] == [
        (4, 8), (8, 12),
    ]
