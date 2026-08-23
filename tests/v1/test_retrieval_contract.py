# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lmcache.v1.retrieval_contract import (
    expected_retrieval_count,
    normalize_retrieved_count,
    validate_retrieval_count,
)


def test_retrieval_contract_counts_boolean_masks_and_scalar_results():
    mask = torch.tensor([False, False, True, True, True])
    assert expected_retrieval_count(mask, len(mask)) == 3
    assert normalize_retrieved_count(torch.tensor(3)) == 3
    assert validate_retrieval_count(
        expected=3,
        actual=torch.tensor(3),
        request_id="req-ok",
        path="test",
    ) == 3


def test_retrieval_contract_counts_returned_masks():
    actual_mask = torch.tensor([False, True, True, False])
    assert normalize_retrieved_count(actual_mask) == 2


def test_retrieval_contract_fails_closed_with_request_context():
    with pytest.raises(RuntimeError, match=r"request=req-bad, path=plain"):
        validate_retrieval_count(
            expected=4,
            actual=torch.tensor(3),
            request_id="req-bad",
            path="plain",
        )


def test_retrieval_contract_treats_missing_result_as_zero():
    with pytest.raises(RuntimeError, match="retrieval loaded 0"):
        validate_retrieval_count(
            expected=1,
            actual=None,
            request_id=None,
            path="missing",
        )
