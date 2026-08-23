# SPDX-License-Identifier: Apache-2.0
"""Shared validation for scheduler lookup promises."""

from typing import Optional, Union

import torch


RetrievedCount = Optional[Union[int, torch.Tensor]]


def expected_retrieval_count(
    mask: Optional[torch.Tensor],
    num_tokens: int,
) -> int:
    if mask is None:
        return num_tokens
    return int(mask.to(dtype=torch.int64).sum().item())


def normalize_retrieved_count(value: RetrievedCount) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        return int(value.to(dtype=torch.int64).sum().item())
    return int(value)


def validate_retrieval_count(
    *,
    expected: int,
    actual: RetrievedCount,
    request_id: Optional[str],
    path: str,
) -> int:
    actual_count = normalize_retrieved_count(actual)
    if actual_count != expected:
        req = request_id or "<unknown>"
        raise RuntimeError(
            "LMCache lookup/retrieve contract violated: "
            f"request={req}, path={path}, scheduler promised {expected} "
            f"tokens but retrieval loaded {actual_count}. Refusing to run "
            "with incomplete KV."
        )
    return actual_count
