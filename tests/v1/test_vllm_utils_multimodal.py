# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from lmcache.integration.vllm.utils import extract_image_grid_thw


class _Data:
    def __init__(self, grid):
        self._grid = grid

    def get_data(self):
        return {"image_grid_thw": self._grid}


def _request(*grids):
    return SimpleNamespace(
        mm_features=[SimpleNamespace(data=_Data(grid)) for grid in grids]
    )


def test_extract_image_grid_thw_preserves_single_flat_triplet():
    assert extract_image_grid_thw(_request([1, 22, 30])) == [[1, 22, 30]]


def test_extract_image_grid_thw_normalizes_tensor_and_flat_multiple():
    request = _request(torch.tensor([[1, 22, 30], [2, 10, 14]]), [3, 4, 5, 6, 7, 8])
    assert extract_image_grid_thw(request) == [
        [1, 22, 30],
        [2, 10, 14],
        [3, 4, 5],
        [6, 7, 8],
    ]


def test_extract_image_grid_thw_rejects_malformed_flat_list():
    assert extract_image_grid_thw(_request([1, 22])) == []
