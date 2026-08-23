# SPDX-License-Identifier: Apache-2.0

import torch
from types import SimpleNamespace

from lmcache.v1.gpu_connector import _mrope_delta_rotate_k, _rotate_half
from lmcache.v1.compute.blend.blender import LMCBlender


def test_mrope_position_builder_accepts_trailing_vision_start_boundary():
    """A 4096-token write-back prefix may end at vision_start itself."""
    vision_start, image_token, video_token = 100, 101, 102
    blender = object.__new__(LMCBlender)
    blender._mrope_model_config = {
        "image_token_id": image_token,
        "video_token_id": video_token,
        "vision_start_token_id": vision_start,
        "spatial_merge_size": 1,
    }
    blender._active_metadata = SimpleNamespace(
        input_ids=[10, vision_start, image_token, image_token, 20,
                   vision_start],
        image_grid_thw=[[1, 1, 2], [1, 1, 2]],
    )

    positions = blender._compute_mrope_positions(6, torch.device("cpu"))

    assert positions.shape == (3, 6)
    # The incomplete second media has supplied no image placeholder yet, so
    # its start marker remains a regular tail position in this prefix.
    torch.testing.assert_close(positions[:, -1], positions[:, -2] + 1)


def test_mrope_delta_rotation_matches_reencode_at_new_3d_positions():
    torch.manual_seed(7)
    num_tokens, num_heads, head_size = 5, 2, 8
    rotary_half = head_size // 2
    sections = [2, 1, 1]

    positions = torch.arange(128, dtype=torch.float64).unsqueeze(1)
    frequencies = torch.tensor(
        [[0.01, 0.03, 0.07, 0.11]], dtype=torch.float64,
    )
    cos_sin_cache = torch.cat([
        torch.cos(positions * frequencies),
        torch.sin(positions * frequencies),
    ], dim=-1)
    old_positions = torch.tensor([
        [20, 21, 22, 23, 24],
        [7, 8, 9, 10, 11],
        [3, 4, 5, 6, 7],
    ])
    new_positions = torch.tensor([
        [4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5],
    ])
    raw_key = torch.randn(
        num_tokens, num_heads * head_size, dtype=torch.float64,
    )

    def rotate(key: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        cached = cos_sin_cache[pos]
        cos_axes = cached[..., :rotary_half]
        sin_axes = cached[..., rotary_half:]
        cos_parts = cos_axes.split(sections, dim=-1)
        sin_parts = sin_axes.split(sections, dim=-1)
        cos = torch.cat([
            part[axis] for axis, part in enumerate(cos_parts)
        ], dim=-1)
        sin = torch.cat([
            part[axis] for axis, part in enumerate(sin_parts)
        ], dim=-1)
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
        shaped = key.view(num_tokens, num_heads, head_size)
        return (
            shaped * cos + _rotate_half(shaped) * sin
        ).reshape(num_tokens, -1)

    cached_key = rotate(raw_key, old_positions)
    repaired_key = _mrope_delta_rotate_k(
        cached_key,
        old_positions,
        new_positions,
        cos_sin_cache,
        head_size,
        sections,
        False,
    )

    torch.testing.assert_close(
        repaired_key, rotate(raw_key, new_positions), atol=1e-10, rtol=1e-10,
    )


def test_interleaved_mrope_delta_matches_qwen3_vl_layout():
    """Qwen3-VL uses interleaved T/H/W frequencies, not chunked sections."""
    torch.manual_seed(11)
    num_tokens, num_heads, head_size = 6, 2, 12
    rotary_half = head_size // 2
    sections = [2, 2, 2]

    positions = torch.arange(128, dtype=torch.float64).unsqueeze(1)
    frequencies = torch.tensor(
        [[0.01, 0.02, 0.04, 0.07, 0.11, 0.17]], dtype=torch.float64,
    )
    cos_sin_cache = torch.cat([
        torch.cos(positions * frequencies),
        torch.sin(positions * frequencies),
    ], dim=-1)
    old_positions = torch.tensor([
        [31, 32, 33, 34, 35, 36],
        [12, 13, 14, 15, 16, 17],
        [5, 6, 7, 8, 9, 10],
    ])
    new_positions = torch.tensor([
        [7, 8, 9, 10, 11, 12],
        [3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6],
    ])
    raw_key = torch.randn(
        num_tokens, num_heads * head_size, dtype=torch.float64,
    )

    def rotate(key: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        cached = cos_sin_cache[pos]
        cos_axes = cached[..., :rotary_half]
        sin_axes = cached[..., rotary_half:]
        # This is vLLM apply_interleaved_rope's exact indexing scheme.
        cos = cos_axes[0].clone()
        sin = sin_axes[0].clone()
        cos[..., 1:sections[1] * 3:3] = cos_axes[
            1, ..., 1:sections[1] * 3:3
        ]
        cos[..., 2:sections[2] * 3:3] = cos_axes[
            2, ..., 2:sections[2] * 3:3
        ]
        sin[..., 1:sections[1] * 3:3] = sin_axes[
            1, ..., 1:sections[1] * 3:3
        ]
        sin[..., 2:sections[2] * 3:3] = sin_axes[
            2, ..., 2:sections[2] * 3:3
        ]
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
        shaped = key.view(num_tokens, num_heads, head_size)
        return (
            shaped * cos + _rotate_half(shaped) * sin
        ).reshape(num_tokens, -1)

    cached_key = rotate(raw_key, old_positions)
    repaired_key = _mrope_delta_rotate_k(
        cached_key,
        old_positions,
        new_positions,
        cos_sin_cache,
        head_size,
        sections,
        True,
    )

    torch.testing.assert_close(
        repaired_key, rotate(raw_key, new_positions), atol=1e-10, rtol=1e-10,
    )
