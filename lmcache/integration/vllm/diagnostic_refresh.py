"""Opt-in dense InternVL refresh controls; never enabled by paper defaults."""
from typing import Any, Sequence
import os


POLICIES = {
    "rope_fix_reuse",
    "multi_anchor",
    "shifted_anchor",
    "center_contiguous",
    "random",
    "full_refresh",
}


MATCHED_BUDGET_POLICIES = {
    "multi_anchor",
    "shifted_anchor",
    "center_contiguous",
    "random",
}


def visual_positions(placeholder):
    start, length = int(placeholder.offset), int(placeholder.length)
    mask = getattr(placeholder, "is_embed", None)
    offsets = range(length) if mask is None else mask.nonzero().flatten().tolist()
    positions = [start + i for i in offsets]
    if start < 0 or len(positions) != 256:
        raise ValueError(f"diagnostic requires 256 visual embeddings: span={length}, visual={len(positions)}")
    return positions


def apply_control(copied: bytearray, placeholders: Sequence[Any],
                  decisions: Sequence[Any], policy: str, seed: int):
    import torch
    if policy not in POLICIES:
        raise ValueError(f"unknown diagnostic refresh policy: {policy}")
    if len(placeholders) != len(decisions):
        raise ValueError("diagnostic frame decisions must match placeholders")
    frames, anchors = [], []
    for placeholder, decision in zip(placeholders, decisions, strict=True):
        start, length = int(placeholder.offset), int(placeholder.length)
        end = start + length
        positions = visual_positions(placeholder)
        if end <= len(copied) and all(copied[start:end]):
            frames.append(positions)
            if decision.is_anchor:
                anchors.extend(positions)
    if len(frames) != 64 or len(anchors) != 1024:
        raise ValueError(f"diagnostic warm overlap mismatch: frames={len(frames)}, anchors={len(anchors)}")
    candidates = [p for frame in frames for p in frame]
    if policy == "rope_fix_reuse":
        selected = []
    elif policy == "multi_anchor":
        selected = anchors
    elif policy == "shifted_anchor":
        # Preserve the four-frame periodicity of GOP anchors while shifting
        # every selection by half a GOP.  This isolates codec alignment from
        # the benefit of regular, frame-contiguous refreshes.
        selected = [position for frame in frames[8::16] for position in frame]
    elif policy == "center_contiguous":
        # Refresh one centered four-frame span.  It has the same visual-token
        # budget as the four distributed anchor frames, but a different
        # temporal layout and memory-access pattern.
        selected = [position for frame in frames[30:34] for position in frame]
    elif policy == "full_refresh":
        selected = candidates
    else:
        order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed))[:len(anchors)]
        selected = sorted(candidates[i] for i in order.tolist())
    expected = (
        0
        if policy == "rope_fix_reuse"
        else len(candidates)
        if policy == "full_refresh"
        else len(anchors)
    )
    if len(selected) != expected or len(set(selected)) != expected:
        raise ValueError(
            f"diagnostic refresh budget mismatch: policy={policy}, "
            f"selected={len(selected)}, expected={expected}"
        )
    # Recompute all nonvisual tokens outside the visual refresh budget.
    copied[:] = b"\x00" * len(copied)
    for position in candidates:
        copied[position] = 1
    for position in selected:
        copied[position] = 0
    result = dict(diagnostic_refresh_policy=policy, diagnostic_refresh_seed=seed,
                diagnostic_overlap_visual_tokens=len(candidates),
                diagnostic_refresh_visual_tokens=len(selected),
                diagnostic_nonvisual_recompute=True)
    if os.environ.get("COSTREAM_DIAGNOSTIC_AUDIT", "1") == "1":
        result.update(diagnostic_refresh_visual_positions=selected,
                      diagnostic_overlap_frame_positions=frames)
    return result
