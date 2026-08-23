# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)


def _evenly_spaced(start: int, end: int, count: int) -> list[int]:
    if end <= start or count <= 0:
        return []
    if end - start <= count:
        return list(range(start, end))
    return sorted({
        start + round(i * (end - start - 1) / (count - 1))
        for i in range(count)
    })


class KVDiagnostic:
    """Opt-in snapshots of sampled tokens from vLLM's paged KV cache."""

    def __init__(self, num_layers: int, rank: int):
        root = os.environ.get("LMCACHE_KV_DIAG_DIR", "").strip()
        self.enabled = bool(root)
        self.root = Path(root) / f"rank_{rank:02d}" if root else None
        self.num_layers = num_layers
        self.rank = rank
        self.layers = self._parse_layers(
            os.environ.get("LMCACHE_KV_DIAG_LAYERS", "0,1,mid,last")
        )
        self.per_frame = max(
            1, int(os.environ.get("LMCACHE_KV_DIAG_TOKENS_PER_FRAME", "8"))
        )
        self.edge_tokens = max(
            0, int(os.environ.get("LMCACHE_KV_DIAG_EDGE_TOKENS", "8"))
        )
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            logger.info(
                "KV diagnostics enabled: dir=%s layers=%s tokens_per_frame=%d",
                self.root,
                sorted(self.layers),
                self.per_frame,
            )

    def _parse_layers(self, raw: str) -> set[int]:
        aliases = {"first": 0, "mid": self.num_layers // 2,
                   "last": self.num_layers - 1}
        layers: set[int] = set()
        for item in raw.split(","):
            value = item.strip().lower()
            if not value:
                continue
            layer = aliases.get(value)
            if layer is None:
                try:
                    layer = int(value)
                except ValueError:
                    logger.warning("Ignoring invalid diagnostic layer %r", item)
                    continue
            if layer < 0:
                layer += self.num_layers
            if 0 <= layer < self.num_layers:
                layers.add(layer)
        return layers

    def wants_layer(self, layer_id: int) -> bool:
        return self.enabled and layer_id in self.layers

    def capture(
        self,
        phase: str,
        layer_id: int,
        layer_name: str,
        kv_layer: torch.Tensor,
        request: Any,
        token_limit: Optional[int] = None,
    ) -> None:
        if not self.wants_layer(layer_id) or self.root is None:
            return
        limit = min(len(request.token_ids), token_limit or len(request.token_ids))
        positions, frame_indices, kinds = self._sample_positions(request, limit)
        if not positions:
            return

        slots = request.slot_mapping[positions].to(
            device=kv_layer.device, dtype=torch.long
        )
        values = self._gather(kv_layer, slots).detach().cpu()
        request_id = str(request.req_id)
        digest = hashlib.sha1(
            ",".join(map(str, request.model_token_ids[:limit])).encode()
        ).hexdigest()
        payload = {
            "schema_version": 1,
            "phase": phase,
            "rank": self.rank,
            "request_id": request_id,
            "layer_id": layer_id,
            "layer_name": layer_name,
            "token_count": len(request.token_ids),
            "token_limit": limit,
            "token_sha1": digest,
            "sample_positions": torch.tensor(positions, dtype=torch.int32),
            "sample_frame_indices": torch.tensor(frame_indices, dtype=torch.int16),
            "sample_kinds": kinds,
            "mm_hashes": list(request.mm_hashes or []),
            "kv": values,
        }
        safe_id = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in request_id
        )[-96:]
        name = f"{safe_id}.n{limit}.l{layer_id:03d}.{phase}.pt"
        target = self.root / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)

    def _sample_positions(
        self, request: Any, limit: int
    ) -> tuple[list[int], list[int], list[str]]:
        samples: dict[int, tuple[int, str]] = {}
        mm_positions = list(request.mm_positions or [])
        first_mm = min(
            (int(getattr(item, "offset", limit)) for item in mm_positions),
            default=limit,
        )
        for position in _evenly_spaced(0, min(first_mm, limit), self.edge_tokens):
            samples[position] = (-1, "prefix")

        last_mm = 0
        for frame_index, item in enumerate(mm_positions):
            start = max(0, int(getattr(item, "offset", 0)))
            end = min(limit, start + int(getattr(item, "length", 0)))
            last_mm = max(last_mm, end)
            for position in _evenly_spaced(start, end, self.per_frame):
                samples[position] = (frame_index, "visual")

        for position in _evenly_spaced(
            max(last_mm, 0), limit, self.edge_tokens
        ):
            samples.setdefault(position, (-1, "suffix"))

        positions = sorted(samples)
        frame_indices = [samples[position][0] for position in positions]
        kinds = [samples[position][1] for position in positions]
        return positions, frame_indices, kinds

    @staticmethod
    def _gather(kv_layer: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        if kv_layer.ndim < 4:
            raise ValueError(f"Unsupported paged KV shape: {tuple(kv_layer.shape)}")
        if kv_layer.shape[0] == 2:
            block_size = kv_layer.shape[2]
            blocks, offsets = slots // block_size, slots % block_size
            values = kv_layer[:, blocks, offsets]
        elif kv_layer.shape[1] == 2:
            block_size = kv_layer.shape[2]
            blocks, offsets = slots // block_size, slots % block_size
            values = kv_layer[blocks, :, offsets].movedim(1, 0)
        else:
            raise ValueError(f"Unsupported paged KV shape: {tuple(kv_layer.shape)}")
        return values.flatten(start_dim=2).contiguous()
