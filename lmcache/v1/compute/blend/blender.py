# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Callable, Optional, Sequence, Union
import os
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import BLEND_MODES, LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.retrieval_contract import (
    expected_retrieval_count,
    validate_retrieval_count,
)

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        # layerwise_model 内部已兼容 Qwen2.5-VL 的层解析
        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # 使用 layerwise_model 暴露的层与切片信息，避免硬编码 vllm_model.model.layers
        self.layers = self.layerwise_model.layers
        self.start_layer = getattr(self.layerwise_model, "start_layer", 0)
        self.end_layer = getattr(self.layerwise_model, "end_layer", len(self.layers))
        self.num_layers = self.end_layer - self.start_layer

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        blend_mode = getattr(config, "blend_mode", "") or ""
        if not blend_mode:
            blend_mode = "codecsight" if config.is_codecsight else "topk"
        if blend_mode not in BLEND_MODES:
            raise ValueError(
                f"Unknown blend_mode {blend_mode!r}; valid modes: {BLEND_MODES}"
            )
        self.blend_mode = blend_mode
        self.gop = max(int(config.GOP), 1)
        self.vlcache_recompute_ratio = float(
            getattr(config, "vlcache_recompute_ratio", 0.05)
        )
        self.vlcache_mode = str(
            getattr(config, "vlcache_mode", "per_frame")
        )
        logger.info("Blender blend_mode=%s, GOP=%d, vlcache_ratio=%.3f, vlcache_mode=%s",
                     self.blend_mode, self.gop, self.vlcache_recompute_ratio,
                     self.vlcache_mode)

        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
            blend_mode=self.blend_mode,
            GOP=config.GOP,
            vlcache_recompute_ratio=self.vlcache_recompute_ratio,
        )
        self.skip_ffn = False
        self.skip_ffn_only_codecsight = True
        # Direct reuse does not need a model recompute pass.
        self.direct_reuse_retrieve_only = True
        self.exact_prefix_fast_path = (
            os.environ.get("LMCACHE_EXACT_PREFIX_FAST_PATH", "1") != "0"
        )
        # Default refresh budget in frames.
        self.codecsight_refresh_frames = 3
        self._single_zero_idx: dict[torch.device, torch.Tensor] = {}
        # Optional ablation instrumentation.
        self._time_sel = os.environ.get("LMCACHE_TIME_SELECTION") == "1"
        self._equal_k = os.environ.get("LMCACHE_EQUAL_K") == "1"
        if config.extra_config is not None:
            self.skip_ffn = bool(config.extra_config.get("skip_ffn", False))
            self.skip_ffn_only_codecsight = bool(
                config.extra_config.get("skip_ffn_only_codecsight", True)
            )
            if "direct_reuse_retrieve_only" in config.extra_config:
                self.direct_reuse_retrieve_only = bool(
                    config.extra_config.get("direct_reuse_retrieve_only", True)
                )
            self.codecsight_refresh_frames = int(
                config.extra_config.get(
                    "codecsight_refresh_frames",
                    config.extra_config.get("codecsight_prefix_frames", 3),
                )
            )
        if self.blend_mode == "codecsight":
            logger.info(
                "CodecSight selection: %s (refresh_frames=%d)",
                "contiguous prefix of the reused span",
                self.codecsight_refresh_frames,
            )
        if self.skip_ffn:
            logger.warning(
                "FFN skip is enabled (only_codecsight=%s). This may reduce output quality.",
                self.skip_ffn_only_codecsight,
            )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        # blend() replaces this with request-local metadata.
        self._active_metadata = self.metadata
        self._rotary_by_layer = [
            self._get_rotary_emb(layer.self_attn) for layer in self.layers
        ]

        self.is_mrope = getattr(self.layerwise_model, "is_mrope", False)
        self.mrope_section = getattr(self.layerwise_model, "mrope_section", None)
        self._mrope_model_config = None
        if self.is_mrope:
            try:
                vllm_cfg = self.layerwise_model.vllm_model.config
                self._mrope_model_config = {
                    "image_token_id": getattr(vllm_cfg, "image_token_id", 151655),
                    "video_token_id": getattr(vllm_cfg, "video_token_id", 151656),
                    "vision_start_token_id": getattr(vllm_cfg, "vision_start_token_id", 151652),
                    "spatial_merge_size": getattr(
                        getattr(vllm_cfg, "vision_config", None),
                        "spatial_merge_size", 2),
                }
                logger.info("mRoPE blender initialized with config: %s", self._mrope_model_config)
            except Exception as e:
                raise RuntimeError(
                    "Could not initialize exact mRoPE metadata for cache reuse"
                ) from e

    def _get_rotary_emb(self, attn_layer):
        if hasattr(attn_layer, "rotary_emb"):
            return attn_layer.rotary_emb
        if hasattr(attn_layer, "rotary_emb_func"):
            return attn_layer.rotary_emb_func
        raise AttributeError("Attention layer does not expose rotary embedding module.")

    def _compute_mrope_positions(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Compute Qwen3-VL positions with shape [3, num_tokens]."""
        input_ids = self._active_metadata.input_ids
        image_grid_thw = self._active_metadata.image_grid_thw
        cfg = self._mrope_model_config

        if input_ids is None or cfg is None:
            raise RuntimeError("M-RoPE input IDs or model metadata are missing")

        input_ids_for_pos = input_ids[:num_tokens]

        image_token_id = cfg["image_token_id"]
        video_token_id = cfg["video_token_id"]
        vision_start_token_id = cfg["vision_start_token_id"]
        spatial_merge_size = cfg["spatial_merge_size"]

        if image_grid_thw is None:
            image_grid_thw = []

        # Accept nested and flattened vLLM grid metadata.
        try:
            grid_values = torch.as_tensor(image_grid_thw, dtype=torch.int64)
            if grid_values.numel() % 3 != 0:
                raise ValueError(
                    f"image_grid_thw has {grid_values.numel()} values, not a multiple of 3"
                )
            flat_grid = grid_values.reshape(-1, 3).tolist()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError("Malformed image_grid_thw for mRoPE reuse") from exc

        input_tokens_tensor = torch.tensor(input_ids_for_pos)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        # Ignore a trailing vision-start marker with no placeholder token.
        vision_start_indices = vision_start_indices[
            vision_start_indices + 1 < input_tokens_tensor.numel()
        ]
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum())
        video_nums = int((vision_tokens == video_token_id).sum())

        video_grid_thw_expanded: list = []

        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        image_index, video_index = 0, 0

        for _ in range(image_nums + video_nums):
            ed_image = len(input_ids_for_pos) + 1
            ed_video = len(input_ids_for_pos) + 1
            if remain_images > 0:
                try:
                    ed_image = input_ids_for_pos.index(image_token_id, st)
                except ValueError:
                    pass
            if remain_videos > 0:
                try:
                    ed_video = input_ids_for_pos.index(video_token_id, st)
                except ValueError:
                    pass

            sentinel = len(input_ids_for_pos) + 1
            if ed_image >= sentinel and ed_video >= sentinel:
                break

            if ed_image < ed_video:
                if image_index < len(flat_grid):
                    t, h, w = flat_grid[image_index]
                else:
                    raise RuntimeError(
                        "image_grid_thw does not cover all image placeholders: "
                        f"index={image_index}, grids={len(flat_grid)}"
                    )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            elif ed_video < sentinel:
                if video_index < len(video_grid_thw_expanded):
                    t, h, w = video_grid_thw_expanded[video_index]
                else:
                    raise RuntimeError(
                        "video placeholder reached the image-path mRoPE cache "
                        "implementation without video grid metadata"
                    )
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            else:
                break

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            text_len = ed - st

            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if llm_pos_ids_list
                else 0
            )
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_ids_for_pos):
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if llm_pos_ids_list
                else 0
            )
            text_len = len(input_ids_for_pos) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        if not llm_pos_ids_list:
            return torch.arange(num_tokens, device=device, dtype=torch.int64)

        positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        return positions.to(device=device, dtype=torch.int64)

    def _compute_hit_indices(self, effective_len: int, device: torch.device):
        """Return indices of cache-*hit* tokens (excluding gaps)."""
        gap_positions = getattr(
            self.gpu_connector, "current_gap_positions", None
        )
        if gap_positions is None or gap_positions.numel() == 0:
            return torch.arange(effective_len, device=device, dtype=torch.long)
        hit_mask = torch.ones(effective_len, device=device, dtype=torch.bool)
        if gap_positions.device != device or gap_positions.dtype != torch.long:
            gap_positions = gap_positions.to(device=device, dtype=torch.long)
        valid_gap = gap_positions[
            (gap_positions >= 0) & (gap_positions < effective_len)
        ]
        hit_mask[valid_gap] = False
        return torch.where(hit_mask)[0]

    def _codecsight_select(
        self,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select a contiguous visual prefix for CodecSight refresh."""
        return self._prefix_select(hit_indices, effective_len, device)

    def _prefix_select(
        self,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select the first N visual frames as one contiguous token run."""
        n_frames = max(1, int(self.codecsight_refresh_frames))
        hits = hit_indices[hit_indices < effective_len]
        if hits.numel() == 0:
            return hit_indices.new_empty((0,))

        tokens_per_frame = int(self._active_metadata.tokens_per_frame or 0)
        mm_positions: Optional[Sequence[Any]] = self._active_metadata.mm_positions

        # Exclude cached template tokens from the refresh budget.
        start = end = None
        if mm_positions:
            for placeholder in mm_positions:
                off = int(getattr(placeholder, "offset", 0))
                length = int(getattr(placeholder, "length", 0))
                if length <= 0 or off >= effective_len:
                    continue
                if start is None:
                    start = off
                end = min(off + length, effective_len)
                n_frames -= 1
                if n_frames == 0:
                    break
        if start is None:
            # Approximate the budget when placeholder metadata is absent.
            per = tokens_per_frame if tokens_per_frame > 0 else 256
            start = int(hits.min().item())
            end = min(start + max(1, int(self.codecsight_refresh_frames)) * per,
                      effective_len)
            logger.warning_once(
                "codecsight prefix selection has no mm_positions; anchoring the "
                "prefix at the first cache hit, which may include prompt tokens."
            ) if hasattr(logger, "warning_once") else logger.warning(
                "codecsight prefix selection has no mm_positions; anchoring the "
                "prefix at the first cache hit, which may include prompt tokens."
            )

        selected = hits[(hits >= start) & (hits < end)]
        if selected.numel() == 0:
            selected = hits[:1]
        return selected

    def _random_select(
        self,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select a deterministic random control with CodecSight's budget."""
        k = int(self._codecsight_select(hit_indices, effective_len, device).numel())
        n = int(hit_indices.numel())
        if k <= 0 or n == 0:
            return hit_indices[:1] if n > 0 else hit_indices
        if k >= n:
            return hit_indices
        g = torch.Generator(device="cpu")
        g.manual_seed(1234 + effective_len)  # stable across layers/runs
        perm = torch.randperm(n, generator=g).to(device)
        sel = hit_indices[perm[:k]]
        sel, _ = torch.sort(sel)
        return sel

    def _vlcache_select(
        self,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """VLCache baseline: select image-token prefix for recomputation.

        Two sub-modes controlled by ``self.vlcache_mode``:
          - ``per_frame``: floor(r * T_i) from each frame (paper-faithful)
          - ``prefix``:    first r% of all image tokens concatenated
        Requires multimodal positions so the configured VLCache strategy is
        not silently replaced by a different selection policy.
        """
        mm_positions: Optional[Sequence[Any]] = self._active_metadata.mm_positions
        r = self.vlcache_recompute_ratio

        if not mm_positions:
            raise RuntimeError(
                "VLCache selection requires mm_positions; refusing a silent "
                "prefix-of-all fallback"
            )

        logger.debug(
            "vlcache mode=%s: %d frames in mm_positions, effective_len=%d, r=%.4f",
            self.vlcache_mode, len(mm_positions), effective_len, r,
        )

        if self.vlcache_mode == "prefix":
            selected = self._vlcache_video_prefix(
                mm_positions, r, hit_indices, effective_len, device,
            )
        else:
            selected = self._vlcache_per_frame(
                mm_positions, r, hit_indices, effective_len, device,
            )

        logger.debug(
            "vlcache mode=%s selected %d tokens for recompute out of %d hit tokens",
            self.vlcache_mode, selected.numel(), hit_indices.numel(),
        )
        return selected

    def _vlcache_per_frame(
        self,
        mm_positions: Sequence[Any],
        r: float,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Paper-faithful: floor(r * T_i) prefix tokens from each frame."""
        selected: list[torch.Tensor] = []
        for placeholder in mm_positions:
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= effective_len:
                continue
            end = min(start + length, effective_len)
            n = max(1, int((end - start) * r))
            frame_hit = hit_indices[
                (hit_indices >= start) & (hit_indices < start + n)
            ]
            if frame_hit.numel() > 0:
                selected.append(frame_hit)
        if selected:
            return torch.cat(selected)
        if hit_indices.numel() > 0:
            return hit_indices[:1]
        return hit_indices.new_empty((0,))

    def _vlcache_video_prefix(
        self,
        mm_positions: Sequence[Any],
        r: float,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Whole-video prefix: first r% of all image tokens concatenated."""
        all_image: list[torch.Tensor] = []
        for placeholder in mm_positions:
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= effective_len:
                continue
            end = min(start + length, effective_len)
            frame_hit = hit_indices[
                (hit_indices >= start) & (hit_indices < end)
            ]
            if frame_hit.numel() > 0:
                all_image.append(frame_hit)
        if not all_image:
            if hit_indices.numel() > 0:
                return hit_indices[:1]
            return hit_indices.new_empty((0,))
        all_image_t = torch.cat(all_image)
        budget = max(1, int(all_image_t.numel() * r))
        return all_image_t[:budget]

    def _apply_selected_indices(
        self,
        selected_indices: torch.Tensor,
        effective_len: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        residual: torch.Tensor,
        attn_output: torch.Tensor,
        attn_metadata: LMCAttnMetadata,
        rotary,
        layer_id: int,
        mode_label: str,
    ):
        """Apply the first layer's token selection to one decoder layer."""
        num_tokens = q.shape[0]

        if self._active_metadata.imp_indices is None:
            self._active_metadata.imp_indices = selected_indices
            if self._active_metadata.positions.ndim == 2:
                self._active_metadata.positions = self._active_metadata.positions[:, selected_indices]
            else:
                self._active_metadata.positions = self._active_metadata.positions[selected_indices]
            self._active_metadata.selection_effective_len = effective_len
            self._active_metadata.is_full_selection = (
                selected_indices.numel() == effective_len
            )
            logger.debug(
                "%s selected %d/%d tokens for recompute.",
                mode_label,
                int(selected_indices.numel()),
                num_tokens,
            )

        imp_indices = self._active_metadata.imp_indices
        assert imp_indices is not None
        sel_eff_len = int(self._active_metadata.selection_effective_len or 0)
        if sel_eff_len > 0 and effective_len >= sel_eff_len:
            layer_imp = imp_indices
            layer_positions = self._active_metadata.positions
        else:
            valid_mask = imp_indices < effective_len
            layer_imp = imp_indices[valid_mask]
            if self._active_metadata.positions.ndim == 2:
                layer_positions = self._active_metadata.positions[:, valid_mask]
            else:
                layer_positions = self._active_metadata.positions[valid_mask]

        if layer_imp.numel() == 0 and effective_len > 0:
            if q.device not in self._single_zero_idx:
                self._single_zero_idx[q.device] = torch.tensor(
                    [0], device=q.device, dtype=torch.long
                )
            layer_imp = self._single_zero_idx[q.device]
            if self._active_metadata.positions.ndim == 2:
                layer_positions = self._single_zero_idx[q.device].unsqueeze(0).expand(3, -1)
            else:
                layer_positions = self._single_zero_idx[q.device]
            self._active_metadata.imp_indices = layer_imp
            self._active_metadata.positions = layer_positions

        full_range = (
            self._active_metadata.is_full_selection
            and effective_len == num_tokens
            and sel_eff_len == num_tokens
        )
        if not full_range:
            if self._contiguous_key_len(layer_imp) is None:
                self._active_metadata.causal_blocks = self._causal_blocks(
                    layer_imp
                )
            else:
                self._active_metadata.causal_blocks = None
                attn_metadata.update_from_top_indices(layer_imp)
            k = k.index_select(0, layer_imp)
            v = v.index_select(0, layer_imp)
            q = q.index_select(0, layer_imp)
            residual = residual.index_select(0, layer_imp)
            attn_output = attn_output[: len(layer_imp)]

        q, k = rotary(layer_positions, q, k)

        old_k[layer_imp] = k
        old_v[layer_imp] = v

        # Truncate keys so FlashAttention's causal alignment matches the prefix.
        key_len = self._contiguous_key_len(layer_imp)
        if key_len is not None and key_len < old_k.shape[0]:
            attn_metadata.truncate_keys(key_len)
            return (q, old_k[:key_len], old_v[:key_len], residual,
                    attn_output, attn_metadata)
        return q, old_k, old_v, residual, attn_output, attn_metadata

    @staticmethod
    def _causal_blocks(indices: torch.Tensor) -> list[tuple[int, int, int]]:
        """Map consecutive absolute-token runs to query slices and key ends."""
        if indices.numel() == 0:
            return []
        values = indices.tolist()
        blocks: list[tuple[int, int, int]] = []
        query_start = 0
        run_start = 0
        for cursor in range(1, len(values) + 1):
            if cursor < len(values) and values[cursor] == values[cursor - 1] + 1:
                continue
            query_end = query_start + cursor - run_start
            blocks.append((query_start, query_end, values[cursor - 1] + 1))
            query_start = query_end
            run_start = cursor
        return blocks

    def forward_attention(
        self,
        backend: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: LMCAttnMetadata,
    ) -> torch.Tensor:
        blocks = self._active_metadata.causal_blocks
        if not blocks:
            return backend.forward_contiguous(q, k, v, output, attn_metadata)
        truncate_keys = getattr(attn_metadata, "truncate_keys", None)
        if truncate_keys is None:
            raise RuntimeError(
                "scattered refresh requires blockwise causal attention metadata"
            )
        imp_indices = self._active_metadata.imp_indices
        if imp_indices is None:
            raise RuntimeError("causal attention blocks have no selected indices")
        for query_start, query_end, key_end in blocks:
            query_indices = imp_indices[query_start:query_end]
            attn_metadata.update_from_top_indices(query_indices)
            truncate_keys(key_end)
            backend.forward_contiguous(
                q[query_start:query_end],
                k[:key_end],
                v[:key_end],
                output[query_start:query_end],
                attn_metadata,
            )
        return output

    @staticmethod
    def _contiguous_key_len(idx: torch.Tensor) -> Optional[int]:
        """Return a contiguous run's exclusive end, or None if scattered."""
        n = int(idx.numel())
        if n == 0:
            return None
        lo = int(idx[0].item())
        hi = int(idx[-1].item())
        if hi - lo + 1 != n:
            return None
        return hi + 1

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        """Apply the configured cache-refresh strategy to one layer."""
        logger.debug("Blender is processing KV for layer %d", layer_id)
        try:
            old_k, old_v = self.gpu_connector.get_kv(layer_id)
        except ValueError as exc:
            raise RuntimeError(
                f"KV cache for layer {layer_id} was not loaded"
            ) from exc

        if attn_output is None:
            attn_output = torch.empty(
                q.shape, dtype=q.dtype, device=q.device,
            )

        # Initialize positions once per blend request.
        if self._active_metadata.positions is None:
            if self.is_mrope and self._mrope_model_config is not None:
                self._active_metadata.positions = self._compute_mrope_positions(
                    q.shape[0], q.device,
                )
                logger.info(
                    "Computed M-RoPE positions: shape=%s",
                    list(self._active_metadata.positions.shape),
                )
            else:
                self._active_metadata.positions = torch.arange(
                    q.shape[0], device=q.device, dtype=torch.int64
                )

        rotary = self._rotary_by_layer[layer_id]

        # Direct reuse performs only positional correction.
        if self.blend_mode == "direct_reuse":
            q, _ = rotary(self._active_metadata.positions, q, k)
            return q, old_k, old_v, residual, attn_output, attn_metadata

        # CacheBlend selects tokens by K-vector divergence.
        if self.blend_mode == "topk":
            q, k = rotary(self._active_metadata.positions, q, k)
            write_indices = self._active_metadata.imp_indices

            if layer_id in self.common_metadata.check_layers:
                if self._time_sel:
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                diff_k = torch.sum(
                    (k.to(torch.float32) - old_k.to(torch.float32)) ** 2,
                    dim=[1],
                )
                total_len = diff_k.shape[0]
                assert self.common_metadata.recomp_ratios is not None
                if self._equal_k:
                    # equal-K: match the I-frame (codecsight) refresh budget
                    _hit = self._compute_hit_indices(total_len, k.device)
                    topk_num = int(
                        self._codecsight_select(_hit, total_len, k.device).numel())
                else:
                    topk_num = int(
                        total_len * self.common_metadata.recomp_ratios[0]
                    )
                logger.info(
                    "TOPK check layer=%d: total=%d, topk_num=%d, "
                    "diff_k min=%.6f max=%.6f mean=%.6f median=%.6f, "
                    "k_norm=%.4f old_k_norm=%.4f, "
                    "nonzero_diff=%d/%d",
                    layer_id, total_len, topk_num,
                    diff_k.min().item(), diff_k.max().item(),
                    diff_k.mean().item(), diff_k.median().item(),
                    k.norm().item(), old_k.norm().item(),
                    (diff_k > 1e-6).sum().item(), total_len,
                )
                top_indices = torch.topk(diff_k, k=topk_num).indices
                top_indices, _ = torch.sort(top_indices)
                if self._time_sel:
                    torch.cuda.synchronize()
                    logger.info("SELECT_TIME mode=topk layer=%d ms=%.4f k=%d",
                                layer_id, (time.perf_counter() - _t0) * 1000, topk_num)

                k, v = k[top_indices], v[top_indices]
                q = q[top_indices]
                residual = residual[top_indices]

                self._active_metadata.imp_indices = top_indices
                if self._contiguous_key_len(top_indices) is None:
                    self._active_metadata.causal_blocks = self._causal_blocks(
                        top_indices
                    )
                else:
                    self._active_metadata.causal_blocks = None
                if self._active_metadata.positions.ndim == 2:
                    self._active_metadata.positions = self._active_metadata.positions[:, top_indices]
                else:
                    self._active_metadata.positions = self._active_metadata.positions[top_indices]
                attn_output = attn_output[:topk_num]
                if self._active_metadata.causal_blocks is None:
                    attn_metadata.update_from_top_indices(top_indices)
                write_indices = top_indices

            if write_indices is not None:
                old_k[write_indices] = k
                old_v[write_indices] = v
                return q, old_k, old_v, residual, attn_output, attn_metadata
            return q, k, v, residual, attn_output, attn_metadata

        # CodecSight and VLCache use explicit token indices.
        first_layer = self._active_metadata.imp_indices is None

        if first_layer:
            effective_len = min(q.shape[0], old_k.shape[0])
            hit_indices = self._compute_hit_indices(effective_len, q.device)
            if self._time_sel:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()
            if self.blend_mode == "codecsight":
                selected = self._codecsight_select(
                    hit_indices, effective_len, q.device,
                )
            elif self.blend_mode == "random":
                selected = self._random_select(
                    hit_indices, effective_len, q.device,
                )
            else:
                selected = self._vlcache_select(
                    hit_indices, effective_len, q.device,
                )
            if self._time_sel:
                torch.cuda.synchronize()
                logger.info("SELECT_TIME mode=%s layer=%d ms=%.4f k=%d",
                            self.blend_mode, layer_id,
                            (time.perf_counter() - _t0) * 1000, int(selected.numel()))
            return self._apply_selected_indices(
                selected, effective_len, q, k, v, old_k, old_v,
                residual, attn_output, attn_metadata, rotary,
                layer_id, self.blend_mode,
            )

        # Subsequent layers: q/k/v/residual are already reduced to
        # the selected subset by compute_layer. Apply rotary and write
        # into the full KV cache at the stored indices.
        imp_indices = self._active_metadata.imp_indices
        if imp_indices is None:
            raise RuntimeError("selected-token metadata was lost between layers")
        if (imp_indices.numel() > 0
                and int(imp_indices[-1].item()) >= old_k.shape[0]):
            raise RuntimeError(
                f"layer {layer_id} KV length {old_k.shape[0]} does not cover "
                f"selected token {int(imp_indices[-1].item())}"
            )
        q, k = rotary(self._active_metadata.positions, q, k)
        old_k[imp_indices] = k
        old_v[imp_indices] = v
        key_len = self._contiguous_key_len(imp_indices)
        if key_len is not None and key_len < old_k.shape[0]:
            attn_metadata.truncate_keys(key_len)
            return (q, old_k[:key_len], old_v[:key_len], residual,
                    attn_output, attn_metadata)
        return q, old_k, old_v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to orchestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwise retrieve + blending.
        """
        # TODO(Jiayi): store is currently not included in this function
        md = self._active_metadata
        check_layers = self.common_metadata.check_layers
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        deepstack_input_embeds = kwargs.pop("deepstack_input_embeds", None)
        embedding_provider: Optional[
            Callable[[], tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ] = kwargs.pop("embedding_provider", None)
        model_input_ids = kwargs.pop("model_input_ids", tokens)
        # Cache token IDs may contain hash sentinels; the model needs originals.
        if not torch.is_tensor(model_input_ids):
            model_input_ids = torch.tensor(
                model_input_ids,
                dtype=torch.long,
                device=tokens.device,
            )
        elif model_input_ids.device != tokens.device:
            model_input_ids = model_input_ids.to(device=tokens.device)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        # warmup retriever
        warmup_retrieved = next(layerwise_retriever)
        has_retrieved_tokens = False
        if warmup_retrieved is not None:
            if torch.is_tensor(warmup_retrieved):
                has_retrieved_tokens = int(warmup_retrieved.item()) > 0
            else:
                has_retrieved_tokens = int(warmup_retrieved) > 0

        expected_retrieved = expected_retrieval_count(mask, len(tokens))
        validate_retrieval_count(
            expected=expected_retrieved,
            actual=warmup_retrieved,
            request_id=kwargs.get("req_id"),
            path=f"blend:{self.blend_mode}",
        )
        yield

        if not has_retrieved_tokens:
            logger.debug("No retrievable tokens in layerwise retrieve; skip blending compute.")
            for _ in range(self.num_layers):
                next(layerwise_retriever)
                yield
            next(layerwise_retriever)
            md.clean()
            yield
            return

        if self.blend_mode == "direct_reuse" and self.direct_reuse_retrieve_only:
            logger.info(
                "direct_reuse: retrieve-only path (skip layerwise compute_layer); "
                "KV load matches non-blending retrieve_layer."
            )
            for _ in range(self.num_layers):
                next(layerwise_retriever)
                yield
            next(layerwise_retriever)
            md.clean()
            yield
            return

        layerwise_model_executor = None
        for layer_id in range(self.num_layers):
            self._active_metadata = md
            next(layerwise_retriever)
            if (
                layer_id == 0
                and getattr(self, "exact_prefix_fast_path", True)
                and getattr(
                    self.gpu_connector, "current_exact_prefix_match", False
                )
            ):
                logger.info(
                    "%s: exact-prefix reuse (skip positional repair and "
                    "selective recompute)",
                    self.blend_mode,
                )
                yield
                for _ in range(1, self.num_layers):
                    next(layerwise_retriever)
                    yield
                next(layerwise_retriever)
                md.clean()
                yield
                return

            if layerwise_model_executor is None:
                if embedding_provider is not None:
                    inputs_embeds, deepstack_input_embeds = embedding_provider()
                layerwise_model_executor = self.layerwise_model.compute_layer(
                    check_layers,
                    model_input_ids,
                    inputs_embeds=inputs_embeds,
                    deepstack_input_embeds=deepstack_input_embeds,
                )
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)

        md.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Retrieve and refresh the selected cached tokens eagerly."""

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()
        logger.info("enter blend")
        md = LMCBlendMetadata(imp_indices=None, attn_mask=None, positions=None)
        self._active_metadata = md
        tokens_per_frame = kwargs.get("tokens_per_frame")
        if tokens_per_frame is not None:
            self._active_metadata.tokens_per_frame = int(tokens_per_frame)
        mm_positions = kwargs.get("mm_positions")
        if mm_positions is not None:
            self._active_metadata.mm_positions = mm_positions
        image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is not None:
            self._active_metadata.image_grid_thw = image_grid_thw
        model_input_ids = kwargs.get("model_input_ids", tokens)
        if isinstance(model_input_ids, torch.Tensor):
            self._active_metadata.input_ids = model_input_ids.tolist()
        else:
            self._active_metadata.input_ids = list(model_input_ids)

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)
        # Two extra yields open and close the layerwise retrieval pipeline.
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
        return None
