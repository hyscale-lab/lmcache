# SPDX-License-Identifier: Apache-2.0
# Standard
import hashlib
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Generator, Mapping, Optional, Union

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    RefreshSpan,
    RefreshSpec,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.sampling_params import SamplingParams

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache.utils import cdiv

# Try to import from old location before merged https://github.com/vllm-project/vllm/pull/26908
try:
    # Third Party
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
except ImportError:
    # Third Party
    from vllm.utils import get_kv_cache_torch_dtype

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.resident_kv_registry import (
    FrameDecision,
    ResidentKVRegistry,
    build_frame_token_block_plan,
    get_default_resident_kv_registry,
)
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
from lmcache import utils
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_image_grid_thw,
    extract_mm_features,
    lmcache_get_or_create_config,
    mla_enabled,
)
from lmcache.integration.vllm.kv_diagnostics import KVDiagnostic
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    LMCacheEngineBuilder,
    LayerwiseRetrievalBatchInfo,
    LayerwiseRetrievalRequest,
)
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.compute.models.base import _resolve_decoder_layers
from lmcache.v1.compute.positional_encoding import get_fused_rope_from_vllm
from lmcache.v1.config import LMCacheEngineConfig, _validate_and_set_config_value
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
    _mrope_delta_rotate_k,
)
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupServer,
)
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.plugin_launcher import PluginLauncher
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.v1.retrieval_contract import (
    expected_retrieval_count,
    validate_retrieval_count,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _context_prefix_hashes(
    context_tokens: Optional[torch.Tensor],
    ends: list[int],
) -> dict[int, bytes]:
    if context_tokens is None or not ends:
        return {}
    tokens = context_tokens.contiguous().numpy()
    raw = memoryview(tokens).cast("B")
    item_size = int(tokens.dtype.itemsize)
    digest = hashlib.sha256()
    result: dict[int, bytes] = {}
    previous = 0
    for end in sorted(set(ends)):
        if end < previous or end > int(tokens.size):
            raise ValueError("context prefix endpoint is out of range")
        digest.update(raw[previous * item_size:end * item_size])
        result[end] = digest.digest()
        previous = end
    return result


def _resident_kv_modes(
    is_codecsight: bool,
    extra_config: Optional[Mapping[str, Any]],
    blend_mode: Optional[str] = None,
) -> tuple[bool, bool]:
    """Resolve CoStream-only shadow and pinned-prototype feature flags."""
    extra = extra_config or {}
    if not is_codecsight or blend_mode not in (None, "", "codecsight"):
        return False, False
    return (
        bool(extra.get("resident_kv_shadow", False)),
        bool(extra.get("resident_kv_zero_copy_single_stream", False)),
    )


def _resident_frame_block_records(
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
    block_ids: list[int],
    block_size: int,
    token_ids: Optional[list[int]] = None,
    decisions: Optional[list[FrameDecision]] = None,
) -> tuple[list[dict[str, Any]], int]:
    """Resolve each frame's full interior blocks and isolate its boundaries."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    context_tokens: Optional[torch.Tensor] = None
    if token_ids is not None:
        context_tokens = torch.as_tensor(
            token_ids, dtype=torch.int64, device="cpu"
        ).clone()
        apply_mm_hashes_to_token_ids(
            context_tokens, mm_hashes, mm_positions
        )

    prefix_limit = len(block_ids) * block_size
    if context_tokens is not None:
        prefix_limit = min(prefix_limit, int(context_tokens.numel()))
    plan = build_frame_token_block_plan(
        mm_hashes,
        mm_positions,
        block_size,
        decisions=decisions,
        prefix_limit=prefix_limit,
    )

    records: list[dict[str, Any]] = []
    decisions_by_index = {
        frame.decision.frame_index: frame.decision for frame in plan.frames
    }
    context_hashes = _context_prefix_hashes(
        context_tokens,
        [segment.token_start + segment.token_length
         for segment in plan.reusable_segments],
    )
    for segment in plan.reusable_segments:
        start = segment.token_start
        length = segment.token_length
        end = start + length
        records.append({
            "key": segment.key,
            "content_hash": segment.content_hash,
            "frame_index": segment.frame_index,
            "frame_relative_start": segment.frame_relative_start,
            "frame_decision": decisions_by_index[segment.frame_index],
            "token_start": start,
            "token_length": length,
            "position_fingerprint": f"{start}:{length}",
            "context_hash": context_hashes.get(end),
            "block_ids": tuple(block_ids[start // block_size:end // block_size]),
        })
    non_reusable_frames = (
        len(plan.frames) - plan.frames_with_reusable_blocks
        + plan.malformed_frames
    )
    return records, non_reusable_frames


def _resident_frame_token_records(
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
    block_ids: list[int],
    block_size: int,
    token_ids: Optional[list[int]] = None,
    decisions: Optional[list[FrameDecision]] = None,
) -> tuple[list[dict[str, Any]], int]:
    """Map complete frame token slices across arbitrary block boundaries."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    prefix_limit = len(block_ids) * block_size
    context_tokens: Optional[torch.Tensor] = None
    if token_ids is not None:
        context_tokens = torch.as_tensor(
            token_ids, dtype=torch.int64, device="cpu"
        ).clone()
        apply_mm_hashes_to_token_ids(
            context_tokens, mm_hashes, mm_positions
        )
        prefix_limit = min(prefix_limit, int(context_tokens.numel()))
    plan = build_frame_token_block_plan(
        mm_hashes,
        mm_positions,
        block_size,
        decisions=decisions,
        prefix_limit=prefix_limit,
    )
    records: list[dict[str, Any]] = []
    context_hashes = _context_prefix_hashes(
        context_tokens,
        [frame.token_start + frame.token_length for frame in plan.frames
         if frame.decision.action != "drop"],
    )
    for frame in plan.frames:
        decision = frame.decision
        if decision.action == "drop":
            continue
        start = frame.token_start
        length = frame.token_length
        end = start + length
        first_block = start // block_size
        last_block = (end - 1) // block_size
        frame_blocks = tuple(block_ids[first_block:last_block + 1])
        if not frame_blocks:
            continue
        records.append({
            "key": (
                "costream-frame-token-v1",
                decision.content_hash,
                decision.codec_digest or "",
                length,
            ),
            "content_hash": decision.content_hash,
            "frame_index": decision.frame_index,
            "frame_decision": decision,
            "token_start": start,
            "token_length": length,
            "source_token_offset": start % block_size,
            "position_fingerprint": f"{start}:{length}",
            "context_hash": context_hashes.get(end),
            "block_ids": frame_blocks,
        })
    return records, plan.malformed_frames


def _resident_prompt_block_record(
    prompt_token_ids: Optional[list[int]],
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
    block_size: int,
    *,
    block_ids: Optional[list[int]] = None,
    cache_salt: Optional[str] = None,
    lora_id: int = 0,
) -> Optional[dict[str, Any]]:
    """Build the exact-prefix identity used for resident block adoption."""
    if not prompt_token_ids or block_size <= 0:
        return None
    prefix_tokens = ((len(prompt_token_ids) - 1) // block_size) * block_size
    if prefix_tokens <= 0:
        return None
    required_blocks = prefix_tokens // block_size
    if block_ids is not None and len(block_ids) < required_blocks:
        return None

    identity_tokens = torch.as_tensor(
        prompt_token_ids, dtype=torch.int64, device="cpu"
    ).clone()
    if mm_hashes and mm_positions:
        apply_mm_hashes_to_token_ids(
            identity_tokens, mm_hashes, mm_positions
        )
    digest = hashlib.sha256(
        identity_tokens[:prefix_tokens].contiguous().numpy().tobytes()
    ).digest()
    digest_hex = digest.hex()
    key = (
        "vllm-resident-prefix-v1",
        int(lora_id),
        str(cache_salt or ""),
        prefix_tokens,
        digest_hex,
    )
    return {
        "key": key,
        "content_hash": digest_hex,
        "context_hash": digest,
        "token_start": 0,
        "token_length": prefix_tokens,
        "position_fingerprint": f"0:{prefix_tokens}",
        "block_ids": (
            tuple(block_ids[:required_blocks])
            if block_ids is not None else ()
        ),
    }


def _resident_text_prefix_record(
    prompt_token_ids: Optional[list[int]],
    mm_positions: list["PlaceholderRange"],
    block_size: int,
    *,
    block_ids: Optional[list[int]] = None,
    cache_salt: Optional[str] = None,
    lora_id: int = 0,
    stream_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Describe the exact causal text prefix before the first MM item.

    Unlike the whole-prompt record, this slice may end inside a KV block.  The
    overlap path copies only its exact token slots into fresh destination
    blocks, so the visual tokens sharing its final source block are never
    adopted accidentally.  Stream identity prevents an identical global
    prefix from being republished out from under an in-flight stream.
    """
    if not prompt_token_ids or not mm_positions or block_size <= 0:
        return None
    first_mm_offset = min(
        int(getattr(position, "offset", -1)) for position in mm_positions
    )
    if first_mm_offset <= 0:
        return None
    computed_prefix = (
        len(block_ids) * block_size
        if block_ids is not None
        else ((len(prompt_token_ids) - 1) // block_size) * block_size
    )
    token_length = min(
        first_mm_offset, len(prompt_token_ids), computed_prefix
    )
    if token_length <= 0:
        return None
    digest = hashlib.sha256(
        torch.as_tensor(
            prompt_token_ids[:token_length], dtype=torch.int64, device="cpu"
        ).contiguous().numpy().tobytes()
    ).digest()
    required_blocks = cdiv(token_length, block_size)
    if block_ids is not None and len(block_ids) < required_blocks:
        return None
    digest_hex = digest.hex()
    return {
        "key": (
            "costream-resident-text-prefix-v1",
            str(stream_id or ""),
            int(lora_id),
            str(cache_salt or ""),
            token_length,
            digest_hex,
        ),
        "content_hash": digest_hex,
        "context_hash": digest,
        "token_start": 0,
        "token_length": token_length,
        "source_token_offset": 0,
        "position_fingerprint": f"0:{token_length}",
        "block_ids": (
            tuple(block_ids[:required_blocks])
            if block_ids is not None else ()
        ),
    }


def _patch_vllm_model_registration():
    """
    Some vLLM builds miss the LMCache registration hook. Patch GPUModelRunner
    so the underlying model is registered once it is loaded, allowing the
    blender to fetch it safely later.
    """
    try:
        # Third Party
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Skip GPUModelRunner patch: %s", exc)
        return

    if getattr(GPUModelRunner.load_model, "_lmcache_patched", False):
        return

    orig_load_model = GPUModelRunner.load_model

    def _load_model_with_register(self, *args, **kwargs):
        orig_load_model(self, *args, **kwargs)
        try:
            VLLMModelTracker.register_model(ENGINE_NAME, self.model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to register vLLM model for LMCache: %s", exc)
        try:
            VLLMModelTracker.register_encoder_cache(
                ENGINE_NAME, self.encoder_cache
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not register encoder_cache: %s", exc)
        try:
            recompute = getattr(
                self, "_recompute_lmcache_encoder_outputs", None
            )
            if callable(recompute):
                VLLMModelTracker.register_encoder_recompute_callback(
                    ENGINE_NAME, recompute
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to register encoder recompute callback: %s", exc
            )

    _load_model_with_register._lmcache_patched = True  # type: ignore[attr-defined]
    GPUModelRunner.load_model = _load_model_with_register


_patch_vllm_model_registration()


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith(("lmcache.", "costream.")):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


def _extract_costream_frame_decisions(
    request: Any,
    mm_hashes: list[str],
) -> tuple[Optional[list[FrameDecision]], Optional[str]]:
    """Decode frame metadata while keeping server MM hashes authoritative."""
    sampling_params = getattr(request, "sampling_params", None)
    configs = (
        extract_request_configs(sampling_params)
        if sampling_params is not None
        else getattr(request, "request_configs", None)
    ) or {}
    payload = configs.get("costream.frame_decisions")
    if payload is None:
        return None, None
    if not isinstance(payload, list) or len(payload) != len(mm_hashes):
        return None, "frame_decision_count_mismatch"

    decisions: list[FrameDecision] = []
    try:
        for index, (content_hash, item) in enumerate(
            zip(mm_hashes, payload, strict=True)
        ):
            if not isinstance(item, Mapping):
                return None, "frame_decision_not_mapping"
            if int(item.get("request_frame_index", index)) != index:
                return None, "frame_decision_order_mismatch"
            decisions.append(FrameDecision(
                frame_index=index,
                content_hash=str(content_hash),
                action=str(item.get("action", "keep")),
                frame_id=(
                    str(item["frame_id"])
                    if item.get("frame_id") is not None else None
                ),
                pts_seconds=(
                    float(item["pts_seconds"])
                    if item.get("pts_seconds") is not None else None
                ),
                gop_id=(
                    int(item["gop_id"])
                    if item.get("gop_id") is not None else None
                ),
                frame_type=(
                    str(item["frame_type"])
                    if item.get("frame_type") is not None else None
                ),
                is_anchor=bool(item.get("is_anchor", False)),
                anchor_frame_index=(
                    int(item["anchor_frame_index"])
                    if item.get("anchor_frame_index") is not None else None
                ),
                motion_score=(
                    float(item["motion_score"])
                    if item.get("motion_score") is not None else None
                ),
                kept_tokens=(
                    int(item["kept_tokens"])
                    if item.get("kept_tokens") is not None else None
                ),
                dense_tokens=(
                    int(item["dense_tokens"])
                    if item.get("dense_tokens") is not None else None
                ),
                codec_digest=(
                    str(item["codec_digest"])
                    if item.get("codec_digest") is not None else None
                ),
            ))
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_frame_decision"
    try:
        padding_count = int(configs.get(
            "costream.padding_frame_count", 0
        ))
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_padding_frame_count"
    if not 0 <= padding_count <= len(decisions):
        return None, "invalid_padding_frame_count"
    if padding_count and any(
        decision.action != "refresh"
        for decision in decisions[-padding_count:]
    ):
        return None, "padding_frame_not_refreshed"
    anchor_gops = {
        decision.gop_id for decision in decisions
        if decision.is_anchor and decision.gop_id is not None
    }
    if padding_count and any(
        decision.gop_id is not None
        and not decision.is_anchor
        and decision.gop_id not in anchor_gops
        for decision in decisions
    ):
        return None, "gop_anchor_missing"
    return decisions, None


def _costream_force_full_compute(request: Any) -> bool:
    sampling_params = getattr(request, "sampling_params", None)
    configs = (
        extract_request_configs(sampling_params)
        if sampling_params is not None
        else getattr(request, "request_configs", None)
    ) or {}
    return bool(configs.get("costream.force_full_compute", False))


def _costream_stream_id(request: Any) -> Optional[str]:
    sampling_params = getattr(request, "sampling_params", None)
    configs = (
        extract_request_configs(sampling_params)
        if sampling_params is not None
        else getattr(request, "request_configs", None)
    ) or {}
    value = configs.get("costream.stream_id")
    return str(value) if value is not None else None


def _costream_padding_frame_count(request: Any) -> int:
    sampling_params = getattr(request, "sampling_params", None)
    configs = (
        extract_request_configs(sampling_params)
        if sampling_params is not None
        else getattr(request, "request_configs", None)
    ) or {}
    try:
        return int(configs.get("costream.padding_frame_count", 0))
    except (TypeError, ValueError, OverflowError):
        return 0


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # Per-image grid dimensions [t, h, w] for M-RoPE position computation
    image_grid_thw: Optional[list] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)
        image_grid_thw = extract_image_grid_thw(new_request)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            image_grid_thw=image_grid_thw or None,
            skip_save=skip_save,
            request_configs=request_configs,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        self.token_ids.extend(new_token_ids)

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Token IDs before cache-key sentinels are applied.
    model_token_ids: list[int]
    # Slot mapping
    slot_mapping: torch.Tensor
    # Prefix length eligible for write-back. Loading may require a longer,
    # non-chunk-aligned prefix than storage accepts.
    save_token_count: int

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # Number of tokens produced by one frame.
    tokens_per_frame: Optional[int] = None
    # Multimodal placeholder positions for precise frame alignment.
    mm_positions: Optional[list["PlaceholderRange"]] = None
    # Multimodal content hashes for encoder_cache lookup.
    mm_hashes: Optional[list[str]] = None
    # Per-image grid dimensions [t, h, w] for M-RoPE position computation
    image_grid_thw: Optional[list] = None
    refresh_spec: Optional[RefreshSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 1024,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)
        
        is_last_prefill = False
        if input_token_len == tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (
                discard_partial_chunks
                and tracker.num_saved_tokens > 0
                and input_token_len < chunk_boundary
            )
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len
        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        metadata_token_count = num_tokens_to_save
        if load_spec is not None and load_spec.can_load:
            if load_spec.lmcache_cached_tokens > input_token_len:
                raise RuntimeError(
                    "LMCache cache hit exceeds the scheduled request prefix "
                    f"({load_spec.lmcache_cached_tokens} > {input_token_len})"
                )
            metadata_token_count = max(
                metadata_token_count,
                load_spec.lmcache_cached_tokens,
            )

        token_ids = input_token_ids[:metadata_token_count]
        model_token_ids = token_ids.copy()

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!"
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.lmcache_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        tokens_per_frame: Optional[int] = None
        if tracker.mm_positions and len(tracker.mm_positions) > 0:
            tokens_per_frame = int(getattr(tracker.mm_positions[0], "length", 0))
            if tokens_per_frame <= 0:
                tokens_per_frame = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            model_token_ids=model_token_ids,
            slot_mapping=slot_mapping,
            save_token_count=num_tokens_to_save,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            tokens_per_frame=tokens_per_frame,
            mm_positions=tracker.mm_positions,
            mm_hashes=tracker.mm_hashes,
            image_grid_thw=tracker.image_grid_thw,
        )


@dataclass(frozen=True)
class EmbeddingReconstructionResult:
    """Outcome of rebuilding embeddings for selective KV refresh."""

    inputs_embeds: Optional[torch.Tensor]
    deepstack_input_embeds: Optional[torch.Tensor]
    status: str
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def no_visual_prefix(self) -> bool:
        return self.status == "no_visual_prefix"


def _has_visual_prefix(
    mm_hashes: Optional[list[str]],
    mm_positions: Optional[list["PlaceholderRange"]],
    num_tokens: int,
) -> bool:
    if not mm_hashes or not mm_positions:
        return False
    return any(
        int(getattr(placeholder, "length", 0)) > 0
        and int(getattr(placeholder, "offset", 0)) < num_tokens
        for _, placeholder in zip(mm_hashes, mm_positions, strict=False)
    )


def _select_prefix_refresh_span(
    mm_positions: list["PlaceholderRange"],
    cached_tokens: int,
    refresh_frames: int,
    minimum_start: int = 0,
) -> Optional[tuple[int, int]]:
    span_start = None
    span_end = None
    remaining = max(1, refresh_frames)
    for placeholder in mm_positions:
        start = int(getattr(placeholder, "offset", 0))
        length = int(getattr(placeholder, "length", 0))
        if length <= 0 or start >= cached_tokens:
            continue
        end = min(start + length, cached_tokens)
        if end <= minimum_start:
            continue
        if span_start is None:
            span_start = max(start, minimum_start)
        span_end = end
        remaining -= 1
        if remaining == 0:
            break
    if span_start is None or span_end is None or span_start >= span_end:
        return None
    return span_start, span_end


def _select_multi_anchor_refresh_spans(
    mm_positions: list["PlaceholderRange"],
    decisions: list[FrameDecision],
    cached_tokens: int,
    minimum_start: int = 0,
) -> tuple[RefreshSpan, ...]:
    """Select every codec I-frame represented in the retrieved prefix."""
    if len(mm_positions) != len(decisions):
        raise ValueError("multi-anchor decisions must match MM positions")
    spans: list[RefreshSpan] = []
    for index, (placeholder, decision) in enumerate(
        zip(mm_positions, decisions, strict=True)
    ):
        if not (decision.is_anchor or decision.action == "refresh"):
            continue
        start = max(
            int(getattr(placeholder, "offset", 0)), minimum_start
        )
        end = min(
            int(getattr(placeholder, "offset", 0))
            + int(getattr(placeholder, "length", 0)),
            cached_tokens,
        )
        if start < end:
            spans.append(RefreshSpan(start, end, index))
    return tuple(spans)


def _configured_refresh_policy(extra_config: Mapping[str, Any]) -> str:
    policy = str(extra_config.get(
        "codecsight_refresh_policy",
        extra_config.get("refresh_policy", "prefix"),
    ))
    if policy not in ("prefix", "multi_anchor"):
        raise ValueError(f"unknown CodecSight refresh policy: {policy}")
    return policy


def _count_visual_tokens_in_spans(
    mm_positions: Optional[list["PlaceholderRange"]],
    spans: tuple[tuple[int, int], ...],
) -> int:
    """Count multimodal placeholder tokens covered by disjoint spans.

    This deliberately uses the same placeholder-token definition as the eager
    blender's controlled-budget counters. Text separators inside a joint
    refresh span are decoder work, but are not visual recompute budget.
    """
    if not mm_positions or not spans:
        return 0
    count = 0
    for placeholder in mm_positions:
        visual_start = max(0, int(getattr(placeholder, "offset", 0)))
        visual_length = max(0, int(getattr(placeholder, "length", 0)))
        visual_end = visual_start + visual_length
        if visual_start >= visual_end:
            continue
        for span_start, span_end in spans:
            count += max(
                0,
                min(visual_end, int(span_end))
                - max(visual_start, int(span_start)),
            )
    return count


def _visual_indices_in_spans(
    mm_positions: Optional[list["PlaceholderRange"]],
    spans: tuple[tuple[int, int], ...],
) -> list[int]:
    if not mm_positions or not spans:
        return []
    selected: list[int] = []
    for index, placeholder in enumerate(mm_positions):
        visual_start = max(0, int(getattr(placeholder, "offset", 0)))
        visual_end = visual_start + max(
            0, int(getattr(placeholder, "length", 0)))
        if any(
            max(visual_start, int(span_start))
            < min(visual_end, int(span_end))
            for span_start, span_end in spans
        ):
            selected.append(index)
    return selected


def _joint_refresh_selection_stats(
    refresh_spec: RefreshSpec,
    mm_positions: Optional[list["PlaceholderRange"]],
) -> dict[str, Any]:
    """Return CodecSight joint-refresh counters in eager-baseline units."""
    candidate_visual = _count_visual_tokens_in_spans(
        mm_positions,
        ((0, refresh_spec.cached_prefix_tokens),),
    )
    refresh_spans = tuple(
        (span.start, span.end) for span in refresh_spec.spans
    )
    selected_visual = _count_visual_tokens_in_spans(
        mm_positions,
        refresh_spans,
    )
    return {
        "recompute_selection_mode": f"joint_{refresh_spec.policy}_refresh",
        "recompute_candidate_tokens": int(refresh_spec.cached_prefix_tokens),
        "recompute_selected_tokens": int(refresh_spec.num_refresh_tokens),
        "recompute_candidate_visual_tokens": candidate_visual,
        "recompute_selected_visual_tokens": selected_visual,
        "recompute_candidate_visual_indices": _visual_indices_in_spans(
            mm_positions, ((0, refresh_spec.cached_prefix_tokens),)),
        "recompute_selected_visual_indices": _visual_indices_in_spans(
            mm_positions, refresh_spans),
        "recompute_refresh_spans": [list(span) for span in refresh_spans],
        "recompute_visual_ratio": (
            selected_visual / candidate_visual
            if candidate_visual else None
        ),
    }


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    if lmcache_config.enable_pd:
        return False
    else:
        return True


def _calculate_draft_layers(vllm_config, model_config):
    num_draft_layers = 0
    if vllm_config is not None and vllm_config.speculative_config is not None:
        logger.info(f"vllm_config.speculative_config: {vllm_config.speculative_config}")
        # TODO(baoloongmao): Support other MTP/draft methods
        if vllm_config.speculative_config.method == "deepseek_mtp":
            num_draft_layers = getattr(
                model_config.hf_config, "num_nextn_predict_layers", 0
            )
        elif vllm_config.speculative_config.use_eagle():
            try:
                draft_model_config = vllm_config.speculative_config.draft_model_config
                num_draft_layers = draft_model_config.get_num_layers(
                    vllm_config.parallel_config
                )
                logger.info(f"EAGLE detected {num_draft_layers} extra layer(s)")
            except Exception:
                logger.info(
                    "EAGLE detected, but failed to get the number of extra layers"
                    "falling back to 1"
                )
                num_draft_layers = 1
    return num_draft_layers


def _init_lmcache_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
    role: str,
) -> LMCacheEngine:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param lmcache_config: The LMCache configuration.
    :type lmcache_config: LMCacheEngineConfig
    :param vllm_config: The vLLM configuration.
    :type vllm_config: VllmConfig

    :return: The initialized LMCache engine
    :rtype: LMCacheEngine
    """
    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    assert isinstance(lmcache_config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = mla_enabled(model_config)
    if use_mla and (
        lmcache_config.remote_serde != "naive"
        and lmcache_config.remote_serde is not None
    ):
        raise ValueError("MLA only works with naive serde mode..")

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    num_draft_layers = _calculate_draft_layers(vllm_config, model_config)
    num_layer += num_draft_layers
    chunk_size = lmcache_config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
    logger.info(
        f"use mla: {use_mla}, kv shape: {kv_shape}, num_draft_layers:{num_draft_layers}"
    )

    # Change current device.
    num_gpus = torch.cuda.device_count()
    local_rank = parallel_config.rank % num_gpus
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    metadata = LMCacheEngineMetadata(
        model_config.model,
        parallel_config.world_size,
        parallel_config.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
        role,
    )

    use_gpu = need_gpu_interm_buffer(lmcache_config)
    vllm_gpu_connector: Optional[GPUConnectorInterface]

    if use_mla and lmcache_config.use_layerwise:
        raise ValueError("layerwise MLA connector is not supported yet")

    # When use_mla is True, num_kv_head is 1
    hidden_dim_size = num_kv_head * head_size
    if role == "scheduler":
        vllm_gpu_connector = None
        # Create a dummy tpg object with broadcast and broadcast_object methods
        tpg = SimpleNamespace()
        tpg.broadcast = lambda tensor, src: tensor
        tpg.broadcast_object = lambda obj, src: obj
    elif lmcache_config.use_layerwise:
        if lmcache_config.enable_blending:
            # Use layerwise connector for blending
            vllm_gpu_connector = VLLMBufferLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            vllm_gpu_connector = VLLMPagedMemLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        tpg = get_tp_group()
    else:
        vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size,
            num_layer,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=kv_dtype,
            device=device,
            use_mla=use_mla,
        )
        tpg = get_tp_group()
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        lmcache_config,
        metadata,
        vllm_gpu_connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )
    if role == "scheduler" and lmcache_config.enable_scheduler_bypass_lookup:
        assert engine.save_only_first_rank or lmcache_config.get_extra_config_value(
            "remote_enable_mla_worker_id_as0", metadata.use_mla
        ), (
            "enable_scheduler_bypass_lookup is only supported with "
            "save_only_first_rank or remote_enable_mla_worker_id_as0"
        )
    return engine


@dataclass(frozen=True)
class ResidentTokenCopySpec:
    request_id: str
    source_slots: tuple[int, ...]
    target_slots: tuple[int, ...]
    source_token_start: int
    target_token_start: int
    position_mode: str
    repair_positions: bool

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("resident token copy requires a request ID")
        if not self.source_slots or len(self.source_slots) != len(
            self.target_slots
        ):
            raise ValueError("resident source/target token slots must match")
        if any(slot < 0 for slot in self.source_slots + self.target_slots):
            raise ValueError("resident token slots must be non-negative")
        if self.source_token_start < 0 or self.target_token_start < 0:
            raise ValueError("resident logical token starts must be non-negative")
        if self.position_mode not in ("rope_1d", "mrope_3d"):
            raise ValueError(
                f"unknown resident position mode: {self.position_mode}"
            )

    @property
    def num_tokens(self) -> int:
        return len(self.source_slots)


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    resident_refresh_specs: list[RefreshSpec] = field(default_factory=list)
    resident_copy_specs: list[ResidentTokenCopySpec] = field(
        default_factory=list
    )

    def get_refresh_specs(self) -> tuple[RefreshSpec, ...]:
        request_specs = tuple(
            request.refresh_spec
            for request in self.requests
            if request.refresh_spec is not None
        )
        return request_specs + tuple(self.resident_refresh_specs)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        # Put the leading with "lmcache." and matched configs from
        # vllm extra_config to the config
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if _validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            f"Updated config {config_key} from vLLM "
                            f"extra config: {value}"
                        )

        self.config = config
        self.use_layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending
        resident_extra = config.extra_config or {}
        self._cacheblend_mode = config.blend_mode == "topk"
        self._cacheblend_event_writeback = bool(
            resident_extra.get("cacheblend_event_writeback", False)
        )
        if self._cacheblend_event_writeback and not self._cacheblend_mode:
            raise ValueError(
                "cacheblend_event_writeback is restricted to blend_mode=topk"
            )
        self._cacheblend_pending_writebacks: dict[str, dict[str, Any]] = {}
        self._cacheblend_waiting_finished_ids: set[str] = set()
        # Resident indexing is a CoStream-only storage prototype.  CacheBlend
        # and VLCache retain their materialized eager-recompute paths even when
        # they share the same base YAML.
        (
            self._resident_shadow_enabled,
            self._resident_zero_copy_enabled,
        ) = _resident_kv_modes(
            bool(config.is_codecsight),
            resident_extra,
            getattr(config, "blend_mode", None),
        )
        self._resident_max_entries = int(
            resident_extra.get("resident_kv_max_entries", 4096)
        )
        if self._resident_max_entries <= 0:
            raise ValueError("resident_kv_max_entries must be positive")
        resident_max_bytes = int(
            resident_extra.get("resident_kv_max_bytes", 0)
        )
        if resident_max_bytes < 0:
            raise ValueError("resident_kv_max_bytes must be non-negative")
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        kv_dtype = get_kv_cache_torch_dtype(
            vllm_config.cache_config.cache_dtype,
            model_config.dtype,
        )
        kv_components = 1 if mla_enabled(model_config) else 2
        self._resident_bytes_per_token_aggregate = (
            model_config.get_num_layers(parallel_config)
            * kv_components
            * model_config.get_num_kv_heads(parallel_config)
            * model_config.get_head_size()
            * torch.empty((), dtype=kv_dtype).element_size()
            * self.worker_count
        )
        resident_block_bytes = (
            self._resident_bytes_per_token_aggregate
            * vllm_config.cache_config.block_size
        )
        self._resident_max_unique_blocks = (
            resident_max_bytes // resident_block_bytes
            if resident_max_bytes else None
        )
        if resident_max_bytes and not self._resident_max_unique_blocks:
            raise ValueError(
                "resident_kv_max_bytes is smaller than one aggregate KV block"
            )
        self._resident_overlap_enabled = bool(
            self._resident_zero_copy_enabled
            and resident_extra.get("resident_kv_overlap_window", False)
        )
        self._resident_allow_approximate_overlap = bool(
            resident_extra.get(
                "resident_kv_allow_approximate_overlap", False
            )
        )
        self._diagnostic_refresh_policy = os.environ.get(
            "COSTREAM_DIAGNOSTIC_REFRESH_POLICY", ""
        )
        if self._diagnostic_refresh_policy:
            from lmcache.integration.vllm.diagnostic_refresh import POLICIES
            if (os.environ.get("COSTREAM_DIAGNOSTIC_REFRESH") != "1"
                    or self._diagnostic_refresh_policy not in POLICIES
                    or not self._resident_overlap_enabled
                    or "internvl" not in str(self._vllm_config.model_config.model).lower()):
                raise ValueError("refresh diagnostic requires explicit opt-in and resident InternVL")
            if os.environ.get("VLLM_INTERNVL_PRUNE") == "1":
                raise ValueError("refresh diagnostic forbids codec pruning")
        if self._resident_overlap_enabled and _configured_refresh_policy(
            resident_extra
        ) != "multi_anchor":
            raise ValueError(
                "resident overlapping-window reuse requires "
                "codecsight_refresh_policy=multi_anchor"
            )
        self._resident_pending_blank_blocks: dict[str, tuple[int, ...]] = {}
        self._resident_pending_refresh_specs: dict[str, RefreshSpec] = {}
        self._resident_pending_copy_specs: dict[
            str, tuple[ResidentTokenCopySpec, ...]
        ] = {}
        self._resident_disable_lmcache_storage = bool(
            self._resident_zero_copy_enabled
            and resident_extra.get(
                "resident_kv_disable_lmcache_storage", False
            )
        )

        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._layerwise_batch_retriever: Optional[
            Generator[Optional[list[torch.Tensor]], None, None]
        ] = None
        self._layerwise_batch_requests: list[ReqMeta] = []
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._log_writeback_timing = (
            os.environ.get("LMCACHE_LOG_WRITEBACK_TIMING", "0") == "1"
        )
        self._writeback_inline_seconds = 0.0
        self._writeback_started_at: Optional[float] = None
        self._writeback_request_ids: list[str] = []
        self._request_profiles: dict[str, dict[str, Any]] = {}
        self._lmcache_local_gpu_peak_actual_used_bytes_per_rank = 0
        self._resident_repack_timing: Optional[
            tuple[torch.cuda.Event, torch.cuda.Event, int]
        ] = None
        self._resident_repack_component_timing: Optional[tuple[
            tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...],
            tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...],
            int,
        ]] = None
        self._resident_slot_positions: Optional[torch.Tensor] = None
        self._resident_rotary_embedding = None
        self._resident_fused_rope = None
        self._kv_diag = KVDiagnostic(
            vllm_config.model_config.get_num_layers(
                vllm_config.parallel_config
            ),
            vllm_config.parallel_config.rank,
        )
        self._kv_diag_force_miss = (
            os.environ.get("LMCACHE_KV_DIAG_FORCE_MISS", "0") == "1"
        )
        self._layerwise_load_requests: list[ReqMeta] = []
        if role == KVConnectorRole.SCHEDULER:
            self.lmcache_engine: Optional[LMCacheEngine] = None
            # Check if bypass lookup is enabled for scheduler
            if config.enable_scheduler_bypass_lookup:
                # Create LMCacheEngine for scheduler when bypass is enabled
                self.lmcache_engine = _init_lmcache_engine(
                    config,
                    vllm_config,
                    role="scheduler",
                )
            # Create lookup client using factory
            self.lookup_client = LookupClientFactory.create_lookup_client(
                vllm_config, config, self.lmcache_engine
            )
            self._unfinished_requests: dict[str, Request] = {}
            self.lmcache_engine = None
        else:
            self.lmcache_engine = _init_lmcache_engine(
                config,
                vllm_config,
                role="worker",
            )

            # Blender is built lazily after model registration.
            self.blender = None

            # Create lookup server using factory
            assert self.lmcache_engine is not None
            self.lookup_server = LookupClientFactory.create_lookup_server(
                self.lmcache_engine, vllm_config
            )

            self.offload_server = ZMQOffloadServer(
                self.lmcache_engine,
                vllm_config,
                get_tensor_model_parallel_rank(),
            )

            # In case of MLA, the lookup server is only created on worker 0
            if self.async_loading and self.lookup_server is not None:
                assert isinstance(self.lookup_server, LMCacheAsyncLookupServer)
                self.lmcache_engine.post_init(async_lookup_server=self.lookup_server)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        # request_id -> (vllm cached tokens, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: Optional[KVCacheManager] = None

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self._save_decode_cache = config.save_decode_cache

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(
            os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False)
            or self._resident_disable_lmcache_storage
        )

        self._requests_priority: dict[str, int] = {}

        # TODO(baoloongmao): Internal api server & plugin framework support dp > 1
        if vllm_config.parallel_config.data_parallel_rank_local == 0:
            # Start internal API server if enabled
            # The enabled check is in the InternalAPIServer constructor
            self.api_server = InternalAPIServer(self)
            self.api_server.start()
            # Launch plugins
            self.plugin_launcher = PluginLauncher(
                self.config,
                role,
                self.worker_count,
                -1
                if self.lmcache_engine is None  # scheduler side
                else self.lmcache_engine.metadata.worker_id,
            )
            self.plugin_launcher.launch_plugins()
        else:
            self.api_server = None  # type: ignore[assignment]
            self.plugin_launcher = None  # type: ignore[assignment]
        logger.info(
            f"LMCache initialized for role {role} with version {utils.get_version()}, "
            f"vllm version {VLLM_VERSION}, "
            "lmcache cache_engine metadata: "
            f"{getattr(self.lmcache_engine, 'metadata', None)}"
        )

    def _resident_registry(self) -> Optional[ResidentKVRegistry]:
        if not (
            self._resident_shadow_enabled or self._resident_zero_copy_enabled
        ):
            return None
        registry = get_default_resident_kv_registry()
        if registry is not None and registry.max_entries is None:
            registry.max_entries = self._resident_max_entries
        if (registry is not None
                and registry.max_unique_blocks is None
                and getattr(
                    self, "_resident_max_unique_blocks", None
                ) is not None):
            registry.max_unique_blocks = self._resident_max_unique_blocks
        return registry

    @staticmethod
    def _merge_refresh_block_indices(
        block_indices: list[int],
        block_size: int,
    ) -> tuple[RefreshSpan, ...]:
        if not block_indices:
            return ()
        spans: list[RefreshSpan] = []
        run_start = previous = block_indices[0]
        for block_index in block_indices[1:]:
            if block_index != previous + 1:
                spans.append(RefreshSpan(
                    run_start * block_size,
                    (previous + 1) * block_size,
                ))
                run_start = block_index
            previous = block_index
        spans.append(RefreshSpan(
            run_start * block_size,
            (previous + 1) * block_size,
        ))
        return tuple(spans)

    def _clear_pending_resident_adoption(
        self,
        request_id: str,
        *,
        release_blocks: bool,
    ) -> None:
        blank_blocks = getattr(
            self, "_resident_pending_blank_blocks", None
        )
        refresh_specs = getattr(
            self, "_resident_pending_refresh_specs", None
        )
        copy_specs = getattr(self, "_resident_pending_copy_specs", None)
        block_ids = (blank_blocks.pop(request_id, ())
                     if blank_blocks is not None else ())
        if refresh_specs is not None:
            refresh_specs.pop(request_id, None)
        if copy_specs is not None:
            copy_specs.pop(request_id, None)
        if release_blocks and block_ids:
            registry = self._resident_registry()
            if registry is not None:
                registry.block_pool.free_blocks(
                    registry.block_pool.get_blocks_by_id(block_ids)
                )

    def _get_overlapping_resident_prefix(
        self,
        request: "Request",
        mm_hashes: list[str],
        mm_positions: list["PlaceholderRange"],
    ) -> tuple[Optional[tuple[list[int], ...]], int]:
        """Compose a resident prefix from overlap hits and refresh blocks."""
        if not getattr(self, "_resident_overlap_enabled", False):
            return None, 0
        decisions, decision_error = _extract_costream_frame_decisions(
            request, mm_hashes
        )
        profile = request.system_profile
        if decisions is None:
            profile["resident_overlap_fallback_reason"] = (
                decision_error or "missing_frame_decisions"
            )
            return None, 0

        prefix_tokens = (
            (len(request.prompt_token_ids) - 1) // self._block_size
        ) * self._block_size
        if prefix_tokens <= 0:
            return None, 0
        required_blocks = prefix_tokens // self._block_size
        synthetic_blocks = list(range(1, required_blocks + 1))
        records, _ = _resident_frame_token_records(
            mm_hashes,
            mm_positions,
            synthetic_blocks,
            self._block_size,
            request.prompt_token_ids,
            decisions,
        )
        registry = self._resident_registry()
        if registry is None:
            return None, 0

        stream_id = _costream_stream_id(request)
        lora_request = getattr(request, "lora_request", None)
        text_record = _resident_text_prefix_record(
            request.prompt_token_ids,
            mm_positions,
            self._block_size,
            cache_salt=getattr(request, "cache_salt", None),
            lora_id=int(getattr(lora_request, "lora_int_id", 0) or 0),
            stream_id=stream_id,
        )
        text_entry = (
            registry.lookup(text_record["key"])
            if text_record is not None else None
        )
        text_prefix_tokens = (
            int(text_record["token_length"])
            if text_record is not None else 0
        )
        text_prefix_hit = bool(
            text_record is not None
            and text_entry is not None
            and text_entry.token_start == 0
            and text_entry.token_length == text_record["token_length"]
            and text_entry.position_fingerprint
            == text_record["position_fingerprint"]
            and text_entry.context_hash == text_record["context_hash"]
        )
        architectures = (
            getattr(
                self._vllm_config.model_config.hf_config,
                "architectures",
                [],
            )
            or []
        )
        position_mode = (
            "mrope_3d"
            if any("Qwen3VL" in name for name in architectures)
            else "rope_1d"
        )

        reusable: list[tuple[dict[str, Any], Any, bool]] = []
        reused_frames = 0
        reused_tokens = 0
        relocated_frames = 0
        context_mismatch_frames = 0
        approximate_frames = 0
        for record in records:
            decision = record["frame_decision"]
            if (decision.is_anchor or decision.action == "refresh") and not getattr(
                self, "_diagnostic_refresh_policy", ""
            ):
                continue
            entry = registry.lookup(record["key"])
            if entry is None:
                continue
            source_offset = int(entry.metadata.get(
                "source_token_offset", -1
            ))
            source_capacity = (
                len(entry.block_ids) * self._block_size - source_offset
            )
            if source_offset < 0 or source_capacity < record["token_length"]:
                continue
            position_matches = (
                entry.position_fingerprint == record["position_fingerprint"]
            )
            context_matches = (
                entry.context_hash is not None
                and entry.context_hash == record["context_hash"]
            )
            if not position_matches:
                relocated_frames += 1
            if not context_matches:
                context_mismatch_frames += 1
            approximate = not (position_matches and context_matches)
            if approximate and not getattr(
                self, "_resident_allow_approximate_overlap", False
            ):
                continue
            reusable.append((record, entry, approximate))
            reused_frames += 1
            reused_tokens += int(record["token_length"])
            approximate_frames += int(approximate)

        if reused_tokens == 0:
            profile["resident_overlap_fallback_reason"] = (
                "no_reusable_overlap_blocks"
            )
            return None, 0

        try:
            blank_blocks = registry.block_pool.get_new_blocks(
                required_blocks
            )
        except ValueError:
            profile["resident_overlap_fallback_reason"] = (
                "insufficient_blocks_for_refresh"
            )
            return None, 0
        blank_ids = tuple(block.block_id for block in blank_blocks)
        copied = bytearray(prefix_tokens)
        copy_specs: list[ResidentTokenCopySpec] = []
        text_prefix_reused_tokens = 0
        if text_prefix_hit and text_record is not None and text_entry is not None:
            length = min(text_prefix_tokens, prefix_tokens)
            if length > 0:
                source_offset = int(text_entry.metadata.get(
                    "source_token_offset", 0
                ))
                source_slots = tuple(
                    text_entry.block_ids[
                        (source_offset + index) // self._block_size
                    ] * self._block_size
                    + (source_offset + index) % self._block_size
                    for index in range(length)
                )
                target_slots = tuple(
                    blank_ids[index // self._block_size] * self._block_size
                    + index % self._block_size
                    for index in range(length)
                )
                copy_specs.append(ResidentTokenCopySpec(
                    request_id=request.request_id,
                    source_slots=source_slots,
                    target_slots=target_slots,
                    source_token_start=0,
                    target_token_start=0,
                    position_mode=position_mode,
                    repair_positions=False,
                ))
                copied[:length] = b"\x01" * length
                text_prefix_reused_tokens = length
        position_repair_tokens = 0
        for record, entry, approximate in reusable:
            target_start = int(record["token_start"])
            length = min(
                int(record["token_length"]), prefix_tokens - target_start
            )
            if length <= 0:
                continue
            source_offset = int(entry.metadata["source_token_offset"])
            source_slots = tuple(
                entry.block_ids[(source_offset + index) // self._block_size]
                * self._block_size
                + (source_offset + index) % self._block_size
                for index in range(length)
            )
            target_slots = tuple(
                blank_ids[(target_start + index) // self._block_size]
                * self._block_size
                + (target_start + index) % self._block_size
                for index in range(length)
            )
            copy_specs.append(ResidentTokenCopySpec(
                request_id=request.request_id,
                source_slots=source_slots,
                target_slots=target_slots,
                source_token_start=int(entry.token_start),
                target_token_start=target_start,
                position_mode=position_mode,
                repair_positions=approximate,
            ))
            if approximate:
                position_repair_tokens += length
            copied[target_start:target_start + length] = b"\x01" * length

        diagnostic_policy = getattr(self, "_diagnostic_refresh_policy", "")
        if diagnostic_policy:
            from lmcache.integration.vllm.diagnostic_refresh import apply_control
            diagnostic_stats = apply_control(
                copied, mm_positions, decisions, diagnostic_policy,
                int(os.environ.get("COSTREAM_DIAGNOSTIC_REFRESH_SEED", "1701")),
            )
            profile.update(diagnostic_stats)
            if os.environ.get("COSTREAM_DIAGNOSTIC_AUDIT", "1") == "1":
                logger.info("REFRESH_DIAGNOSTIC %s", diagnostic_stats)
        spans_list: list[RefreshSpan] = []
        cursor = 0
        while cursor < prefix_tokens:
            if copied[cursor]:
                cursor += 1
                continue
            start = cursor
            while cursor < prefix_tokens and not copied[cursor]:
                cursor += 1
            spans_list.append(RefreshSpan(start, cursor))
        spans = tuple(spans_list)
        if spans:
            # Joint refresh bypasses the scheduler's encoder cache. Explicitly
            # retain these embeddings for the next window's GOP-anchor refresh.
            if diagnostic_policy in (
                "random",
                "shifted_anchor",
                "center_contiguous",
                "full_refresh",
            ):
                resident_encoder_hashes = tuple(mm_hashes)
            elif diagnostic_policy == "rope_fix_reuse":
                resident_encoder_hashes = ()
            else:
                resident_encoder_hashes = tuple(
                    mm_hash
                    for mm_hash, decision in zip(
                        mm_hashes, decisions, strict=True
                    )
                    if decision.is_anchor or decision.action == "refresh"
                )
            self._resident_pending_refresh_specs[request.request_id] = (
                RefreshSpec(
                    request_id=request.request_id,
                    model_id=self._vllm_config.model_config.model,
                    cache_schema_version=str(
                        (self.config.extra_config or {}).get(
                            "cache_schema_version", "costream-resident-v1"
                        )
                    ),
                    policy="multi_anchor",
                    position_mode=position_mode,
                    cached_prefix_tokens=prefix_tokens,
                    expected_retrieved_tokens=prefix_tokens,
                    spans=spans,
                    source_hashes=tuple(mm_hashes),
                    resident_encoder_hashes=resident_encoder_hashes,
                )
            )
        self._resident_pending_blank_blocks[request.request_id] = blank_ids
        self._resident_pending_copy_specs[request.request_id] = tuple(
            copy_specs
        )
        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=prefix_tokens,
            lmcache_cached_tokens=prefix_tokens,
            can_load=False,
        )
        profile.update({
            "requested_path": "resident_overlap_multi_anchor",
            "executed_path": "resident_overlap_multi_anchor",
            "resident_exact_prefix_hit": False,
            "resident_overlap_hit": True,
            "resident_overlap_reused_frames": reused_frames,
            "resident_overlap_reused_visual_tokens": reused_tokens,
            "resident_overlap_reused_tokens": (
                reused_tokens + text_prefix_reused_tokens
            ),
            "resident_text_prefix_candidate_tokens": text_prefix_tokens,
            "resident_text_prefix_hit": text_prefix_hit,
            "resident_text_prefix_reused_tokens": text_prefix_reused_tokens,
            "resident_text_prefix_recomputed_tokens": (
                text_prefix_tokens - text_prefix_reused_tokens
            ),
            "resident_overlap_repack_tokens": sum(
                spec.num_tokens for spec in copy_specs
            ),
            "resident_overlap_refresh_tokens": sum(
                span.num_tokens for span in spans
            ),
            "resident_overlap_refresh_spans": [
                [span.start, span.end] for span in spans
            ],
            "resident_overlap_relocated_frames": relocated_frames,
            "resident_overlap_context_mismatch_frames": (
                context_mismatch_frames
            ),
            "resident_overlap_approximate_frames": approximate_frames,
            "resident_position_mode": position_mode,
            "resident_position_repair_expected_tokens": (
                position_repair_tokens
            ),
            "resident_overlap_boundary_policy": "token_slice_repack",
            "resident_candidate_kv_tokens": prefix_tokens,
            "vllm_reused_kv_tokens": (
                reused_tokens + text_prefix_reused_tokens
            ),
            "lmcache_reused_kv_tokens": 0,
            "loaded_kv_tokens": 0,
            "kv_load_bytes": 0,
            "kv_materialization_mode": "vllm_resident_token_repack",
            "fallback_reason": None,
            "silent_fallback": False,
        })
        return (list(blank_ids),), prefix_tokens

    def _probe_resident_frames(
        self,
        request: "Request",
        mm_hashes: list[str],
        mm_positions: list["PlaceholderRange"],
    ) -> None:
        registry = self._resident_registry()
        if registry is None:
            return
        # Probe alignment and content identity before LMCache lookup.  A
        # synthetic block list is sufficient because this side only needs the
        # logical ranges; no ownership is acquired in shadow mode.
        num_blocks = cdiv(len(request.prompt_token_ids), self._block_size)
        synthetic_blocks = list(range(1, num_blocks + 1))
        decisions, decision_error = _extract_costream_frame_decisions(
            request, mm_hashes
        )
        records, rejected = _resident_frame_block_records(
            mm_hashes,
            mm_positions,
            synthetic_blocks,
            self._block_size,
            request.prompt_token_ids,
            decisions,
        )
        hits = 0
        zero_copy = 0
        cow = 0
        relocated = 0
        context_mismatch = 0
        exact_context = 0
        exact_context_tokens = 0
        hit_tokens = 0
        for record in records:
            entry = registry.lookup(record["key"])
            if entry is None:
                continue
            hits += 1
            hit_tokens += int(record["token_length"])
            position_matches = (
                entry.position_fingerprint == record["position_fingerprint"]
            )
            context_matches = (
                entry.context_hash is not None
                and entry.context_hash == record["context_hash"]
            )
            if not position_matches:
                relocated += 1
            if not context_matches:
                context_mismatch += 1
            if not (position_matches and context_matches):
                continue
            exact_context += 1
            exact_context_tokens += int(record["token_length"])
            mode = registry.inspect_write_mode(record["key"])
            if mode == "zero_copy":
                zero_copy += 1
            elif mode == "copy_on_write":
                cow += 1
        profile = request.system_profile
        profile["resident_kv_mode"] = (
            "zero_copy_prototype"
            if self._resident_zero_copy_enabled
            else "shadow"
        )
        profile["resident_shadow_candidate_frames"] = len(mm_hashes)
        profile["costream_frame_decisions_present"] = decisions is not None
        profile["costream_frame_decision_error"] = decision_error
        profile["costream_anchor_frames"] = sum(
            decision.is_anchor for decision in (decisions or ())
        )
        profile["costream_pruned_frames"] = sum(
            decision.action == "prune" for decision in (decisions or ())
        )
        profile["costream_padding_frames"] = (
            _costream_padding_frame_count(request)
        )
        profile["resident_shadow_aligned_frames"] = len(records)
        profile["resident_shadow_unaligned_frames"] = rejected
        profile["resident_shadow_hit_frames"] = hits
        profile["resident_shadow_hit_tokens"] = hit_tokens
        profile["resident_position_relocation_frames"] = relocated
        profile["resident_context_mismatch_frames"] = context_mismatch
        profile["resident_exact_context_frames"] = exact_context
        profile["resident_exact_context_hit_tokens"] = exact_context_tokens
        profile["resident_refresh_required_frames"] = hits - exact_context
        profile["resident_zero_copy_eligible_frames"] = zero_copy
        profile["resident_cow_required_frames"] = cow

    def _publish_resident_frames(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[int, int]:
        registry = self._resident_registry()
        if registry is None:
            return 0, 0
        mm_hashes, mm_positions = extract_mm_features(request)
        decisions, decision_error = _extract_costream_frame_decisions(
            request, mm_hashes
        )
        overlap_enabled = bool(getattr(
            self, "_resident_overlap_enabled", False
        ))
        record_builder = (
            _resident_frame_token_records
            if overlap_enabled else _resident_frame_block_records
        )
        records, rejected = record_builder(
            mm_hashes,
            mm_positions,
            block_ids,
            self._block_size,
            request.prompt_token_ids,
            decisions,
        )
        active_frame_hashes = [
            record["content_hash"] for record in records
        ]
        stream_id = _costream_stream_id(request)
        retired_frames = registry.retain_kind_content_hashes(
            "frame_token_slice" if overlap_enabled else "frame_interior_blocks",
            active_frame_hashes,
            stream_id=stream_id,
        )
        for record in records:
            registry.publish(
                record["key"],
                record["block_ids"],
                token_start=record["token_start"],
                token_length=record["token_length"],
                content_hash=record["content_hash"],
                position_fingerprint=record["position_fingerprint"],
                context_hash=record["context_hash"],
                persistent=overlap_enabled,
                metadata={
                    "kind": (
                        "frame_token_slice"
                        if overlap_enabled else "frame_interior_blocks"
                    ),
                    "frame_index": record["frame_index"],
                    "frame_decision": record["frame_decision"],
                    "stream_id": stream_id,
                    **({
                        "source_token_offset": record[
                            "source_token_offset"
                        ],
                    } if overlap_enabled else {
                        "frame_relative_start": record[
                            "frame_relative_start"
                        ],
                    }),
                },
            )
        stats = registry.stats()
        profile = request.system_profile
        profile["resident_published_frames"] = len(records)
        profile["resident_retired_window_frames"] = (
            int(profile.get("resident_retired_window_frames", 0))
            + retired_frames
        )
        profile["costream_frame_decisions_present"] = decisions is not None
        profile["costream_frame_decision_error"] = decision_error
        profile["resident_registry_entries"] = stats["entries"]
        profile["resident_registry_block_references"] = stats[
            "block_references"
        ]
        profile["resident_registry_unique_block_references"] = stats[
            "unique_block_references"
        ]
        profile["resident_registry_unique_token_capacity"] = (
            stats["unique_block_references"] * self._block_size
        )
        profile["resident_registry_unique_kv_bytes_aggregate"] = (
            stats["unique_block_references"]
            * self._block_size
            * int(getattr(
                self, "_resident_bytes_per_token_aggregate", 0
            ))
        )
        profile["resident_registry_max_unique_blocks"] = stats[
            "max_unique_blocks"
        ]
        profile["resident_registry_watermark_evictions"] = stats[
            "watermark_evicted"
        ]
        profile["resident_registry_pinned_block_references"] = stats[
            "persistent_block_references"
        ]
        profile["resident_registry_index_payload_bytes_estimate"] = stats[
            "index_payload_bytes_estimate"
        ]
        return len(records), rejected

    def _publish_resident_prompt(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> int:
        if not self._resident_zero_copy_enabled:
            return 0
        registry = self._resident_registry()
        if registry is None:
            return 0
        mm_hashes, mm_positions = extract_mm_features(request)
        lora_request = getattr(request, "lora_request", None)
        record = _resident_prompt_block_record(
            request.prompt_token_ids,
            mm_hashes,
            mm_positions,
            self._block_size,
            block_ids=block_ids,
            cache_salt=getattr(request, "cache_salt", None),
            lora_id=int(getattr(lora_request, "lora_int_id", 0) or 0),
        )
        if record is None:
            return 0
        if int(getattr(request, "num_computed_tokens", 0)) < record[
            "token_length"
        ]:
            return 0
        registry.publish(
            record["key"],
            record["block_ids"],
            token_start=record["token_start"],
            token_length=record["token_length"],
            content_hash=record["content_hash"],
            position_fingerprint=record["position_fingerprint"],
            context_hash=record["context_hash"],
            persistent=True,
            metadata={
                "kind": "exact_prompt_prefix",
                "stream_id": _costream_stream_id(request),
            },
            replace_existing_kind=True,
        )
        request.system_profile["resident_published_prefix_tokens"] = record[
            "token_length"
        ]
        return int(record["token_length"])

    def _publish_resident_text_prefix(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> int:
        if not (
            self._resident_zero_copy_enabled
            and self._resident_overlap_enabled
        ):
            return 0
        registry = self._resident_registry()
        if registry is None:
            return 0
        mm_hashes, mm_positions = extract_mm_features(request)
        del mm_hashes
        lora_request = getattr(request, "lora_request", None)
        stream_id = _costream_stream_id(request)
        record = _resident_text_prefix_record(
            request.prompt_token_ids,
            mm_positions,
            self._block_size,
            block_ids=block_ids,
            cache_salt=getattr(request, "cache_salt", None),
            lora_id=int(getattr(lora_request, "lora_int_id", 0) or 0),
            stream_id=stream_id,
        )
        if record is None:
            return 0
        if int(getattr(request, "num_computed_tokens", 0)) < record[
            "token_length"
        ]:
            return 0
        registry.publish(
            record["key"],
            record["block_ids"],
            token_start=record["token_start"],
            token_length=record["token_length"],
            content_hash=record["content_hash"],
            position_fingerprint=record["position_fingerprint"],
            context_hash=record["context_hash"],
            persistent=True,
            metadata={
                "kind": "exact_text_prefix",
                "stream_id": stream_id,
                "source_token_offset": 0,
            },
            replace_existing_kind=True,
        )
        request.system_profile["resident_published_text_prefix_tokens"] = (
            record["token_length"]
        )
        return int(record["token_length"])

    def _ensure_blender_initialized(self):
        """
        Lazily build the blender once the vLLM model has been registered.
        If the model is unavailable, skip blending for this round instead of
        failing startup.
        """
        if not self.enable_blending or self.blender is not None:
            return

        try:
            _ = VLLMModelTracker.get_model(ENGINE_NAME)
        except Exception as exc:
            logger.warning(
                "Blending requested but vLLM model not registered yet: %s", exc
            )
            return

        assert self.lmcache_engine.gpu_connector is not None, (
            "GPU connector must be available for blending"
        )
        self.blender = LMCBlenderBuilder.get_or_create(
            ENGINE_NAME,
            self.lmcache_engine,
            self.lmcache_engine.gpu_connector,
            self.config,
        )

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    ####################
    # Worker side APIs
    ####################

    @staticmethod
    def _scatter_vision_embeds(
        text_embeds: torch.Tensor,
        vision_embeds: list[Optional[torch.Tensor]],
        mm_positions: list,
        num_tokens: int,
    ) -> torch.Tensor:
        """Overlay vision embeddings onto text embeddings at mm_positions.

        Handles NaN rows from ``scatter_mm_placeholders`` (structural tokens
        like ``<img>``/``</img>``) by preserving the text embedding there.
        ``None`` entries in *vision_embeds* (encoder_cache misses) are
        skipped, leaving the text embedding in place at that position.
        """
        inputs_embeds = text_embeds.clone()
        ve_idx = 0
        merged = 0
        for placeholder in mm_positions:
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= num_tokens:
                continue
            end = min(start + length, num_tokens)
            if ve_idx >= len(vision_embeds):
                break
            ve = vision_embeds[ve_idx]
            ve_idx += 1
            if ve is None:
                continue
            actual_len = end - start
            ve_slice = ve[:actual_len].to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            valid_mask = ~torch.isnan(ve_slice).any(dim=-1)
            if merged == 0:
                nan_count = int((~valid_mask).sum().item())
                logger.debug(
                    "vision_embed[0]: shape=%s, nan_rows=%d/%d, dtype=%s",
                    ve.shape, nan_count, actual_len, ve.dtype)
            if ve_slice.shape[0] >= actual_len:
                if valid_mask.all():
                    inputs_embeds[start:end] = ve_slice
                else:
                    inputs_embeds[start:end][valid_mask] = \
                        ve_slice[valid_mask]
            else:
                sub_len = ve_slice.shape[0]
                sub_mask = valid_mask[:sub_len]
                if sub_mask.any():
                    inputs_embeds[start:start + sub_len][sub_mask] = \
                        ve_slice[sub_mask]
            merged += 1
        final_has_nan = bool(torch.isnan(inputs_embeds).any())
        logger.debug(
            "Reconstructed inputs_embeds: shape=%s, "
            "vision_embeds_merged=%d/%d, num_tokens=%d, has_nan=%s",
            inputs_embeds.shape, merged, len(vision_embeds), num_tokens,
            final_has_nan,
        )
        return inputs_embeds

    @staticmethod
    def _normalize_cached_mm_embed(
        embed: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Normalize cached multimodal embed to [tokens, dim] then crop tokens."""
        if embed.ndim == 1:
            embed = embed.unsqueeze(0)
        elif embed.ndim > 2:
            embed = embed.reshape(-1, embed.shape[-1])
        return embed[:length]

    def _reconstruct_inputs_embeds(
        self,
        token_ids: list[int],
        mm_hashes: Optional[list[str]],
        mm_positions: Optional[list["PlaceholderRange"]],
        num_tokens: int,
        request_id: Optional[str] = None,
    ) -> EmbeddingReconstructionResult:
        """Rebuild a cached prefix's multimodal input embeddings."""
        if not mm_hashes or not mm_positions:
            return EmbeddingReconstructionResult(
                None, None, "no_visual_prefix",
                "request has no multimodal metadata",
            )

        visual_items = []
        for mm_hash, placeholder in zip(mm_hashes, mm_positions):
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= num_tokens:
                continue
            visual_items.append((mm_hash, placeholder))
        if not visual_items:
            return EmbeddingReconstructionResult(
                None, None, "no_visual_prefix",
                f"cached prefix of {num_tokens} tokens contains no visual span",
            )

        try:
            vllm_model = VLLMModelTracker.get_model(ENGINE_NAME)
        except (ValueError, KeyError) as exc:
            return EmbeddingReconstructionResult(
                None, None, "model_unavailable", str(exc),
            )

        encoder_cache = VLLMModelTracker.get_encoder_cache(ENGINE_NAME)
        if encoder_cache is None:
            logger.warning(
                "encoder_cache not registered; vision token recompute disabled"
            )
            return EmbeddingReconstructionResult(
                None, None, "encoder_cache_unavailable",
                "encoder_cache is not registered",
            )

        token_ids_t = torch.tensor(
            token_ids[:num_tokens], dtype=torch.long, device="cuda"
        )
        # Selective refresh requires embeddings for the entire cached prefix.
        if token_ids_t.shape[0] < num_tokens:
            detail = (
                f"cached prefix incomplete: have {int(token_ids_t.shape[0])} "
                f"of {num_tokens} expected tokens"
            )
            return EmbeddingReconstructionResult(
                None, None, "incomplete_prefix", detail,
            )

        # Encoder embeddings may be evicted while decoder KV remains cached.
        # Re-encode missing items transiently and include that work in latency.
        transient_encoder_outputs: Mapping[str, Any] = {}
        missing_visual_items = [
            (mm_hash, placeholder)
            for mm_hash, placeholder in visual_items
            if encoder_cache.get(mm_hash) is None
        ]
        if missing_visual_items and request_id is not None:
            recompute = VLLMModelTracker.get_encoder_recompute_callback(
                ENGINE_NAME
            )
            if recompute is not None:
                try:
                    transient_encoder_outputs = recompute(
                        request_id,
                        [item[0] for item in missing_visual_items],
                        [item[1] for item in missing_visual_items],
                        num_tokens,
                    )
                except Exception as exc:
                    return EmbeddingReconstructionResult(
                        None,
                        None,
                        "encoder_recompute_failure",
                        str(exc),
                    )

        # Preserve one entry per visual span, including cache misses.
        vision_embeds: list[Optional[torch.Tensor]] = []
        num_encoder_misses = 0
        from vllm.v1.worker.utils import gather_mm_placeholders

        for mm_hash, placeholder in visual_items:
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= num_tokens:
                continue
            end = min(start + length, num_tokens)
            enc_out = encoder_cache.get(mm_hash)
            if enc_out is None:
                enc_out = transient_encoder_outputs.get(mm_hash)
            if enc_out is None:
                logger.debug("encoder_cache miss: hash=%s", mm_hash)
                vision_embeds.append(None)
                num_encoder_misses += 1
                continue

            if isinstance(enc_out, torch.Tensor):
                enc_slice = self._normalize_cached_mm_embed(
                    enc_out, end - start)
            else:
                enc_slice = self._normalize_cached_mm_embed(
                    torch.as_tensor(enc_out), end - start)
            is_embed = getattr(placeholder, "is_embed", None)
            if is_embed is not None:
                is_embed = torch.as_tensor(
                    is_embed[:end - start],
                    dtype=torch.bool,
                    device=enc_slice.device,
                )
            enc_slice = gather_mm_placeholders(enc_slice, is_embed)
            vision_embeds.append(enc_slice)
        if num_encoder_misses > 0:
            return EmbeddingReconstructionResult(
                None,
                None,
                "encoder_cache_miss",
                f"missed {num_encoder_misses}/{len(vision_embeds)} visual items",
            )

        if not any(ve is not None for ve in vision_embeds):
            return EmbeddingReconstructionResult(
                None, None, "encoder_cache_miss",
                "no visual embeddings were found for visual spans in prefix",
            )

        prepare_refresh = getattr(
            vllm_model, "prepare_kv_refresh_inputs", None
        )
        if callable(prepare_refresh):
            try:
                inputs_embeds, deepstack_input_embeds = prepare_refresh(
                    token_ids_t,
                    tuple(ve for ve in vision_embeds if ve is not None),
                )
            except Exception as exc:
                return EmbeddingReconstructionResult(
                    None, None, "model_refresh_failure", str(exc),
                )
            return EmbeddingReconstructionResult(
                inputs_embeds, deepstack_input_embeds, "ready",
            )

        # Content-hash sentinels make mm_positions the authoritative layout.
        lang_model = getattr(vllm_model, "language_model", vllm_model)
        embed_fn = getattr(lang_model, "get_input_embeddings", None)
        if embed_fn is None:
            embed_fn = getattr(lang_model, "embed_tokens", None)
        if embed_fn is None:
            return EmbeddingReconstructionResult(
                None, None, "embedding_api_unavailable",
                "model exposes neither get_input_embeddings nor embed_tokens",
            )

        text_embeds = embed_fn(token_ids_t)
        text_has_nan = bool(torch.isnan(text_embeds).any())
        logger.debug(
            "text_embeds: shape=%s, has_nan=%s, norm=%.4f, "
            "token_ids min=%d max=%d",
            text_embeds.shape, text_has_nan,
            text_embeds.norm().item() if not text_has_nan else float('nan'),
            token_ids_t.min().item(), token_ids_t.max().item(),
        )

        deepstack_input_embeds: Optional[torch.Tensor] = None
        use_deepstack = getattr(vllm_model, "use_deepstack", False)
        visual_dim = int(getattr(vllm_model, "visual_dim", text_embeds.shape[-1]))
        multiscale_dim = int(getattr(vllm_model, "multiscale_dim", 0))
        expected_mm_dim = visual_dim + multiscale_dim

        # Codec metadata may append four mRoPE channels.
        vision_embeds_norm: list[Optional[torch.Tensor]] = []
        for ve in vision_embeds:
            if ve is None:
                vision_embeds_norm.append(None)
                continue
            if ve.ndim == 1:
                ve = ve.unsqueeze(0)
            if ve.ndim > 2:
                ve = ve.reshape(-1, ve.shape[-1])

            last_dim = ve.shape[-1]
            if expected_mm_dim > 0 and last_dim == expected_mm_dim + 4:
                ve = ve[:, :-4]
                last_dim = ve.shape[-1]
            vision_embeds_norm.append(ve)

        vision_embeds = vision_embeds_norm
        if use_deepstack:
            return EmbeddingReconstructionResult(
                None,
                None,
                "embedding_api_unavailable",
                "DeepStack refresh requires prepare_kv_refresh_inputs",
            )

        # Non-DeepStack models consume only the language-width feature slice.
        hidden = text_embeds.shape[-1]
        vision_embeds_scatter: list[Optional[torch.Tensor]] = []
        for idx, ve in enumerate(vision_embeds):
            if ve is None:
                vision_embeds_scatter.append(None)
                continue
            if ve.shape[-1] == hidden:
                vision_embeds_scatter.append(ve)
                continue
            if expected_mm_dim > 0 and ve.shape[-1] == expected_mm_dim:
                vision_embeds_scatter.append(ve[:, :hidden])
                continue
            return EmbeddingReconstructionResult(
                None,
                None,
                "embedding_shape_mismatch",
                f"visual item {idx} has dim {ve.shape[-1]}, expected {hidden}",
            )

        inputs_embeds = self._scatter_vision_embeds(
            text_embeds, vision_embeds_scatter, mm_positions, num_tokens,
        )
        return EmbeddingReconstructionResult(
            inputs_embeds, deepstack_input_embeds, "ready",
        )

    def _compute_request_cache_positions(
        self,
        request: ReqMeta,
        num_tokens: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return exact three-axis positions for an mRoPE cache hit."""
        self._ensure_blender_initialized()
        if self.blender is None or not self.blender.is_mrope:
            return None

        visual_items = []
        for mm_hash, placeholder in zip(
            request.mm_hashes or [], request.mm_positions or [], strict=False,
        ):
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length > 0 and start < num_tokens:
                visual_items.append((mm_hash, start, min(length, num_tokens - start)))

        if not visual_items:
            return torch.arange(
                num_tokens, device=device, dtype=torch.int64,
            ).view(1, -1).expand(3, -1)

        try:
            vllm_model = VLLMModelTracker.get_model(ENGINE_NAME)
        except (ValueError, KeyError):
            vllm_model = None
        encoder_cache = VLLMModelTracker.get_encoder_cache(ENGINE_NAME)
        encoder_position_cache = (
            VLLMModelTracker.get_encoder_position_cache(ENGINE_NAME) or {}
        )
        recompute = getattr(vllm_model, "recompute_mrope_positions", None)

        if encoder_cache is not None and recompute is not None:
            visual_dim = int(getattr(vllm_model, "visual_dim", 0))
            multiscale_dim = int(getattr(vllm_model, "multiscale_dim", 0))
            expected_dim = visual_dim + multiscale_dim
            cached_embeds = []
            for mm_hash, _, length in visual_items:
                enc_out = encoder_cache.get(mm_hash)
                if enc_out is None:
                    cached_embeds = []
                    break
                embed = self._normalize_cached_mm_embed(
                    enc_out if isinstance(enc_out, torch.Tensor)
                    else torch.as_tensor(enc_out),
                    length,
                )
                if expected_dim <= 0 or embed.shape[-1] != expected_dim + 4:
                    cached_embeds = []
                    break
                cached_embeds.append(embed.to(device=device))

            if len(cached_embeds) == len(visual_items):
                base_positions = torch.arange(
                    num_tokens, device=device, dtype=torch.int64,
                ).view(1, -1).expand(3, -1).clone()
                _, positions, _ = recompute(
                    list(request.model_token_ids[:num_tokens]),
                    tuple(cached_embeds),
                    base_positions,
                    0,
                )
                if positions.shape == (3, num_tokens):
                    return positions.to(device=device, dtype=torch.int64)
                raise RuntimeError(
                    "Qwen cached mRoPE metadata produced positions with "
                    f"shape {tuple(positions.shape)}, expected (3, {num_tokens})"
                )

        cached_positions = []
        for mm_hash, _, length in visual_items:
            value = encoder_position_cache.get(mm_hash)
            if value is None:
                cached_positions = []
                break
            value = self._normalize_cached_mm_embed(
                value if isinstance(value, torch.Tensor)
                else torch.as_tensor(value),
                length,
            )
            if value.shape != (length, 4):
                cached_positions = []
                break
            cached_positions.append(value.to(device=device).permute(1, 0))

        if len(cached_positions) == len(visual_items):
            # First Party
            from vllm.multimodal.evs import recompute_mrope_positions

            cfg = self.blender._mrope_model_config
            if cfg is None:
                raise RuntimeError("Qwen M-RoPE model metadata is missing")
            base_positions = torch.arange(
                num_tokens, device=device, dtype=torch.int64,
            ).view(1, -1).expand(3, -1).clone()
            positions, _ = recompute_mrope_positions(
                torch.as_tensor(
                    request.model_token_ids[:num_tokens],
                    device=device, dtype=torch.long,
                ),
                cached_positions,
                base_positions,
                0,
                cfg["vision_start_token_id"],
                cfg["image_token_id"],
                cfg["video_token_id"],
            )
            if positions.shape == (3, num_tokens):
                return positions.to(device=device, dtype=torch.int64)
            raise RuntimeError(
                "Qwen position-only cache produced positions with "
                f"shape {tuple(positions.shape)}, expected (3, {num_tokens})"
            )

        # First Party
        from lmcache.v1.compute.blend.metadata import LMCBlendMetadata

        md = LMCBlendMetadata(
            imp_indices=None, attn_mask=None, positions=None,
        )
        md.input_ids = list(request.model_token_ids[:num_tokens])
        md.mm_positions = request.mm_positions
        md.image_grid_thw = request.image_grid_thw
        previous_md = self.blender._active_metadata
        try:
            self.blender._active_metadata = md
            positions = self.blender._compute_mrope_positions(
                num_tokens, device,
            )
        finally:
            self.blender._active_metadata = previous_md

        if positions.ndim != 2 or positions.shape[0] != 3:
            raise RuntimeError(
                "Qwen mRoPE cache reuse requires exact [3, num_tokens] "
                "positions; refusing the unsafe 1D fallback."
            )
        if positions.shape[1] != num_tokens:
            raise RuntimeError(
                "Qwen mRoPE cache positions do not match the scheduled "
                f"multimodal chunk ({positions.shape[1]} != {num_tokens})"
            )
        return positions

    def _start_checked_layerwise_retrieval(
        self,
        *,
        request_id: str,
        tokens: list[int],
        mask: torch.Tensor,
        kvcaches,
        slot_mapping: torch.Tensor,
        sync: bool,
        cache_positions: Optional[torch.Tensor],
        request_configs: Optional[dict],
        protected_prefix_tokens: int,
        path: str,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """Prime layerwise retrieval and verify the scheduler's promise once."""
        assert self.lmcache_engine is not None
        retriever = self.lmcache_engine.retrieve_layer(
            tokens,
            mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            sync=sync,
            cache_positions=cache_positions,
            request_configs=request_configs,
            protected_prefix_tokens=protected_prefix_tokens,
            req_id=request_id,
        )
        retrieved_count = next(retriever)
        validate_retrieval_count(
            expected=expected_retrieval_count(mask, len(tokens)),
            actual=retrieved_count,
            request_id=request_id,
            path=path,
        )
        # Prime the layerwise pipeline.
        next(retriever)
        return retriever

    def _ensure_resident_slot_positions(
        self,
        axes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if axes not in (1, 3):
            raise RuntimeError(f"unsupported resident position axes: {axes}")
        if not self.kv_caches:
            raise RuntimeError("resident position tracking requires KV caches")
        first_cache = next(iter(self.kv_caches.values()))
        if first_cache.ndim == 5:
            if first_cache.shape[0] != 2:
                raise RuntimeError(
                    "resident position tracking requires K/V-major cache layout"
                )
            slot_capacity = int(first_cache.shape[1] * first_cache.shape[2])
        elif first_cache.ndim == 3:
            slot_capacity = int(first_cache.shape[0] * first_cache.shape[1])
        else:
            raise RuntimeError(
                "unsupported resident KV tensor layout: "
                f"{tuple(first_cache.shape)}"
            )
        table = self._resident_slot_positions
        if table is None:
            table = torch.full(
                (axes, slot_capacity),
                -1,
                dtype=torch.long,
                device=device,
            )
            self._resident_slot_positions = table
        elif table.shape != (axes, slot_capacity) or table.device != device:
            raise RuntimeError(
                "resident KV position table is incompatible with the active cache"
            )
        return table

    def _ensure_resident_rotary_repair(self, position_mode: str) -> Any:
        if self._resident_rotary_embedding is None:
            try:
                vllm_model = VLLMModelTracker.get_model(ENGINE_NAME)
            except (ValueError, KeyError) as exc:
                raise RuntimeError(
                    "resident position repair requires the registered vLLM model"
                ) from exc
            _, layers = _resolve_decoder_layers(vllm_model)
            if not layers:
                raise RuntimeError(
                    "resident position repair found no decoder layers"
                )
            self._resident_rotary_embedding = (
                layers[0].self_attn.rotary_emb
            )

        rotary = self._resident_rotary_embedding
        is_mrope = bool(getattr(rotary, "mrope_section", None))
        if position_mode == "mrope_3d":
            if not is_mrope:
                raise RuntimeError(
                    "resident mRoPE repair requires a three-axis rotary embedding"
                )
            return rotary
        if position_mode != "rope_1d" or is_mrope:
            raise RuntimeError(
                "resident RoPE repair mode does not match the active model"
            )
        if self._resident_fused_rope is None:
            fused = get_fused_rope_from_vllm(rotary)
            if fused is None:
                raise RuntimeError(
                    "resident RoPE repair requires a supported rotary embedding"
                )
            fused.rope_cache_to_device(next(iter(self.kv_caches.values())).device)
            self._resident_fused_rope = fused
        return self._resident_fused_rope

    def record_kv_positions(
        self,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if not self._resident_overlap_enabled or positions.numel() == 0:
            return
        if positions.ndim == 1:
            normalized = positions.unsqueeze(0)
        elif positions.ndim == 2 and positions.shape[0] == 3:
            normalized = positions
        else:
            raise RuntimeError(
                "resident KV positions must have shape [tokens] or [3, tokens]"
            )
        slots = slot_mapping.reshape(-1)
        if normalized.shape[1] != slots.numel():
            raise RuntimeError(
                "resident KV positions and slot mapping have different lengths"
            )
        torch._assert_async(
            torch.all(slots >= 0),
            "resident KV position recording found an invalid slot",
        )
        table = self._ensure_resident_slot_positions(
            int(normalized.shape[0]), normalized.device
        )
        table.index_copy_(1, slots.to(device=table.device), normalized.long())

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

        self.lmcache_engine.post_init(kvcaches=kvcaches)

        self.layerwise_retrievers = []
        self._layerwise_load_requests = []
        self._layerwise_batch_retriever = None
        self._layerwise_batch_requests = []
        self._request_profiles = {}
        self._resident_repack_timing = None
        self._resident_repack_component_timing = None
        resident_repack_profiles: dict[str, dict[str, Any]] = {}
        if metadata.resident_copy_specs:
            component_profile = (
                os.environ.get("COSTREAM_RUNTIME_OVERHEAD_PROFILE", "0")
                == "1"
            )
            copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            rope_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

            def timed_events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
                return (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            source_slots_flat: list[int] = []
            target_slots_flat: list[int] = []
            repair_indices_flat: list[int] = []
            position_modes = {
                copy_spec.position_mode
                for copy_spec in metadata.resident_copy_specs
            }
            if len(position_modes) != 1:
                raise RuntimeError(
                    "one resident repack batch cannot mix position modes"
                )
            position_mode = next(iter(position_modes))
            token_cursor = 0
            for copy_spec in metadata.resident_copy_specs:
                if len(copy_spec.source_slots) != len(copy_spec.target_slots):
                    raise RuntimeError(
                        "resident repack source/target slot count mismatch"
                    )
                if len(copy_spec.source_slots) != copy_spec.num_tokens:
                    raise RuntimeError(
                        "resident repack slot count does not match token count"
                    )
                source_slots_flat.extend(copy_spec.source_slots)
                target_slots_flat.extend(copy_spec.target_slots)
                if copy_spec.repair_positions:
                    repair_indices_flat.extend(range(
                        token_cursor,
                        token_cursor + copy_spec.num_tokens,
                    ))
                stats = resident_repack_profiles.setdefault(
                    copy_spec.request_id,
                    {
                        "resident_repack_tokens": 0,
                        "resident_repack_copy_specs": 0,
                        "resident_rope_repaired_tokens": 0,
                        "resident_position_repair_applied": True,
                        "resident_position_mode": position_mode,
                        "resident_repack_bytes": 0,
                        "resident_repack_bytes_aggregate": 0,
                        "resident_repack_bytes_scope": (
                            "per_tensor_parallel_rank"
                        ),
                    },
                )
                stats["resident_repack_tokens"] += copy_spec.num_tokens
                stats["resident_repack_copy_specs"] += 1
                if copy_spec.repair_positions:
                    stats["resident_rope_repaired_tokens"] += (
                        copy_spec.num_tokens
                    )
                token_cursor += copy_spec.num_tokens
            source_slots = torch.tensor(
                source_slots_flat,
                dtype=torch.long,
                device=kvcaches[0].device,
            )
            target_slots = torch.tensor(
                target_slots_flat,
                dtype=torch.long,
                device=kvcaches[0].device,
            )
            position_axes = 3 if position_mode == "mrope_3d" else 1
            position_table = self._ensure_resident_slot_positions(
                position_axes, kvcaches[0].device
            )
            source_positions = position_table.index_select(1, source_slots)
            torch._assert_async(
                torch.all(source_positions >= 0),
                "resident source slots are missing recorded positions",
            )
            target_positions = source_positions.clone()
            prompt_positions_cpu = kwargs.get("request_prompt_positions") or {}
            prompt_positions_gpu: dict[str, torch.Tensor] = {}
            token_cursor = 0
            for copy_spec in metadata.resident_copy_specs:
                token_end = token_cursor + copy_spec.num_tokens
                if copy_spec.repair_positions:
                    if position_mode == "rope_1d":
                        target_positions[0, token_cursor:token_end] = torch.arange(
                            copy_spec.target_token_start,
                            copy_spec.target_token_start + copy_spec.num_tokens,
                            device=target_positions.device,
                            dtype=torch.long,
                        )
                    else:
                        prompt_positions = prompt_positions_cpu.get(
                            copy_spec.request_id
                        )
                        if prompt_positions is None:
                            raise RuntimeError(
                                "resident mRoPE repair requires current prompt "
                                f"positions for request={copy_spec.request_id}"
                            )
                        if (
                            prompt_positions.ndim != 2
                            or prompt_positions.shape[0] != 3
                            or prompt_positions.shape[1]
                            < copy_spec.target_token_start + copy_spec.num_tokens
                        ):
                            raise RuntimeError(
                                "resident mRoPE prompt positions do not cover "
                                f"request={copy_spec.request_id}"
                            )
                        current_positions = prompt_positions_gpu.get(
                            copy_spec.request_id
                        )
                        if current_positions is None:
                            current_positions = prompt_positions.to(
                                device=target_positions.device,
                                dtype=torch.long,
                                non_blocking=True,
                            )
                            prompt_positions_gpu[
                                copy_spec.request_id
                            ] = current_positions
                        target_positions[:, token_cursor:token_end] = (
                            current_positions[
                                :,
                                copy_spec.target_token_start:
                                copy_spec.target_token_start
                                + copy_spec.num_tokens,
                            ]
                        )
                token_cursor = token_end

            repair_indices = torch.tensor(
                repair_indices_flat,
                dtype=torch.long,
                device=kvcaches[0].device,
            )
            repair_source_positions = source_positions.index_select(
                1, repair_indices
            )
            repair_target_positions = target_positions.index_select(
                1, repair_indices
            )
            rotary_repair = (
                self._ensure_resident_rotary_repair(position_mode)
                if repair_indices_flat else None
            )
            copied_bytes = 0
            for kv_layer in kvcaches:
                original_shape = kv_layer.shape
                if kv_layer.ndim == 5:
                    if original_shape[0] != 2:
                        raise RuntimeError(
                            "resident repack requires K/V-major cache layout"
                        )
                    flattened = kv_layer.reshape(
                        2, original_shape[1] * original_shape[2], -1
                    )
                    gather_events = (
                        timed_events() if component_profile else None
                    )
                    if gather_events is not None:
                        gather_events[0].record()
                    values = flattened.index_select(1, source_slots)
                    if gather_events is not None:
                        gather_events[1].record()
                        copy_events.append(gather_events)
                    if rotary_repair is not None:
                        layer_rope_events = (
                            timed_events() if component_profile else None
                        )
                        if layer_rope_events is not None:
                            layer_rope_events[0].record()
                        repair_keys = values[0].index_select(
                            0, repair_indices
                        )
                        if position_mode == "mrope_3d":
                            rotary = rotary_repair
                            repair_keys = _mrope_delta_rotate_k(
                                repair_keys,
                                repair_source_positions,
                                repair_target_positions,
                                rotary.cos_sin_cache,
                                rotary.head_size,
                                rotary.mrope_section,
                                rotary.mrope_interleaved,
                            )
                        else:
                            repair_keys = rotary_repair(
                                repair_source_positions[0],
                                repair_target_positions[0],
                                repair_keys,
                            )
                        values[0].index_copy_(
                            0, repair_indices, repair_keys
                        )
                        if layer_rope_events is not None:
                            layer_rope_events[1].record()
                            rope_events.append(layer_rope_events)
                    scatter_events = (
                        timed_events() if component_profile else None
                    )
                    if scatter_events is not None:
                        scatter_events[0].record()
                    flattened.index_copy_(1, target_slots, values)
                    if scatter_events is not None:
                        scatter_events[1].record()
                        copy_events.append(scatter_events)
                elif kv_layer.ndim == 3:
                    if rotary_repair is not None:
                        raise RuntimeError(
                            "resident position repair does not support combined "
                            "three-dimensional KV layouts"
                        )
                    flattened = kv_layer.reshape(
                        original_shape[0] * original_shape[1], -1
                    )
                    layer_copy_events = (
                        timed_events() if component_profile else None
                    )
                    if layer_copy_events is not None:
                        layer_copy_events[0].record()
                    values = flattened.index_select(0, source_slots)
                    flattened.index_copy_(0, target_slots, values)
                    if layer_copy_events is not None:
                        layer_copy_events[1].record()
                        copy_events.append(layer_copy_events)
                else:
                    raise RuntimeError(
                        "unsupported resident KV tensor layout: "
                        f"{tuple(original_shape)}"
                    )
                copied_bytes += values.numel() * values.element_size()
            position_copy_events = (
                timed_events() if component_profile else None
            )
            if position_copy_events is not None:
                position_copy_events[0].record()
            position_table.index_copy_(
                1, target_slots, target_positions
            )
            if position_copy_events is not None:
                position_copy_events[1].record()
                copy_events.append(position_copy_events)
            bytes_per_slot = copied_bytes // len(source_slots_flat)
            batch_requests = len(resident_repack_profiles)
            batch_tokens = len(source_slots_flat)
            for stats in resident_repack_profiles.values():
                request_bytes = (
                    int(stats["resident_repack_tokens"]) * bytes_per_slot
                )
                stats["resident_repack_bytes"] = request_bytes
                stats["resident_repack_bytes_aggregate"] = (
                    request_bytes * self.worker_count
                )
                stats["resident_repack_batch_requests"] = batch_requests
                stats["resident_repack_batch_tokens"] = batch_tokens
                stats["resident_position_table_bytes"] = (
                    position_table.numel() * position_table.element_size()
                )
            end_event.record()
            total_tokens = sum(
                int(stats["resident_repack_tokens"])
                for stats in resident_repack_profiles.values()
            )
            self._resident_repack_timing = (
                start_event, end_event, total_tokens
            )
            if component_profile:
                self._resident_repack_component_timing = (
                    tuple(copy_events), tuple(rope_events), total_tokens
                )
            for stats in resident_repack_profiles.values():
                stats["resident_repack_async_host"] = True
        pending_batch_requests: list[LayerwiseRetrievalRequest] = []
        pending_batch_metadata: list[ReqMeta] = []
        batch_fetch_enabled = bool(
            (self.config.extra_config or {}).get("batch_fetch", False)
        )
        joint_refresh_enabled = bool(
            (self.config.extra_config or {}).get("joint_refresh", False)
        )
        joint_refresh_mode = joint_refresh_enabled and (
            self.config.blend_mode == "codecsight"
            or getattr(self.config, "is_codecsight", False)
        )

        bytes_per_token = (
            self.lmcache_engine.gpu_connector.get_shape(1).numel()
            * torch.empty((), dtype=self.lmcache_engine.metadata.kv_dtype)
            .element_size()
            * self.num_layers
            * self.worker_count
        )

        for request in metadata.requests:
            load_spec = request.load_spec
            refresh_spec = request.refresh_spec
            requested = "full_prefill"
            if load_spec is not None:
                if refresh_spec is not None:
                    requested = refresh_spec.policy
                elif self.enable_blending:
                    requested = getattr(
                        getattr(self, "blender", None),
                        "blend_mode",
                        getattr(self.config, "blend_mode", "blending"),
                    )
                else:
                    requested = "direct_reuse"
            profile = {
                "requested_path": requested,
                "executed_path": "pending",
                "fallback_reason": None,
                "silent_fallback": False,
                "reused_kv_tokens": 0,
                "loaded_kv_tokens": 0,
                "external_kv_tokens": 0,
                "protected_apc_tokens": 0,
                "refreshed_kv_tokens": (
                    refresh_spec.num_refresh_tokens
                    if refresh_spec is not None else 0
                ),
                "kv_load_bytes": 0,
                "kv_store_bytes": 0,
                "kv_fetch_ms": 0.0,
                "kv_writeback_ms": 0.0,
                "kv_bytes_scope": "aggregate_tensor_parallel_ranks",
                "kv_materialization_mode": (
                    "vllm_resident_blocks"
                    if self._resident_disable_lmcache_storage
                    else "lmcache_local_gpu_copy"
                ),
                "lmcache_local_gpu_reserved_bytes": int(
                    self.config.max_local_gpu_size * 1024**3
                    * self.worker_count
                ) if self.config.local_gpu else 0,
                "resident_shadow_enabled": self._resident_shadow_enabled,
                "resident_zero_copy_enabled": self._resident_zero_copy_enabled,
            }
            if refresh_spec is not None:
                profile.update(
                    _joint_refresh_selection_stats(
                        refresh_spec,
                        request.mm_positions,
                    )
                )
            if load_spec is not None:
                fetched = max(
                    0,
                    load_spec.lmcache_cached_tokens
                    - (load_spec.vllm_cached_tokens
                       // self._lmcache_chunk_size
                       * self._lmcache_chunk_size),
                )
                profile["external_kv_tokens"] = max(
                    0,
                    load_spec.lmcache_cached_tokens
                    - load_spec.vllm_cached_tokens,
                )
                profile["reused_kv_tokens"] = fetched
                profile["loaded_kv_tokens"] = fetched
                profile["protected_apc_tokens"] = max(
                    0,
                    min(
                        load_spec.vllm_cached_tokens,
                        load_spec.lmcache_cached_tokens,
                    )
                    - (load_spec.vllm_cached_tokens
                       // self._lmcache_chunk_size
                       * self._lmcache_chunk_size),
                )
                profile["kv_load_bytes"] = fetched * bytes_per_token
            self._request_profiles[request.req_id] = profile

        def queue_batch_request(
            request: ReqMeta,
            tokens: list[int],
            mask: torch.Tensor,
            slot_mapping: torch.Tensor,
            cache_positions: Optional[torch.Tensor],
        ) -> None:
            pending_batch_requests.append(
                LayerwiseRetrievalRequest(
                    request_id=request.req_id,
                    tokens=tokens,
                    mask=mask,
                    slot_mapping=slot_mapping,
                    cache_positions=cache_positions,
                    request_configs=request.request_configs,
                    protected_prefix_tokens=(
                        request.load_spec.vllm_cached_tokens
                    ),
                )
            )
            pending_batch_metadata.append(request)
            assert request.load_spec is not None
            self._stats_monitor.update_interval_vllm_hit_tokens(
                request.load_spec.vllm_cached_tokens
            )
            self._stats_monitor.update_interval_prompt_tokens(
                len(request.token_ids)
            )

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                logger.debug("skip request due to load spec is None")
                self._request_profiles[request.req_id][
                    "executed_path"
                ] = "full_prefill"
                continue

            tokens = request.token_ids
            model_tokens = request.model_token_ids or tokens
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            # vLLM has already skipped the advertised cache-hit prefix.
            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if lmcache_cached_tokens > len(tokens):
                raise RuntimeError(
                    "LMCache scheduled more cached tokens than the worker "
                    f"received ({lmcache_cached_tokens} > {len(tokens)}); "
                    "refusing silent partial-prefix execution."
                )
            physical_loaded_tokens = sum(
                end - start
                for start, end, _ in (
                    self.lmcache_engine.token_database.process_tokens(
                        tokens=tokens[:lmcache_cached_tokens],
                        mask=token_mask[:lmcache_cached_tokens],
                        request_configs=request.request_configs,
                    )
                )
            )
            external_tokens = max(
                0,
                lmcache_cached_tokens
                - request.load_spec.vllm_cached_tokens,
            )
            profile = self._request_profiles[request.req_id]
            profile["reused_kv_tokens"] = physical_loaded_tokens
            profile["loaded_kv_tokens"] = physical_loaded_tokens
            profile["protected_apc_tokens"] = max(
                0, physical_loaded_tokens - external_tokens,
            )
            profile["kv_load_bytes"] = (
                physical_loaded_tokens * bytes_per_token
            )
            cache_positions = self._compute_request_cache_positions(
                request, lmcache_cached_tokens, slot_mapping.device,
            )
            logger.debug(
                "start_load_kv: blending=%s, layerwise=%s",
                self.enable_blending,
                self.use_layerwise,
            )
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                logger.debug("start_load_kv: blending=%s", self.enable_blending)
                if self.enable_blending:
                    if joint_refresh_mode and request.refresh_spec is not None:
                        self._request_profiles[request.req_id][
                            "executed_path"
                        ] = f"joint_{request.refresh_spec.policy}_refresh"
                        if batch_fetch_enabled:
                            queue_batch_request(
                                request,
                                tokens[:lmcache_cached_tokens],
                                token_mask[:lmcache_cached_tokens],
                                slot_mapping[:lmcache_cached_tokens],
                                cache_positions,
                            )
                        else:
                            retriever = self._start_checked_layerwise_retrieval(
                                request_id=request.req_id,
                                tokens=tokens[:lmcache_cached_tokens],
                                mask=token_mask[:lmcache_cached_tokens],
                                kvcaches=kvcaches,
                                slot_mapping=slot_mapping[:lmcache_cached_tokens],
                                sync=sync,
                                cache_positions=cache_positions,
                                request_configs=request.request_configs,
                                protected_prefix_tokens=(
                                    request.load_spec.vllm_cached_tokens
                                ),
                                path="joint_refresh",
                            )
                            self.layerwise_retrievers.append(retriever)
                            self._layerwise_load_requests.append(request)
                        continue
                    if joint_refresh_mode:
                        self._request_profiles[request.req_id][
                            "executed_path"
                        ] = "joint_reuse_no_refresh"
                        if batch_fetch_enabled:
                            queue_batch_request(
                                request,
                                tokens[:lmcache_cached_tokens],
                                token_mask[:lmcache_cached_tokens],
                                slot_mapping[:lmcache_cached_tokens],
                                cache_positions,
                            )
                        else:
                            retriever = self._start_checked_layerwise_retrieval(
                                request_id=request.req_id,
                                tokens=tokens[:lmcache_cached_tokens],
                                mask=token_mask[:lmcache_cached_tokens],
                                kvcaches=kvcaches,
                                slot_mapping=slot_mapping[:lmcache_cached_tokens],
                                sync=sync,
                                cache_positions=cache_positions,
                                request_configs=request.request_configs,
                                protected_prefix_tokens=(
                                    request.load_spec.vllm_cached_tokens
                                ),
                                path="joint_reuse_no_refresh",
                            )
                            self.layerwise_retrievers.append(retriever)
                            self._layerwise_load_requests.append(request)
                        continue
                    self._ensure_blender_initialized()
                    if self.blender is None:
                        raise RuntimeError(
                            "LMCache blending was requested but the blender is "
                            f"unavailable for request={request.req_id}; refusing "
                            "to silently execute a different cache strategy."
                        )

                    page_stream = self.lmcache_engine.gpu_connector.get_page_stream()

                    skip_embeds = (
                        getattr(self.blender, "blend_mode", "") == "direct_reuse"
                        and getattr(
                            self.blender, "direct_reuse_retrieve_only", False)
                    )
                    if batch_fetch_enabled and skip_embeds:
                        self._request_profiles[request.req_id][
                            "executed_path"
                        ] = "direct_reuse"
                        queue_batch_request(
                            request,
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            slot_mapping[:lmcache_cached_tokens],
                            cache_positions,
                        )
                        continue
                    if skip_embeds:
                        embedding_provider = None
                    else:
                        if not _has_visual_prefix(
                            request.mm_hashes,
                            request.mm_positions,
                            lmcache_cached_tokens,
                        ):
                            logger.info(
                                "LMCache cache strategy no-op: request=%s, "
                                "requested_mode=%s, "
                                "executed_mode=visual_cache_miss_prefill, "
                                "reason=no_visual_prefix, cached_tokens=%d",
                                request.req_id,
                                getattr(self.blender, "blend_mode", "unknown"),
                                lmcache_cached_tokens,
                            )
                            if batch_fetch_enabled:
                                profile = self._request_profiles[request.req_id]
                                profile["executed_path"] = (
                                    "visual_cache_miss_prefill"
                                )
                                profile["fallback_reason"] = "no_visual_prefix"
                                profile["visual_cache_miss"] = True
                                profile["visual_kv_reused_tokens"] = 0
                                queue_batch_request(
                                    request,
                                    tokens[:lmcache_cached_tokens],
                                    token_mask[:lmcache_cached_tokens],
                                    slot_mapping[:lmcache_cached_tokens],
                                    cache_positions,
                                )
                                continue
                            layerwise_retriever = (
                                self._start_checked_layerwise_retrieval(
                                    request_id=request.req_id,
                                    tokens=tokens[:lmcache_cached_tokens],
                                    mask=token_mask[:lmcache_cached_tokens],
                                    kvcaches=kvcaches,
                                    slot_mapping=slot_mapping[
                                        :lmcache_cached_tokens],
                                    sync=sync,
                                    cache_positions=cache_positions,
                                    request_configs=request.request_configs,
                                    protected_prefix_tokens=(
                                        request.load_spec.vllm_cached_tokens
                                    ),
                                    path="no_visual_prefix",
                                )
                            )
                            self.layerwise_retrievers.append(
                                layerwise_retriever)
                            self._layerwise_load_requests.append(request)
                            profile = self._request_profiles[request.req_id]
                            profile["executed_path"] = (
                                "visual_cache_miss_prefill"
                            )
                            profile["fallback_reason"] = "no_visual_prefix"
                            profile["visual_cache_miss"] = True
                            profile["visual_kv_reused_tokens"] = 0
                            continue

                        def _embedding_provider(
                            model_tokens=model_tokens,
                            mm_hashes=request.mm_hashes,
                            mm_positions=request.mm_positions,
                            num_tokens=lmcache_cached_tokens,
                            request_id=request.req_id,
                        ):
                            reconstruction = self._reconstruct_inputs_embeds(
                                model_tokens,
                                mm_hashes,
                                mm_positions,
                                num_tokens,
                                request_id=request_id,
                            )
                            if not reconstruction.ready:
                                raise RuntimeError(
                                    "LMCache selective refresh cannot "
                                    "reconstruct visual embeddings: "
                                    f"request={request_id}, "
                                    f"status={reconstruction.status}, "
                                    f"detail={reconstruction.detail}. "
                                    "Refusing to silently execute direct reuse."
                                )
                            assert reconstruction.inputs_embeds is not None
                            return (
                                reconstruction.inputs_embeds,
                                reconstruction.deepstack_input_embeds,
                            )

                        embedding_provider = _embedding_provider

                    logger.debug(
                        "start_load_kv: embedding_reconstruction=%s, "
                        "mm_hashes=%d, mm_positions=%d, "
                        "cached_tokens=%d",
                        "deferred" if embedding_provider is not None else "skipped",
                        len(request.mm_hashes) if request.mm_hashes else 0,
                        len(request.mm_positions) if request.mm_positions else 0,
                        lmcache_cached_tokens,
                    )

                    selection_stats = self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        tokens_per_frame=request.tokens_per_frame,
                        mm_positions=request.mm_positions,
                        image_grid_thw=request.image_grid_thw,
                        model_input_ids=model_tokens[:lmcache_cached_tokens],
                        cache_positions=cache_positions,
                        request_configs=request.request_configs,
                        protected_prefix_tokens=(
                            request.load_spec.vllm_cached_tokens
                        ),
                        page_stream=page_stream,
                        sync=sync,
                        embedding_provider=embedding_provider,
                        req_id=request.req_id,
                    )
                    profile = self._request_profiles[request.req_id]
                    profile["executed_path"] = (
                        f"eager_{getattr(self.blender, 'blend_mode', 'blend')}"
                    )
                    if selection_stats:
                        profile.update(selection_stats)
                    for layer_id, (layer_name, kv_layer) in enumerate(
                        self.kv_caches.items()
                    ):
                        self._kv_diag.capture(
                            "prepared", layer_id, layer_name, kv_layer,
                            request, lmcache_cached_tokens,
                        )
                else:
                    if batch_fetch_enabled:
                        self._request_profiles[request.req_id][
                            "executed_path"
                        ] = "direct_reuse"
                        queue_batch_request(
                            request,
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            slot_mapping[:lmcache_cached_tokens],
                            cache_positions,
                        )
                        continue
                    layerwise_retriever = self._start_checked_layerwise_retrieval(
                        request_id=request.req_id,
                        tokens=tokens[:lmcache_cached_tokens],
                        mask=token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        sync=sync,
                        cache_positions=cache_positions,
                        request_configs=request.request_configs,
                        protected_prefix_tokens=(
                            request.load_spec.vllm_cached_tokens
                        ),
                        path="plain_layerwise",
                    )
                    self.layerwise_retrievers.append(layerwise_retriever)
                    self._layerwise_load_requests.append(request)
                    self._request_profiles[request.req_id][
                        "executed_path"
                    ] = "direct_reuse"
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    request_configs=request.request_configs,
                    protected_prefix_tokens=(
                        request.load_spec.vllm_cached_tokens
                    ),
                    req_id=request.req_id,
                    skip_contains_check=True,
                )

                # Check the result
                validate_retrieval_count(
                    expected=expected_retrieval_count(
                        token_mask[:lmcache_cached_tokens],
                        lmcache_cached_tokens,
                    ),
                    actual=ret_token_mask,
                    request_id=request.req_id,
                    path="non_layerwise",
                )
                for layer_id, (layer_name, kv_layer) in enumerate(
                    self.kv_caches.items()
                ):
                    self._kv_diag.capture(
                        "prepared", layer_id, layer_name, kv_layer,
                        request, lmcache_cached_tokens,
                    )
                self._request_profiles[request.req_id][
                    "executed_path"
                ] = "direct_reuse_non_layerwise"

            self._stats_monitor.update_interval_vllm_hit_tokens(
                request.load_spec.vllm_cached_tokens
            )
            self._stats_monitor.update_interval_prompt_tokens(len(tokens))

        if len(pending_batch_requests) > 1:
            batch_retriever = self.lmcache_engine.retrieve_layer_batch(
                pending_batch_requests,
                kvcaches=kvcaches,
            )
            batch_info = next(batch_retriever)
            if not isinstance(batch_info, LayerwiseRetrievalBatchInfo):
                raise RuntimeError("batched retrieval returned invalid preflight info")
            retrieved_counts = batch_info.counts
            for request, pending, count in zip(
                pending_batch_metadata,
                pending_batch_requests,
                retrieved_counts,
                strict=True,
            ):
                validate_retrieval_count(
                    expected=expected_retrieval_count(
                        pending.mask, len(pending.tokens),
                    ),
                    actual=count,
                    request_id=request.req_id,
                    path="plain_layerwise_batch",
                )
            logger.info(
                "LMCache batch fetch admitted %d requests: %s",
                len(pending_batch_metadata),
                ", ".join(
                    f"{request.req_id}={int(count)}"
                    for request, count in zip(
                        pending_batch_metadata,
                        retrieved_counts,
                        strict=True,
                    )
                ),
            )
            for request in pending_batch_metadata:
                self._request_profiles[request.req_id][
                    "fetch_mode"
                ] = "multi_request_batch"
            next(batch_retriever)
            self._layerwise_batch_retriever = batch_retriever
            self._layerwise_batch_requests = pending_batch_metadata
        elif pending_batch_requests:
            pending = pending_batch_requests[0]
            request = pending_batch_metadata[0]
            retriever = self._start_checked_layerwise_retrieval(
                request_id=pending.request_id,
                tokens=pending.tokens,
                mask=pending.mask,
                kvcaches=kvcaches,
                slot_mapping=pending.slot_mapping,
                sync=True,
                cache_positions=pending.cache_positions,
                request_configs=pending.request_configs,
                protected_prefix_tokens=pending.protected_prefix_tokens,
                path="plain_layerwise",
            )
            self.layerwise_retrievers.append(retriever)
            self._layerwise_load_requests.append(request)
            self._request_profiles[request.req_id][
                "fetch_mode"
            ] = "single_request"

        for request_id, repack_profile in resident_repack_profiles.items():
            profile = self._request_profiles.setdefault(request_id, {})
            profile.update(repack_profile)
            profile.update({
                "requested_path": "resident_overlap_multi_anchor",
                "executed_path": "resident_overlap_multi_anchor",
                "kv_materialization_mode": "vllm_resident_token_repack",
                "loaded_kv_tokens": 0,
                "kv_load_bytes": 0,
            })

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        wait_started = time.perf_counter()
        if self.layerwise_retrievers or self._layerwise_batch_retriever:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        if self._layerwise_batch_retriever is not None:
            ret_masks = next(self._layerwise_batch_retriever)
            for request in self._layerwise_batch_requests:
                if self.current_layer < len(self.kv_caches):
                    layer_name_at_index, kv_layer = list(
                        self.kv_caches.items()
                    )[self.current_layer]
                    self._kv_diag.capture(
                        "prepared",
                        self.current_layer,
                        layer_name_at_index,
                        kv_layer,
                        request,
                        request.load_spec.lmcache_cached_tokens,
                    )
            if self.current_layer == self.num_layers - 1:
                assert ret_masks is not None
                logger.debug(
                    "Batched retrieval completed for %d requests",
                    len(ret_masks),
                )

        # Wait for the layer to be loaded
        for request, layerwise_retriever in zip(
            self._layerwise_load_requests,
            self.layerwise_retrievers,
            strict=True,
        ):
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer < len(self.kv_caches):
                layer_name_at_index, kv_layer = list(
                    self.kv_caches.items()
                )[self.current_layer]
                self._kv_diag.capture(
                    "prepared", self.current_layer, layer_name_at_index,
                    kv_layer, request,
                    request.load_spec.lmcache_cached_tokens,
                )

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.debug(f"Retrieved {num_retrieved_tokens} tokens")

        elapsed_ms = (time.perf_counter() - wait_started) * 1000.0
        request_ids = {
            request.req_id for request in self._layerwise_batch_requests
        }
        request_ids.update(
            request.req_id for request in self._layerwise_load_requests
        )
        for request_id in request_ids:
            profile = self._request_profiles.get(request_id)
            if profile is not None:
                profile["kv_fetch_ms"] += elapsed_ms

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        for request in connector_metadata.requests:
            self._kv_diag.capture(
                "final", self.current_layer, layer_name, kv_layer, request
            )

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        if self.current_layer == 0:
            self.layerwise_storers = []
            self._writeback_inline_seconds = 0.0
            self._writeback_started_at = time.perf_counter()
            self._writeback_request_ids = []
            self._cacheblend_writeback_states: list[
                tuple[str, dict[str, Any]]
            ] = []

            is_first = True

            for idx, request in enumerate(connector_metadata.requests):
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids[:request.save_token_count]
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping[:request.save_token_count]
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.cuda()

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                logger.debug(f"kv_role: {self.kv_role}, layer_name: {layer_name}, skip_leading_tokens: {skip_leading_tokens}")
                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "save_kv_layer->Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )
                stored_tokens = len(token_ids) - skip_leading_tokens
                bytes_per_token = (
                    self.lmcache_engine.gpu_connector.get_shape(1).numel()
                    * torch.empty(
                        (), dtype=self.lmcache_engine.metadata.kv_dtype
                    ).element_size()
                    * self.num_layers
                    * self.worker_count
                )
                profile = self._request_profiles.setdefault(
                    request.req_id, {}
                )
                profile["stored_kv_tokens"] = stored_tokens
                profile["kv_store_bytes"] = stored_tokens * bytes_per_token
                if self._cacheblend_mode:
                    profile["cacheblend_writeback_path"] = (
                        "async_direct_cuda_event"
                        if self._cacheblend_event_writeback
                        else "per_layer_sync"
                    )
                    profile["kv_writeback_accounting"] = (
                        "enqueue_critical_path"
                        if self._cacheblend_event_writeback
                        else "synchronous_completion"
                    )
                self._writeback_request_ids.append(request.req_id)

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                writeback_state: dict[str, Any] = {}
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    cache_positions=self._compute_request_cache_positions(
                        request, len(token_ids), slot_mapping.device,
                    ),
                    request_configs=request.request_configs,
                    defer_layer_sync=self._cacheblend_event_writeback,
                    direct_memory_writeback=self._cacheblend_event_writeback,
                    async_publication=self._cacheblend_event_writeback,
                    writeback_state=writeback_state,
                )
                self.layerwise_storers.append(layerwise_storer)
                self._cacheblend_writeback_states.append(
                    (request.req_id, writeback_state)
                )
                if is_first:
                    is_first = False

        inline_start = time.perf_counter()
        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)
        self._writeback_inline_seconds += time.perf_counter() - inline_start

        self.current_layer += 1

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            tail_start = time.perf_counter()
            for layerwise_storer in self.layerwise_storers:
                next(layerwise_storer)
            tail_seconds = time.perf_counter() - tail_start
            writeback_ms = (
                self._writeback_inline_seconds + tail_seconds
            ) * 1000.0
            for request_id in self._writeback_request_ids:
                profile = self._request_profiles.get(request_id)
                if profile is not None:
                    profile["kv_writeback_ms"] = writeback_ms
                    profile["kv_writeback_inline_ms"] = (
                        self._writeback_inline_seconds * 1000.0
                    )
                    profile["kv_writeback_tail_ms"] = tail_seconds * 1000.0

            if self._cacheblend_event_writeback:
                for request_id, state in self._cacheblend_writeback_states:
                    if state.get("completion_event") is None:
                        continue
                    self._cacheblend_pending_writebacks[request_id] = state
                    profile = self._request_profiles.get(request_id)
                    if profile is not None:
                        profile["cacheblend_writeback_pending"] = True

            if self._log_writeback_timing and self._writeback_started_at is not None:
                elapsed_seconds = time.perf_counter() - self._writeback_started_at
                logger.info(
                    "LMCache write-back critical path: inline=%.3f ms, "
                    "tail_wait=%.3f ms, synchronous_total=%.3f ms, "
                    "forward_overlap_window=%.3f ms, requests=%d",
                    self._writeback_inline_seconds * 1000,
                    tail_seconds * 1000,
                    (self._writeback_inline_seconds + tail_seconds) * 1000,
                    elapsed_seconds * 1000,
                    len(self.layerwise_storers),
                )
            self._writeback_started_at = None

            # unpin the kv caches according to req_id
            for request in connector_metadata.requests:
                if request.req_id not in self._cacheblend_pending_writebacks:
                    self.lmcache_engine.lookup_unpin(request.req_id)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids[:request.save_token_count]

            slot_mapping = request.slot_mapping[:request.save_token_count]
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()

            skip_leading_tokens = save_spec.skip_leading_tokens
            if self.kv_role == "kv_producer":
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            logger.debug(f"kv_role: {self.kv_role}, before-skip_leading_tokens: {skip_leading_tokens}")
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            logger.debug(f"kv_role: {self.kv_role}, after-skip_leading_tokens: {skip_leading_tokens}")

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "wait_for_save->Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
            )

            # NOTE(Jiayi): We assume all tokens are saved
            save_spec.skip_leading_tokens = len(token_ids)
            if request.disagg_spec:
                request.disagg_spec.num_transferred_tokens = len(token_ids)

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if not self._cacheblend_event_writeback:
            return None, None

        self._cacheblend_waiting_finished_ids.update(finished_req_ids)
        completed: set[str] = set()
        for request_id in tuple(self._cacheblend_waiting_finished_ids):
            state = self._cacheblend_pending_writebacks.get(request_id)
            event = state.get("completion_event") if state is not None else None
            if event is not None and not event.query():
                continue
            if state is not None:
                for memory_obj in state.get("pinned_memory_objs", ()):
                    memory_obj.metadata.ready_event = None
                    memory_obj.unpin()
                self._cacheblend_pending_writebacks.pop(request_id, None)
                profile = self._request_profiles.get(request_id)
                if profile is not None:
                    profile["cacheblend_writeback_pending"] = False
            assert self.lmcache_engine is not None
            self.lmcache_engine.lookup_unpin(request_id)
            self._cacheblend_waiting_finished_ids.remove(request_id)
            completed.add(request_id)
        return completed or None, None

    def get_request_profiles(self) -> dict[str, dict[str, Any]]:
        timing = getattr(self, "_resident_repack_timing", None)
        if timing is not None and timing[1].query():
            start_event, end_event, total_tokens = timing
            elapsed_ms = start_event.elapsed_time(end_event)
            component_timing = getattr(
                self, "_resident_repack_component_timing", None
            )
            copy_ms = 0.0
            rope_ms = 0.0
            if component_timing is not None:
                copy_pairs, rope_pairs, component_tokens = component_timing
                if component_tokens != total_tokens:
                    raise RuntimeError(
                        "resident component timing token count mismatch"
                    )
                copy_ms = sum(
                    start.elapsed_time(end) for start, end in copy_pairs
                )
                rope_ms = sum(
                    start.elapsed_time(end) for start, end in rope_pairs
                )
            for profile in self._request_profiles.values():
                request_tokens = int(profile.get(
                    "resident_repack_tokens", 0
                ))
                if request_tokens <= 0:
                    continue
                request_ms = (
                    elapsed_ms * request_tokens / total_tokens
                    if total_tokens else 0.0
                )
                profile["resident_repack_ms"] = request_ms
                profile["resident_repack_batch_ms"] = elapsed_ms
                if component_timing is not None:
                    scale = request_tokens / total_tokens if total_tokens else 0.0
                    request_copy_ms = copy_ms * scale
                    request_rope_ms = rope_ms * scale
                    profile["resident_kv_copy_ms"] = request_copy_ms
                    profile["resident_rope_correction_ms"] = request_rope_ms
                    profile["resident_repack_other_ms"] = max(
                        0.0,
                        request_ms - request_copy_ms - request_rope_ms,
                    )
                profile["resident_repack_gbps_per_rank"] = (
                    float(profile.get("resident_repack_bytes", 0))
                    / request_ms / 1_000_000.0
                    if request_ms else 0.0
                )
                profile["resident_repack_gbps_aggregate"] = (
                    float(profile.get(
                        "resident_repack_bytes_aggregate", 0
                    )) / request_ms / 1_000_000.0
                    if request_ms else 0.0
                )
            self._resident_repack_timing = None
            self._resident_repack_component_timing = None
        if self.config.local_gpu:
            # Derive live LocalGPU occupancy from its allocated objects.
            current_bytes = 0
            active_objects = 0
            storage_manager = getattr(
                getattr(self, "lmcache_engine", None),
                "storage_manager", None,
            )
            backend = (
                getattr(storage_manager, "storage_backends", {}).get(
                    "LocalGPUBackend"
                )
                if storage_manager is not None else None
            )
            if backend is not None:
                lock = getattr(backend, "gpu_lock", nullcontext())
                with lock:
                    unique_objects = {
                        id(memory_obj): memory_obj
                        for memory_obj in getattr(
                            backend, "hot_cache", {}
                        ).values()
                    }
                    current_bytes = sum(
                        int(memory_obj.meta.phy_size)
                        for memory_obj in unique_objects.values()
                    )
                    active_objects = len(unique_objects)
            peak_bytes = max(
                int(getattr(
                    self,
                    "_lmcache_local_gpu_peak_actual_used_bytes_per_rank",
                    0,
                )),
                current_bytes,
            )
            self._lmcache_local_gpu_peak_actual_used_bytes_per_rank = (
                peak_bytes
            )
            for profile in self._request_profiles.values():
                profile.update({
                    "lmcache_local_gpu_actual_used_bytes_per_rank": (
                        current_bytes
                    ),
                    "lmcache_local_gpu_peak_actual_used_bytes_per_rank": (
                        peak_bytes
                    ),
                    "lmcache_local_gpu_actual_used_bytes_aggregate": (
                        current_bytes * self.worker_count
                    ),
                    "lmcache_local_gpu_peak_actual_used_bytes_aggregate": (
                        peak_bytes * self.worker_count
                    ),
                    "lmcache_local_gpu_active_objects_per_rank": (
                        active_objects
                    ),
                    "lmcache_local_gpu_usage_scope": (
                        "aggregate_tensor_parallel_ranks"
                    ),
                })
        return {
            request_id: dict(profile)
            for request_id, profile in self._request_profiles.items()
        }

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    ###################
    # Scheduler side APIs
    ####################

    def get_resident_kv_block_ids(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[tuple[list[int], ...]], int]:
        """Return exact resident prefix blocks without materializing K/V."""
        if not self._resident_zero_copy_enabled or num_computed_tokens:
            return None, 0
        if _costream_force_full_compute(request):
            request.system_profile.update({
                "requested_path": "full_prefill",
                "executed_path": "full_prefill",
                "resident_forced_full_compute": True,
                "fallback_reason": None,
                "silent_fallback": False,
            })
            return None, 0
        self._clear_pending_resident_adoption(
            request.request_id, release_blocks=True
        )
        registry = self._resident_registry()
        if registry is None:
            return None, 0

        self._requests_priority[request.request_id] = getattr(
            request, "priority", 0
        )
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            self._probe_resident_frames(request, mm_hashes, mm_positions)
        lora_request = getattr(request, "lora_request", None)
        record = _resident_prompt_block_record(
            request.prompt_token_ids,
            mm_hashes,
            mm_positions,
            self._block_size,
            cache_salt=getattr(request, "cache_salt", None),
            lora_id=int(getattr(lora_request, "lora_int_id", 0) or 0),
        )
        if record is None:
            return self._get_overlapping_resident_prefix(
                request, mm_hashes, mm_positions
            )
        entry = registry.lookup(record["key"])
        if entry is None:
            request.system_profile["resident_exact_prefix_hit"] = False
            return self._get_overlapping_resident_prefix(
                request, mm_hashes, mm_positions
            )
        exact = (
            entry.token_start == 0
            and entry.token_length == record["token_length"]
            and entry.position_fingerprint
            == record["position_fingerprint"]
            and entry.context_hash == record["context_hash"]
        )
        if not exact:
            request.system_profile["resident_exact_prefix_hit"] = False
            request.system_profile["resident_exact_rejected"] = True
            return self._get_overlapping_resident_prefix(
                request, mm_hashes, mm_positions
            )

        resident_tokens = int(entry.token_length)
        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=resident_tokens,
            lmcache_cached_tokens=resident_tokens,
            can_load=False,
        )
        profile = request.system_profile
        profile.update({
            "requested_path": "resident_exact_adopt",
            "executed_path": "resident_exact_adopt",
            "resident_exact_prefix_hit": True,
            "resident_candidate_kv_tokens": resident_tokens,
            "resident_generation_mismatches": 0,
            "resident_position_mismatches": 0,
            "resident_context_mismatches": 0,
            "vllm_reused_kv_tokens": resident_tokens,
            "lmcache_reused_kv_tokens": 0,
            "loaded_kv_tokens": 0,
            "kv_load_bytes": 0,
            "kv_materialization_mode": "vllm_resident_blocks",
            "fallback_reason": None,
            "silent_fallback": False,
        })
        return (list(entry.block_ids),), resident_tokens

    def get_resident_kv_refresh_tokens(self, request: "Request") -> int:
        spec = getattr(
            self, "_resident_pending_refresh_specs", {}
        ).get(request.request_id)
        return spec.num_refresh_tokens if spec is not None else 0

    def rollback_resident_kv_blocks(self, request: "Request") -> None:
        self._clear_pending_resident_adoption(
            request.request_id, release_blocks=True
        )
        self.load_specs.pop(request.request_id, None)

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        self._requests_priority[request.request_id] = getattr(request, "priority", 0)

        if self._kv_diag_force_miss:
            logger.info(
                "KV diagnostic forced miss for request %s",
                request.request_id,
            )
            return 0

        token_ids = request.prompt_token_ids

        # If the request has multimodal hashes, apply them to the token ids
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            self._probe_resident_frames(request, mm_hashes, mm_positions)
        config = getattr(self, "config", None)
        extra_config = getattr(config, "extra_config", None) or {}
        if (
            bool(extra_config.get("joint_refresh", False))
            and _configured_refresh_policy(extra_config) == "multi_anchor"
        ):
            decisions, decision_error = _extract_costream_frame_decisions(
                request, mm_hashes
            )
            if decisions is None:
                request.system_profile.update({
                    "requested_path": "multi_anchor_refresh",
                    "executed_path": "full_prefill",
                    "fallback_reason": (
                        decision_error or "missing_frame_decisions"
                    ),
                    "silent_fallback": False,
                })
                return 0
        if os.environ.get("LMCACHE_DEBUG_MM_HASHES") == "1":
            logger.info(
                "LMCache MM lookup request=%s hashes=%s positions=%s",
                request.request_id,
                [str(value)[:16] for value in (mm_hashes or [])],
                [
                    (int(position.offset), int(position.length))
                    for position in (mm_positions or [])
                ],
            )
        if mm_hashes and mm_positions:
            # TODO(Jiayi): Optimize this
            token_ids = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
            token_ids = token_ids.tolist()

        request_configs = extract_request_configs(request.sampling_params)

        if self.skip_last_n_tokens > 0:
            token_ids = token_ids[: -self.skip_last_n_tokens]

        lookup_id = request.request_id
        logger.debug("request %s: Looking up KV cache for %d tokens with configs %s",
                     request.request_id, len(token_ids), request_configs)
        
        num_external_hit_tokens = self.lookup_client.lookup(
            token_ids,
            lookup_id=lookup_id,
            request_configs=request_configs,
        )
        logger.debug("Lookup result for request %s: %s", request.request_id, num_external_hit_tokens)

        if num_external_hit_tokens is None:
            logger.info(
                "Reqid: %s, Total tokens %d, LMCache hit tokens: None.",
                request.request_id,
                request.num_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            logger.info("Full prompt hit for request %s, need to recompute the last token", request.request_id)
            need_to_allocate -= 1

        logger.info(
            "Reqid: %s, Total tokens %d, LMCache hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if need_to_allocate <= 0:
            return 0

        if (
            num_computed_tokens > 0
            and need_to_allocate < self._block_size
        ):
            request.system_profile["lmcache_bypassed_tail_tokens"] = int(
                need_to_allocate
            )
            logger.info(
                "Reqid: %s, bypassing LMCache for %d-token APC tail",
                request.request_id,
                need_to_allocate,
            )
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    def get_num_kv_refresh_tokens(
        self,
        request: "Request",
        num_external_tokens: int,
    ) -> int:
        if (
            num_external_tokens <= 0
            or not self.enable_blending
            or not bool(
                (self.config.extra_config or {}).get("joint_refresh", False)
            )
        ):
            return 0
        if not (
            self.config.blend_mode == "codecsight"
            or getattr(self.config, "is_codecsight", False)
        ):
            return 0

        load_spec = self.load_specs.get(request.request_id)
        if load_spec is None:
            return 0
        _, mm_positions = extract_mm_features(request)
        if not mm_positions:
            return 0

        extra_config = self.config.extra_config or {}
        policy = _configured_refresh_policy(extra_config)
        if policy == "multi_anchor":
            mm_hashes, _ = extract_mm_features(request)
            decisions, decision_error = _extract_costream_frame_decisions(
                request, mm_hashes
            )
            if decisions is None:
                request.system_profile["fallback_reason"] = (
                    decision_error or "missing_frame_decisions"
                )
                return 0
            spans = _select_multi_anchor_refresh_spans(
                mm_positions,
                decisions,
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
            )
            return sum(span.num_tokens for span in spans)

        span = _select_prefix_refresh_span(
            mm_positions,
            load_spec.lmcache_cached_tokens,
            int(extra_config.get("codecsight_refresh_frames", 3)),
            load_spec.vllm_cached_tokens,
        )
        return 0 if span is None else span[1] - span[0]

    def _build_refresh_spec(
        self,
        tracker: RequestTracker,
        load_spec: Optional[LoadSpec],
    ) -> Optional[RefreshSpec]:
        if (
            load_spec is None
            or not load_spec.can_load
            or not self.enable_blending
            or not bool(
                (self.config.extra_config or {}).get("joint_refresh", False)
            )
            or not (
                self.config.blend_mode == "codecsight"
                or getattr(self.config, "is_codecsight", False)
            )
            or not tracker.mm_positions
        ):
            return None

        extra_config = self.config.extra_config or {}
        policy = _configured_refresh_policy(extra_config)
        if policy == "multi_anchor":
            decisions, _ = _extract_costream_frame_decisions(
                tracker, list(tracker.mm_hashes or ())
            )
            if decisions is None:
                return None
            spans = _select_multi_anchor_refresh_spans(
                tracker.mm_positions,
                decisions,
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
            )
        else:
            span = _select_prefix_refresh_span(
                tracker.mm_positions,
                load_spec.lmcache_cached_tokens,
                int(extra_config.get("codecsight_refresh_frames", 3)),
                load_spec.vllm_cached_tokens,
            )
            spans = () if span is None else (RefreshSpan(*span),)
        if not spans:
            return None

        architectures = (
            getattr(
                self._vllm_config.model_config.hf_config,
                "architectures",
                [],
            )
            or []
        )
        position_mode = (
            "mrope_3d"
            if any("Qwen3VL" in name for name in architectures)
            else "rope_1d"
        )
        expected_retrieved_tokens = (
            load_spec.lmcache_cached_tokens
            - load_spec.vllm_cached_tokens
        )
        return RefreshSpec(
            request_id=tracker.req_id,
            model_id=self._vllm_config.model_config.model,
            cache_schema_version=str(
                extra_config.get(
                    "cache_schema_version", "codecsight-kv-v1",
                )
            ),
            policy=policy,
            position_mode=position_mode,
            cached_prefix_tokens=load_spec.lmcache_cached_tokens,
            expected_retrieved_tokens=expected_retrieved_tokens,
            spans=spans,
            source_hashes=tuple(tracker.mm_hashes or ()),
        )

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(
        self,
        request: "Request",
        num_external_tokens: int,
        blocks: Optional["KVCacheBlocks"] = None,
    ):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Allocation touched every mixed-prefix block. Drop the connector's
        # temporary ownership of newly allocated refresh blocks; request
        # ownership now keeps them live.
        blank_ids = getattr(
            self, "_resident_pending_blank_blocks", {}
        ).pop(request.request_id, ())
        if blank_ids:
            registry = self._resident_registry()
            if registry is None:
                raise RuntimeError("resident registry disappeared after allocation")
            registry.block_pool.free_blocks(
                registry.block_pool.get_blocks_by_id(blank_ids)
            )

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return
        logger.debug(f"num_external_tokens is {num_external_tokens}")
        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        # Only check for non-prompt-hit case
        if (
            self.load_specs[request.request_id].lmcache_cached_tokens
            != request.num_tokens
        ):
            assert (
                num_external_tokens > 0
                and num_external_tokens
                == self.load_specs[request.request_id].lmcache_cached_tokens
                - self.load_specs[request.request_id].vllm_cached_tokens
            ), (
                f"Mismatch in number of tokens: {num_external_tokens} vs "
                f"{self.load_specs[request.request_id].lmcache_cached_tokens} - "
                f"{self.load_specs[request.request_id].vllm_cached_tokens}"
                f" for request {request.request_id}"
            )

        self.load_specs[request.request_id].can_load = True

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for req_id in scheduler_output.num_scheduled_tokens:
            resident_spec = getattr(
                self, "_resident_pending_refresh_specs", {}
            ).pop(req_id, None)
            if resident_spec is not None:
                meta.resident_refresh_specs.append(resident_spec)
            resident_copies = getattr(
                self, "_resident_pending_copy_specs", {}
            ).pop(req_id, ())
            meta.resident_copy_specs.extend(resident_copies)

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
            )
            if req_meta is not None:
                req_meta.refresh_spec = self._build_refresh_spec(
                    request_tracker, load_spec,
                )
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.update(req.new_token_ids, req.new_block_ids)

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=None,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            resident_resume = (
                getattr(self, "_resident_zero_copy_enabled", False)
                and cached_reqs.resumed_from_preemption[i]
            )
            if request := self._unfinished_requests.get(req_id):
                if resident_resume:
                    # A resumed request receives a replacement block table.
                    request_tracker.token_ids = request.all_token_ids[
                        :cached_reqs.num_computed_tokens[i]
                    ].copy()
                    request_tracker.allocated_block_ids = []
                    request_tracker.num_saved_tokens = 0
                    request_tracker.is_decode_phase = False
                num_current_tokens = len(request_tracker.token_ids)
                new_token_ids = request.all_token_ids[
                    num_current_tokens : num_current_tokens + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            request_tracker.update(new_token_ids, new_block_ids)

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=None,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if (
            (self._resident_shadow_enabled or self._resident_zero_copy_enabled)
            and not _costream_force_full_compute(request)
        ):
            if self._resident_overlap_enabled:
                registry = self._resident_registry()
                if registry is not None:
                    active_hashes, _ = extract_mm_features(request)
                    stream_id = _costream_stream_id(request)
                    retired = registry.retain_kind_content_hashes(
                        "frame_token_slice", active_hashes,
                        stream_id=stream_id,
                    )
                    request.system_profile[
                        "resident_retired_window_frames"
                    ] = retired
            published_prefix_tokens = self._publish_resident_prompt(
                request, block_ids
            )
            published_text_prefix_tokens = (
                self._publish_resident_text_prefix(request, block_ids)
            )
            # Replace the previous whole-prompt pin before publishing the new
            # frame views. This avoids a transient two-window KV footprint and
            # needless watermark eviction churn.
            published, rejected = self._publish_resident_frames(
                request, block_ids
            )
            logger.info(
                "Resident KV publication request=%s mode=%s published=%d "
                "unaligned=%d prefix_tokens=%d text_prefix_tokens=%d "
                "registry=%s",
                request.request_id,
                (
                    "zero_copy_prototype"
                    if self._resident_zero_copy_enabled
                    else "shadow"
                ),
                published,
                rejected,
                published_prefix_tokens,
                published_text_prefix_tokens,
                (
                    self._resident_registry().stats()
                    if self._resident_registry() is not None
                    else None
                ),
            )

        if self._cacheblend_event_writeback:
            return True, return_params
        return False, return_params
