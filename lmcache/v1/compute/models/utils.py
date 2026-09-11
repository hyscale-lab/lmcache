# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Callable, Dict, Mapping

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def patch_vllm_dummy_video_inputs() -> bool:
    """Make vLLM dummy video inputs compatible with image processors."""
    try:
        from vllm.multimodal import profiling as mm_profiling
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to import vllm.multimodal.profiling for dummy video patch: %s", exc
        )
        return False

    orig_fn = getattr(mm_profiling.BaseDummyInputsBuilder, "_get_dummy_videos", None)
    if orig_fn is None:
        logger.warning("BaseDummyInputsBuilder has no _get_dummy_videos; skip patch.")
        return False

    if getattr(orig_fn, "_lmcache_patched", False):
        return True

    import numpy as np

    def _get_dummy_videos(self, *, width, height, num_frames, num_videos):
        if num_videos == 0:
            return []
        video = np.full((num_frames, height, width, 3), 255, dtype=np.uint8)
        return [video] * num_videos

    _get_dummy_videos._lmcache_patched = True  # type: ignore[attr-defined]
    mm_profiling.BaseDummyInputsBuilder._get_dummy_videos = _get_dummy_videos
    logger.info("Patched vLLM dummy video generator to return uint8 arrays.")
    return True

# Install before vLLM runs multimodal profiling.
try:
    patch_vllm_dummy_video_inputs()
except Exception as exc:  # pragma: no cover - defensive
    logger.debug("Dummy video patch failed to apply on import: %s", exc)


def infer_model_from_vllm(vllm_model, blender, enable_sparse: bool = False):
    model_name = type(vllm_model).__name__
    if model_name == "LlamaForCausalLM":
        # First Party
        from lmcache.v1.compute.models.llama import LMCLlamaModel

        return LMCLlamaModel(vllm_model, blender, enable_sparse)
    elif model_name in {
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        # LLaVA-OneVision uses a Qwen2 decoder.
        "LlavaOnevisionForConditionalGeneration",
    }:
        # First Party
        from lmcache.v1.compute.models.qwen2vl import LMCQwen2VLModel

        return LMCQwen2VLModel(vllm_model, blender, enable_sparse)
    elif model_name == "Qwen3ForCausalLM":
        # First Party
        from lmcache.v1.compute.models.qwen3 import LMCQwen3Model

        return LMCQwen3Model(vllm_model, blender, enable_sparse)
    elif model_name == "InternVLChatModel":
        # First Party
        from lmcache.v1.compute.models.internvl import LMCInternVLModel

        return LMCInternVLModel(vllm_model, blender, enable_sparse)
    elif model_name == "Qwen3VLForConditionalGeneration":
        # First Party
        from lmcache.v1.compute.models.qwen3vl import LMCQwen3VLModel

        return LMCQwen3VLModel(vllm_model, blender, enable_sparse)
    else:
        # TODO(Jiayi): Add support for more models
        raise NotImplementedError(
            f"Model type {model_name} is not supported in LMCache."
        )


class VLLMModelTracker:
    _vllm_models: Dict[str, nn.Module] = {}
    _encoder_caches: Dict[str, dict] = {}
    _encoder_position_caches: Dict[str, dict] = {}
    _encoder_recompute_callbacks: Dict[
        str, Callable[..., Mapping[str, Any]]
    ] = {}

    @classmethod
    def register_model(
        cls,
        instance_id: str,
        vllm_model: nn.Module,
    ):
        """
        Register a vllm model by instance_id.
        """
        logger.info(f"Registering vllm model for {instance_id}")
        if instance_id not in cls._vllm_models:
            cls._vllm_models[instance_id] = vllm_model
        else:
            logger.warning(
                f"vllm model for {instance_id} already registered, doing nothing."
            )

    @classmethod
    def register_encoder_cache(
        cls,
        instance_id: str,
        encoder_cache: dict,
    ):
        cls._encoder_caches[instance_id] = encoder_cache
        logger.info("Registered encoder_cache for %s", instance_id)

    @classmethod
    def get_encoder_cache(cls, instance_id: str):
        return cls._encoder_caches.get(instance_id)

    @classmethod
    def register_encoder_position_cache(
        cls,
        instance_id: str,
        encoder_position_cache: dict,
    ):
        cls._encoder_position_caches[instance_id] = encoder_position_cache
        logger.info("Registered encoder position cache for %s", instance_id)

    @classmethod
    def get_encoder_position_cache(cls, instance_id: str):
        return cls._encoder_position_caches.get(instance_id)

    @classmethod
    def register_encoder_recompute_callback(
        cls,
        instance_id: str,
        callback: Callable[..., Mapping[str, Any]],
    ) -> None:
        """Register a worker callback for transient encoder re-computation.

        The ordinary vLLM encoder cache is capacity bounded. Selective KV
        refresh still needs visual embeddings for a cached LMCache prefix even
        after those embeddings have been evicted. The callback re-encodes only
        the missing visual items without pretending they were cache hits.
        """
        cls._encoder_recompute_callbacks[instance_id] = callback
        logger.info(
            "Registered encoder recompute callback for %s", instance_id
        )

    @classmethod
    def get_encoder_recompute_callback(cls, instance_id: str):
        return cls._encoder_recompute_callbacks.get(instance_id)

    @classmethod
    def get_model(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the vllm model by instance_id.
        """
        if instance_id not in cls._vllm_models:
            raise ValueError(f"vllm model for {instance_id} not found.")
        return cls._vllm_models[instance_id]
