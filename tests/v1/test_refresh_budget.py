from types import SimpleNamespace

from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    _joint_refresh_selection_stats,
    _select_multi_anchor_refresh_spans,
)


def make_connector(
    blend_mode="codecsight",
    refresh_frames=2,
    is_codecsight=False,
    joint_refresh=True,
    refresh_policy="prefix",
):
    connector = object.__new__(LMCacheConnectorV1Impl)
    connector.enable_blending = True
    connector.config = SimpleNamespace(
        blend_mode=blend_mode,
        is_codecsight=is_codecsight,
        extra_config={
            "codecsight_refresh_frames": refresh_frames,
            "joint_refresh": joint_refresh,
            "codecsight_refresh_policy": refresh_policy,
        },
    )
    connector.load_specs = {
        "request-0": LoadSpec(0, 500, False),
    }
    connector._vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="test/model",
            hf_config=SimpleNamespace(
                architectures=["LlavaOnevisionForConditionalGeneration"],
            ),
        ),
    )
    return connector


def make_request():
    return SimpleNamespace(
        request_id="request-0",
        sampling_params=SimpleNamespace(extra_args=None),
        mm_hashes=["a", "b", "c"],
        mm_positions=[
            SimpleNamespace(offset=32, length=196),
            SimpleNamespace(offset=229, length=196),
            SimpleNamespace(offset=426, length=196),
        ],
    )


def multi_anchor_payload():
    return [{
        "request_frame_index": index,
        "frame_id": str(index),
        "gop_id": 0 if index < 2 else 1,
        "frame_type": "I" if index in (0, 2) else "P",
        "is_anchor": index in (0, 2),
        "anchor_frame_index": 0 if index < 2 else 2,
        "action": "refresh" if index in (0, 2) else "keep",
    } for index in range(3)]


def test_multi_anchor_selector_uses_every_i_frame_and_clamps_prefix():
    request = make_request()
    decisions = [
        SimpleNamespace(is_anchor=index in (0, 2), action=(
            "refresh" if index in (0, 2) else "keep"
        ))
        for index in range(3)
    ]

    spans = _select_multi_anchor_refresh_spans(
        request.mm_positions, decisions, 500
    )

    assert [(span.start, span.end, span.source_index) for span in spans] == [
        (32, 228, 0),
        (426, 500, 2),
    ]


def test_multi_anchor_budget_and_spec_follow_client_gop_metadata():
    connector = make_connector(refresh_policy="multi_anchor")
    load_spec = connector.load_specs["request-0"]
    load_spec.can_load = True
    request = make_request()
    params = {"costream.frame_decisions": multi_anchor_payload()}
    request.sampling_params.extra_args = {
        "kv_transfer_params": params,
    }
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=600,
        token_ids=[0] * 500,
        allocated_block_ids=list(range(32)),
        mm_hashes=request.mm_hashes,
        mm_positions=request.mm_positions,
        request_configs=params,
    )

    assert connector.get_num_kv_refresh_tokens(request, 500) == 270
    spec = connector._build_refresh_spec(tracker, load_spec)
    assert spec is not None
    assert spec.policy == "multi_anchor"
    assert [(span.start, span.end, span.source_index) for span in spec.spans] == [
        (32, 228, 0),
        (426, 500, 2),
    ]
    assert spec.num_refresh_tokens == 270


def test_prefix_refresh_budget_covers_selected_visual_span():
    connector = make_connector()

    assert connector.get_num_kv_refresh_tokens(make_request(), 400) == 393


def test_refresh_budget_is_clamped_to_cached_prefix():
    connector = make_connector(refresh_frames=3)
    connector.load_specs["request-0"].lmcache_cached_tokens = 480

    assert connector.get_num_kv_refresh_tokens(make_request(), 400) == 448


def test_refresh_budget_skips_frames_covered_by_vllm_cache():
    connector = make_connector(refresh_frames=2)
    connector.load_specs["request-0"] = LoadSpec(250, 500, False)

    assert connector.get_num_kv_refresh_tokens(make_request(), 250) == 250


def test_direct_reuse_has_no_refresh_budget():
    connector = make_connector(blend_mode="direct_reuse")

    assert connector.get_num_kv_refresh_tokens(make_request(), 400) == 0


def test_eager_blending_has_no_scheduler_refresh_budget_or_spec():
    connector = make_connector(joint_refresh=False)
    load_spec = connector.load_specs["request-0"]
    load_spec.can_load = True
    request = make_request()
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=600,
        token_ids=[0] * 500,
        allocated_block_ids=list(range(32)),
        mm_hashes=request.mm_hashes,
        mm_positions=request.mm_positions,
    )

    assert connector.get_num_kv_refresh_tokens(request, 400) == 0
    assert connector._build_refresh_spec(tracker, load_spec) is None


def test_legacy_codecsight_flag_enables_refresh_budget_and_spec():
    connector = make_connector(blend_mode="", is_codecsight=True)
    load_spec = connector.load_specs["request-0"]
    load_spec.can_load = True
    request = make_request()
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=600,
        token_ids=[0] * 500,
        allocated_block_ids=list(range(32)),
        mm_hashes=request.mm_hashes,
        mm_positions=request.mm_positions,
    )

    assert connector.get_num_kv_refresh_tokens(request, 400) == 393
    assert connector._build_refresh_spec(tracker, load_spec) is not None


def test_refresh_spec_uses_same_span_as_scheduler_budget():
    connector = make_connector()
    load_spec = connector.load_specs["request-0"]
    load_spec.can_load = True
    request = make_request()
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=600,
        token_ids=[0] * 500,
        allocated_block_ids=list(range(32)),
        mm_hashes=request.mm_hashes,
        mm_positions=request.mm_positions,
    )

    spec = connector._build_refresh_spec(tracker, load_spec)

    assert spec is not None
    assert spec.spans[0].start == 32
    assert spec.spans[0].end == 425
    assert spec.num_refresh_tokens == connector.get_num_kv_refresh_tokens(
        request, 400,
    )


def test_joint_refresh_reports_visual_candidate_and_selected_budget():
    connector = make_connector()
    load_spec = connector.load_specs["request-0"]
    load_spec.can_load = True
    request = make_request()
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=600,
        token_ids=[0] * 500,
        allocated_block_ids=list(range(32)),
        mm_hashes=request.mm_hashes,
        mm_positions=request.mm_positions,
    )
    spec = connector._build_refresh_spec(tracker, load_spec)

    assert spec is not None
    assert _joint_refresh_selection_stats(spec, request.mm_positions) == {
        "recompute_selection_mode": "joint_prefix_refresh",
        "recompute_candidate_tokens": 500,
        "recompute_selected_tokens": 393,
        "recompute_candidate_visual_tokens": 466,
        "recompute_selected_visual_tokens": 392,
        "recompute_candidate_visual_indices": [0, 1, 2],
        "recompute_selected_visual_indices": [0, 1],
        "recompute_refresh_spans": [[32, 425]],
        "recompute_visual_ratio": 392 / 466,
    }


def test_qwen_refresh_spec_selects_mrope_positions():
    connector = make_connector()
    connector._vllm_config.model_config.hf_config.architectures = [
        "Qwen3VLForConditionalGeneration",
    ]
    load_spec = connector.load_specs["request-0"]
    load_spec.can_load = True
    request = make_request()
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=600,
        token_ids=[0] * 500,
        allocated_block_ids=list(range(32)),
        mm_hashes=request.mm_hashes,
        mm_positions=request.mm_positions,
    )

    spec = connector._build_refresh_spec(tracker, load_spec)

    assert spec is not None
    assert spec.position_mode == "mrope_3d"


def test_partial_chunk_hit_writes_suffix_when_unfull_chunks_are_enabled():
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=3915,
        token_ids=[0] * 3915,
        allocated_block_ids=list(range(245)),
        num_saved_tokens=2958,
    )
    metadata = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        lmcache_chunk_size=4096,
        load_spec=LoadSpec(0, 2958, True),
        discard_partial_chunks=False,
    )

    assert metadata is not None
    assert metadata.save_spec is not None
    assert metadata.save_spec.can_save
    assert metadata.save_spec.skip_leading_tokens == 2958
    assert len(metadata.token_ids) == 3915
    assert tracker.num_saved_tokens == 3915


def test_non_aligned_load_prefix_is_not_truncated_to_store_chunk():
    tracker = RequestTracker(
        req_id="request-0",
        prompt_len=16811,
        token_ids=[0] * 16811,
        allocated_block_ids=list(range(1051)),
    )
    metadata = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        lmcache_chunk_size=1024,
        load_spec=LoadSpec(32, 16485, True),
        discard_partial_chunks=True,
    )

    assert metadata is not None
    assert len(metadata.token_ids) == 16485
    assert len(metadata.slot_mapping) == 16485
    assert metadata.save_token_count == 16384
    assert tracker.num_saved_tokens == 16384
