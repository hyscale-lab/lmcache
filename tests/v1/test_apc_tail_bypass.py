from types import SimpleNamespace
from unittest.mock import Mock

from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
)


def make_connector(hit_tokens):
    connector = object.__new__(LMCacheConnectorV1Impl)
    connector.kv_role = "kv_both"
    connector.lookup_client = SimpleNamespace(
        lookup=Mock(return_value=hit_tokens),
    )
    connector._requests_priority = {}
    connector._kv_diag_force_miss = False
    connector.skip_last_n_tokens = 0
    connector._block_size = 16
    connector.load_specs = {}
    return connector


def make_request():
    return SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(256)),
        num_tokens=256,
        sampling_params=SimpleNamespace(extra_args=None),
        mm_features=[],
        priority=0,
        system_profile={},
    )


def test_small_apc_tail_is_prefilled_with_the_suffix():
    connector = make_connector(110)
    request = make_request()

    assert connector.get_num_new_matched_tokens(request, 100) == 0
    assert request.system_profile["lmcache_bypassed_tail_tokens"] == 10


def test_large_external_extension_still_uses_lmcache():
    connector = make_connector(180)
    request = make_request()

    assert connector.get_num_new_matched_tokens(request, 16) == 164
    assert "lmcache_bypassed_tail_tokens" not in request.system_profile
