from types import SimpleNamespace

import torch

from lmcache.integration.vllm.kv_diagnostics import KVDiagnostic


def test_gather_two_major_paged_kv():
    cache = torch.arange(2 * 3 * 4 * 2 * 2).reshape(2, 3, 4, 2, 2)
    slots = torch.tensor([0, 5, 11])
    actual = KVDiagnostic._gather(cache, slots)
    expected = torch.stack((cache[:, 0, 0], cache[:, 1, 1], cache[:, 2, 3]), dim=1)
    torch.testing.assert_close(actual, expected.flatten(start_dim=2))


def test_visual_sampling_is_bounded_and_labeled(monkeypatch, tmp_path):
    monkeypatch.setenv("LMCACHE_KV_DIAG_DIR", str(tmp_path))
    monkeypatch.setenv("LMCACHE_KV_DIAG_TOKENS_PER_FRAME", "3")
    diagnostic = KVDiagnostic(num_layers=4, rank=0)
    request = SimpleNamespace(
        mm_positions=[
            SimpleNamespace(offset=4, length=8),
            SimpleNamespace(offset=14, length=8),
        ]
    )
    positions, frames, kinds = diagnostic._sample_positions(request, 18)
    assert positions == sorted(set(positions))
    assert all(0 <= position < 18 for position in positions)
    assert sum(frame == 0 for frame in frames) == 3
    assert sum(frame == 1 for frame in frames) == 3
    assert kinds[positions.index(4)] == "visual"
