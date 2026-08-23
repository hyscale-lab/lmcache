# SPDX-License-Identifier: Apache-2.0

from lmcache.v1.cache_engine import _classify_layer_group


class _StorageManager:
    def __init__(self, locations):
        self.locations = locations

    def contains(self, key):
        return self.locations.get(key)


def test_layer_group_is_complete_only_when_every_layer_has_one_location():
    keys = ["l0", "l1", "l2"]
    manager = _StorageManager({key: "LocalGPUBackend" for key in keys})
    state, locations = _classify_layer_group(manager, keys)
    assert state == "complete"
    assert locations == ["LocalGPUBackend"] * 3


def test_layer_zero_alone_is_partial_not_complete():
    keys = ["l0", "l1", "l2"]
    manager = _StorageManager({"l0": "LocalGPUBackend"})
    state, locations = _classify_layer_group(manager, keys)
    assert state == "partial"
    assert locations == ["LocalGPUBackend", None, None]


def test_full_group_split_across_backends_requires_repair():
    keys = ["l0", "l1", "l2"]
    manager = _StorageManager({
        "l0": "LocalGPUBackend",
        "l1": "LocalGPUBackend",
        "l2": "LocalCPUBackend",
    })
    state, _ = _classify_layer_group(manager, keys)
    assert state == "partial"


def test_absent_group_stays_on_normal_store_path():
    keys = ["l0", "l1", "l2"]
    state, locations = _classify_layer_group(_StorageManager({}), keys)
    assert state == "absent"
    assert locations == [None, None, None]
