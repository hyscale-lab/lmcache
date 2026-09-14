# SPDX-License-Identifier: Apache-2.0

import threading

from lmcache.v1.cache_engine import _classify_layer_group
from lmcache.v1.storage_backend.storage_manager import StorageManager


class _StorageManager:
    def __init__(self, locations):
        self.locations = locations

    def contains(self, key):
        return self.locations.get(key)


class _AtomicStorageManager(_StorageManager):
    begin_layer_group_publication = StorageManager.begin_layer_group_publication
    commit_layer_group_publication = StorageManager.commit_layer_group_publication
    abort_layer_group_publication = StorageManager.abort_layer_group_publication
    layer_group_publication_state = StorageManager.layer_group_publication_state

    def __init__(self):
        super().__init__({})
        self.manager_lock = threading.RLock()
        self._layer_group_publication = {}


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


def test_concurrent_reader_cannot_observe_partial_writer():
    keys = ["l0", "l1", "l2"]
    manager = _AtomicStorageManager()
    first_layer_written = threading.Event()
    finish_writer = threading.Event()

    def writer():
        manager.begin_layer_group_publication(keys)
        manager.locations["l0"] = "LocalGPUBackend"
        first_layer_written.set()
        finish_writer.wait(timeout=2)
        manager.locations.update({
            "l1": "LocalGPUBackend", "l2": "LocalGPUBackend",
        })
        manager.commit_layer_group_publication(keys)

    thread = threading.Thread(target=writer)
    thread.start()
    assert first_layer_written.wait(timeout=2)
    assert _classify_layer_group(manager, keys)[0] == "absent"
    finish_writer.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert _classify_layer_group(manager, keys)[0] == "complete"


def test_interrupted_writer_stays_hidden_until_repaired():
    keys = ["l0", "l1", "l2"]
    manager = _AtomicStorageManager()
    manager.begin_layer_group_publication(keys)
    manager.locations["l0"] = "LocalGPUBackend"
    manager.abort_layer_group_publication(keys)

    assert _classify_layer_group(manager, keys)[0] == "absent"

    manager.begin_layer_group_publication(keys)
    manager.locations.update({
        key: "LocalGPUBackend" for key in keys
    })
    manager.commit_layer_group_publication(keys)
    assert _classify_layer_group(manager, keys)[0] == "complete"
