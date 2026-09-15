# SPDX-License-Identifier: Apache-2.0
"""Selected-head streaming into real mmap, with mocked driver registration only."""

import ctypes
import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.ops import engram_pinned_host as pinned
from vllm_ascend.ops.engram_offload import EngramOffloadManager, EngramTableShard
from vllm_ascend.worker.engram_runtime import EngramRuntime


@pytest.fixture
def checkpoint(tmp_path):
    full = torch.arange(13 * 4).reshape(13, 4).bfloat16()
    path = tmp_path / "table.safetensors"
    save_file({"embedding": full}, path)
    return path, full


@pytest.fixture
def driver(monkeypatch):
    state = SimpleNamespace(registered=False, calls=[], snapshots=[])

    def register(pointer, size):
        state.snapshots.append(ctypes.string_at(pointer, 8 * 4 * 2))
        state.calls.append("register")
        state.registered = True

    def unregister(pointer):
        state.calls.append("unregister")
        state.registered = False

    state.bind = Mock(side_effect=lambda *args: state.calls.append("bind"))
    state.register = Mock(side_effect=register)
    state.unregister = Mock(side_effect=unregister)
    monkeypatch.setattr(pinned, "_HostMemoryAPI", lambda: state)
    monkeypatch.setattr(torch.npu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: state.registered)
    return state


def load(path, **kwargs):
    return EngramTableShard.from_safetensors(
        path,
        "embedding",
        [0, 2],
        [(0, 3), (8, 13)],
        rows_per_copy=2,
        numa_node=2,
        device=torch.device("npu:7"),
        **kwargs,
    )


def test_selected_chunks_populate_final_storage_before_registration(checkpoint, driver):
    path, full = checkpoint
    shard = load(path)
    expected = torch.cat((full[:3], full[8:]))
    assert driver.snapshots == [expected.view(torch.uint16).numpy().tobytes()]
    torch.testing.assert_close(shard.weight, expected, rtol=0, atol=0)
    assert driver.calls == ["bind", "register"]
    assert shard._pinned_owner.tensor.data_ptr() == shard.weight.data_ptr()
    mapping = weakref.ref(shard._pinned_owner._mapping)
    shard.close()
    gc.collect()
    assert mapping() is None and shard.weight is None
    assert driver.calls[-1] == "unregister"
    shard.close()
    assert driver.unregister.call_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        shard.gather_into(torch.tensor([[0, 3, 8]]), torch.empty((1, 2, 4), dtype=torch.bfloat16))


def test_manager_close_fences_all_device_use_before_releasing_shard(checkpoint, driver):
    path, _ = checkpoint
    shard = load(path)
    manager = EngramOffloadManager.__new__(EngramOffloadManager)
    manager.shards = (shard,)
    manager._closed = False
    manager._prepared = False
    manager._copy_stream = Mock(synchronize=lambda: driver.calls.append("copy_fence"))
    manager._consumed = Mock(synchronize=lambda: driver.calls.append("graph_fence"))
    manager.close()
    assert driver.calls[-3:] == ["copy_fence", "graph_fence", "unregister"]
    assert shard.weight is None and manager._closed
    manager.close()
    assert driver.unregister.call_count == 1


def test_manager_fence_failure_retains_registered_storage(checkpoint, driver):
    path, _ = checkpoint
    shard = load(path)
    manager = EngramOffloadManager.__new__(EngramOffloadManager)
    manager.shards = (shard,)
    manager._closed = manager._prepared = False
    manager._copy_stream = Mock()
    manager._copy_stream.synchronize.side_effect = RuntimeError("copy fence failed")
    manager._consumed = Mock()
    with pytest.raises(RuntimeError, match="copy fence failed"):
        manager.close()
    assert driver.registered and not manager._closed
    manager._consumed.synchronize.assert_not_called()
    driver.unregister.assert_not_called()
    manager._copy_stream.synchronize.side_effect = None
    manager.close()
    assert not driver.registered


def pending_runtime(checkpoint, driver):
    shard = load(checkpoint[0])
    manager = EngramOffloadManager.__new__(EngramOffloadManager)
    manager.shards = (shard,)
    manager.device = torch.device("npu:7")
    manager._closed = False
    manager._prepared = manager._waited = True
    manager._copy_stream = Mock(synchronize=Mock(side_effect=lambda: driver.calls.append("copy_fence")))
    manager._consumed = Mock(synchronize=Mock(side_effect=lambda: driver.calls.append("old_graph_fence")))
    runtime = EngramRuntime.__new__(EngramRuntime)
    runtime.offload = manager
    runtime._closed = False
    runtime._prepared = True
    return runtime, shard


@pytest.mark.parametrize("waited", [False, True])
def test_shutdown_fences_all_streams_before_abandoning_pending_dma_or_forward(checkpoint, driver, monkeypatch, waited):
    runtime, shard = pending_runtime(checkpoint, driver)
    runtime.offload._waited = waited
    synchronize = Mock(side_effect=lambda device: driver.calls.append("all_streams_fence"))
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    # Normal close still rejects even if wait_ready happened before failure.
    with pytest.raises(RuntimeError, match="consumed"):
        runtime.close()
    driver.unregister.assert_not_called()
    runtime.shutdown()
    assert driver.calls[-4:] == ["all_streams_fence", "copy_fence", "old_graph_fence", "unregister"]
    synchronize.assert_called_once_with(torch.device("npu:7"))
    assert runtime._closed and not runtime._prepared
    assert runtime.offload._closed and not runtime.offload._prepared and not runtime.offload._waited
    assert shard.weight is None
    runtime.shutdown()
    assert synchronize.call_count == driver.unregister.call_count == 1


@pytest.mark.parametrize("failure", ["device_fence", "copy_fence", "unregister"])
def test_shutdown_failure_remains_visible_and_retry_retains_owner(checkpoint, driver, monkeypatch, failure):
    runtime, shard = pending_runtime(checkpoint, driver)
    synchronize = Mock()
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    failing = {
        "device_fence": synchronize,
        "copy_fence": runtime.offload._copy_stream.synchronize,
        "unregister": driver.unregister,
    }[failure]
    original_effect = failing.side_effect
    failing.side_effect = RuntimeError(failure)
    with pytest.raises(RuntimeError, match=failure):
        runtime.shutdown()
    assert driver.registered and shard.weight is not None
    assert not runtime._closed and not runtime.offload._closed
    assert runtime._prepared  # Runtime state is committed only after successful cleanup.
    if failure == "device_fence":
        assert runtime.offload._prepared
        runtime.offload._copy_stream.synchronize.assert_not_called()
        driver.unregister.assert_not_called()
    elif failure == "copy_fence":
        driver.unregister.assert_not_called()
    failing.side_effect = original_effect
    runtime.shutdown()
    assert runtime._closed and runtime.offload._closed and not driver.registered


def test_shutdown_cannot_unregister_during_graph_capture(checkpoint, driver, monkeypatch):
    runtime, _ = pending_runtime(checkpoint, driver)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    synchronize = Mock()
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        runtime.shutdown()
    synchronize.assert_not_called()
    driver.unregister.assert_not_called()
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    runtime.shutdown()


def test_copy_failure_never_registers_partial_table(checkpoint, driver, monkeypatch):
    path, _ = checkpoint
    original = torch.Tensor.copy_
    calls = []

    def copy(target, source, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("checkpoint read failed")
        return original(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(OSError, match="checkpoint read failed"):
        load(path)
    driver.register.assert_not_called()
    driver.unregister.assert_not_called()


def test_post_registration_shard_validation_failure_releases_owner(checkpoint, driver):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="unique"):
        EngramTableShard.from_safetensors(
            path,
            "embedding",
            [0, 0],
            [(0, 3), (8, 13)],
            rows_per_copy=2,
            numa_node=2,
            device=torch.device("npu:7"),
        )
    assert driver.calls == ["bind", "register", "unregister"]
    assert not driver.registered


@pytest.mark.parametrize(
    "kwargs",
    [
        {"numa_node": 2},
        {"device": torch.device("npu:7")},
        {"numa_node": 2, "device": torch.device("npu:7"), "pin_memory": False},
    ],
)
def test_incomplete_or_contradictory_numa_options_fail_before_file_access(driver, kwargs):
    with pytest.raises(ValueError, match="NUMA"):
        EngramTableShard.from_safetensors("missing.safetensors", "embedding", [0], [(0, 3)], **kwargs)
    driver.bind.assert_not_called()
