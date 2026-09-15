# SPDX-License-Identifier: Apache-2.0
"""Real CPU mmap/frombuffer lifetime tests with a recorded CANN/NUMA boundary."""

import ctypes
import gc
import os
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import engram_pinned_host as module


@pytest.fixture
def api(monkeypatch):
    calls = []
    state = SimpleNamespace(registered=False, calls=calls, pointer=None, size=None)

    def bind(pointer, size, node):
        calls.append(("bind", node))
        state.pointer, state.size = pointer, size

    def register(pointer, size):
        assert pointer == state.pointer and size == state.size
        assert ctypes.string_at(pointer, size) == bytes(size)
        calls.append(("register",))
        state.registered = True

    def unregister(pointer):
        assert pointer == state.pointer
        calls.append(("unregister",))
        state.registered = False

    state.bind = Mock(side_effect=bind)
    state.register = Mock(side_effect=register)
    state.unregister = Mock(side_effect=unregister)
    monkeypatch.setattr(module, "_HostMemoryAPI", lambda: state)
    monkeypatch.setattr(torch.npu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: state.registered)
    return state


def owner():
    return module.EngramPinnedHostTensor((3, 5), numa_node=2, device=torch.device("npu:7"))


def test_binds_and_first_touches_before_registration_with_page_aligned_size(api):
    with owner() as value:
        assert value.tensor.shape == (3, 5)
        assert value.tensor.dtype == torch.bfloat16
        assert value.tensor.device.type == "cpu"
        assert value.tensor.is_contiguous() and value.tensor.is_pinned()
        assert value.tensor.data_ptr() == api.pointer
        assert api.pointer % os.sysconf("SC_PAGE_SIZE") == 0
        assert api.size == os.sysconf("SC_PAGE_SIZE")
        assert api.calls == [("bind", 2), ("register",)]
    assert api.calls[-1] == ("unregister",)
    assert not api.registered
    value.close()
    assert api.unregister.call_count == 1


def test_close_waits_dma_events_before_unregister_and_deduplicates_ring_events(api):
    value = owner()
    first = Mock(synchronize=lambda: api.calls.append(("wait_first",)))
    second = Mock(synchronize=lambda: api.calls.append(("wait_second",)))
    value.record_event(first)
    value.record_event(second)
    value.record_event(first)
    value.close()
    assert api.calls[-3:] == [("wait_first",), ("wait_second",), ("unregister",)]
    with pytest.raises(RuntimeError, match="closed"):
        _ = value.tensor
    with pytest.raises(RuntimeError, match="closed"):
        value.record_event(first)


def test_close_preserves_tensor_alias_storage_without_leaving_it_registered(api):
    value = owner()
    alias = value.tensor.detach().flatten()[2:7]
    mapping = weakref.ref(value._mapping)
    alias.fill_(13)
    value.close()
    gc.collect()
    assert mapping() is not None
    assert not alias.is_pinned()
    torch.testing.assert_close(alias, torch.full((5,), 13, dtype=torch.bfloat16))
    del alias
    gc.collect()
    assert mapping() is None
    assert api.unregister.call_count == 1


def test_unclosed_owner_registration_lives_until_last_tensor_storage_alias(api):
    value = owner()
    alias = value.tensor.detach()
    mapping = weakref.ref(value._mapping)
    del value
    gc.collect()
    assert mapping() is not None and api.registered
    assert api.unregister.call_count == 0
    del alias
    gc.collect()
    assert mapping() is None
    assert api.unregister.call_count == 1


def test_event_failure_keeps_owner_and_registration_alive_for_retry(api):
    value = owner()
    event = Mock()
    event.synchronize.side_effect = RuntimeError("DMA event failure")
    value.record_event(event)
    with pytest.raises(RuntimeError, match="DMA event failure"):
        value.close()
    assert value.tensor.is_pinned()
    api.unregister.assert_not_called()
    event.synchronize.side_effect = None
    value.close()
    assert api.unregister.call_count == 1


def test_unregister_failure_keeps_storage_for_retry(api):
    value = owner()
    unregister = api.unregister.side_effect
    api.unregister.side_effect = RuntimeError("unregister error")
    with pytest.raises(RuntimeError, match="unregister error"):
        value.close()
    assert value.tensor.is_pinned()
    api.unregister.side_effect = unregister
    value.close()
    assert not api.registered


def test_bind_failure_does_not_register_or_fallback(api):
    api.bind.side_effect = OSError("NUMA policy rejected")
    with pytest.raises(OSError, match="NUMA policy rejected"):
        owner()
    api.register.assert_not_called()
    api.unregister.assert_not_called()


def test_registration_failure_does_not_publish_storage_or_unregister_unowned_memory(api):
    api.register.side_effect = RuntimeError("registration rejected")
    with pytest.raises(RuntimeError, match="registration rejected"):
        owner()
    api.unregister.assert_not_called()
    assert not api.registered


def test_torch_pin_detection_failure_unregisters_owned_mapping(api, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: False)
    with pytest.raises(RuntimeError, match="not recognized as pinned"):
        owner()
    assert api.unregister.call_count == 1
    assert not api.registered


def test_graph_capture_rejected_before_allocating(api, monkeypatch):
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        owner()
    api.bind.assert_not_called()


@pytest.mark.parametrize(
    "shape,node,device",
    [
        ((), 0, "npu:7"),
        ((0, 4), 0, "npu:7"),
        ((2.5,), 0, "npu:7"),
        ((2,), -1, "npu:7"),
        ((2,), 0, "cpu"),
        ((2,), 0, "npu"),
    ],
)
def test_invalid_geometry_or_device_rejected_before_allocation(api, shape, node, device):
    with pytest.raises(ValueError):
        module.EngramPinnedHostTensor(shape, numa_node=node, device=torch.device(device))
    api.bind.assert_not_called()


def test_streaming_initializer_populates_final_storage_before_register(api):
    def register(pointer, size):
        api.calls.append(("register",))
        api.registered = True

    api.register.side_effect = register

    def initialize(tensor):
        assert not tensor.is_pinned()
        api.calls.append(("initialize",))
        tensor.fill_(9)

    with module.EngramPinnedHostTensor(
        (7, 3), numa_node=2, device=torch.device("npu:7"), initialize=initialize
    ) as value:
        assert api.calls == [("bind", 2), ("initialize",), ("register",)]
        torch.testing.assert_close(value.tensor, torch.full((7, 3), 9, dtype=torch.bfloat16))


def test_initializer_failure_never_registers_partial_storage(api):
    def initialize(tensor):
        tensor[0].fill_(1)
        raise ValueError("checkpoint read failed")

    with pytest.raises(ValueError, match="checkpoint read failed"):
        module.EngramPinnedHostTensor((7, 3), numa_node=2, device=torch.device("npu:7"), initialize=initialize)
    api.register.assert_not_called()
    api.unregister.assert_not_called()
