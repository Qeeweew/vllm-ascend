# SPDX-License-Identifier: Apache-2.0
"""Small registered-host NUMA/H2D/graph functionality; no performance claims."""

import ctypes
import os

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.engram_pinned_host import EngramPinnedHostTensor


def resident_nodes(tensor):
    page_size = os.sysconf("SC_PAGE_SIZE")
    pointer = tensor.data_ptr()
    addresses = list(range(pointer, pointer + tensor.numel() * tensor.element_size(), page_size))
    pages = (ctypes.c_void_p * len(addresses))(*addresses)
    status = (ctypes.c_int * len(addresses))()
    numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
    numa.move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    numa.move_pages.restype = ctypes.c_long
    assert numa.move_pages(0, len(addresses), pages, None, status, 0) == 0
    return set(status)


@pytest.fixture(scope="module")
def device():
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(0)
    return torch.device("npu:0")


def test_explicit_node_registration_nonblocking_h2d_and_owner_close(device):
    stream = torch.npu.Stream(device=device)
    destination = torch.empty((512, 256), dtype=torch.bfloat16, device=device)
    owner = EngramPinnedHostTensor(destination.shape, numa_node=0, device=device)
    source = owner.tensor
    try:
        assert source.is_pinned()
        assert resident_nodes(source) == {0}
        source.copy_(torch.arange(source.numel()).remainder(127).reshape_as(source))
        done = torch.npu.Event()
        with torch.npu.stream(stream):
            destination.copy_(source, non_blocking=True)
            done.record(stream)
        owner.record_event(done)
        owner.close()  # Must fence DMA before unregister, without a prior host wait.
        assert not source.is_pinned()
        torch.testing.assert_close(destination.cpu(), source, rtol=0, atol=0)
        assert resident_nodes(source) == {0}
    finally:
        owner.close()


def test_two_registered_host_slots_feed_twenty_changed_input_graph_replays(device):
    shape = (512, 256)
    owners = [EngramPinnedHostTensor(shape, numa_node=0, device=device) for _ in range(2)]
    addresses = [owner.tensor.data_ptr() for owner in owners]
    source = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(source)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output.copy_(source + 1)
    torch.npu.synchronize()
    compute = torch.npu.current_stream(device)
    copy = torch.npu.Stream(device=device)
    ready = [torch.npu.Event(), torch.npu.Event()]
    consumed = torch.npu.Event()
    consumed.record(compute)
    results = []
    try:
        for step in range(20):
            slot = step % 2
            if step >= 2:
                ready[slot].synchronize()  # Refill only after the prior DMA read.
            owners[slot].tensor.fill_(step)
            assert owners[slot].tensor.data_ptr() == addresses[slot]
            with torch.npu.stream(copy):
                copy.wait_event(consumed)
                source.copy_(owners[slot].tensor, non_blocking=True)
                ready[slot].record(copy)
            owners[slot].record_event(ready[slot])
            compute.wait_event(ready[slot])
            graph.replay()
            results.append(output.clone())
            consumed.record(compute)
        # Close can precede graph completion: only DMA still reads host storage.
        for owner in owners:
            owner.close()
        torch.npu.synchronize()
        for step, value in enumerate(results):
            torch.testing.assert_close(value.cpu(), torch.full(shape, step + 1, dtype=torch.bfloat16), rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        for owner in owners:
            owner.close()
