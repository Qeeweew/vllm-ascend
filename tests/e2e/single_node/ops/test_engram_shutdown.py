# SPDX-License-Identifier: Apache-2.0
"""Small real registered-owner cleanup with pending H2D or graph consumption."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.engram_offload import EngramOffloadManager, EngramTableShard
from vllm_ascend.ops.engram_pinned_host import EngramPinnedHostTensor
from vllm_ascend.worker.engram_runtime import EngramRuntime


@torch.inference_mode()
@pytest.mark.parametrize("pending_forward", [False, True])
def test_shutdown_real_registration_after_unconsumed_step(pending_forward):
    torch.npu.set_device(2)
    device = torch.device("npu:2")
    owner = EngramPinnedHostTensor((16, 256), numa_node=4, device=device, initialize=lambda table: table.fill_(7))
    mapping = owner._mapping
    shard = EngramTableShard(owner.tensor, [0], [(0, 16)])
    shard._pinned_owner = owner
    manager = EngramOffloadManager([shard], 8, device)
    # History is unused: this test isolates checked runtime/manager shutdown.
    history = SimpleNamespace(hasher=SimpleNamespace(layout=SimpleNamespace(layer_ids=[1])))
    runtime = EngramRuntime(history, manager)
    ids = torch.tensor([[0], [15]], dtype=torch.int64)
    try:
        manager.prepare([ids], 8)
        manager.wait_ready()
        for _ in range(3):
            manager.device_rows[0] + 1
        torch.npu.synchronize(device)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=torch.npu.Stream(device=device)):
            output = manager.device_rows[0] + 1
        manager.mark_consumed()

        # Deliberately omit consumption after the final prepare. This models
        # termination during pending H2D or an exception after graph launch.
        manager.prepare([ids.flip(0)], 8)
        runtime._prepared = True
        snapshot = None
        if pending_forward:
            manager.wait_ready()
            graph.replay()
            snapshot = output.clone()
        assert mapping.registered and shard.weight.is_pinned()
        with pytest.raises(RuntimeError, match="consumed"):
            runtime.close()
        runtime.shutdown()
        assert runtime._closed and not runtime._prepared
        assert manager._closed and not manager._prepared
        assert not mapping.registered and shard.weight is None
        if snapshot is not None:
            expected = torch.ones((8, 1, 256), dtype=torch.bfloat16)
            expected[:2] = 8
            torch.testing.assert_close(snapshot.cpu(), expected, rtol=0, atol=0)
        runtime.shutdown()
    finally:
        # Checked cleanup also runs if an assertion fails; errors remain visible.
        runtime.shutdown()
