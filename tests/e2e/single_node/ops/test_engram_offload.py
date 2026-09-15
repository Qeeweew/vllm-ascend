# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-address offload/replay ordering with changing token IDs and padding."""

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors.torch import save_file

from vllm_ascend.ops.engram_offload import EngramOffloadManager, EngramTableShard


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("numa_storage", [False, True])
def test_inflight_replay_uses_current_rows_and_zero_padding(graph_mode, numa_storage, tmp_path):
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    full = torch.arange(13 * 256).reshape(13, 256).remainder(127).bfloat16()
    if numa_storage:
        path = tmp_path / "registered-table.safetensors"
        save_file({"embedding": full}, path)
        shard = EngramTableShard.from_safetensors(
            path,
            "embedding",
            [0, 2],
            [(0, 3), (8, 13)],
            rows_per_copy=2,
            numa_node=0,
            device=device,
        )
        assert shard.weight.is_pinned()
        torch.testing.assert_close(shard.weight, torch.cat((full[:3], full[8:])), rtol=0, atol=0)
    else:
        shard = EngramTableShard(torch.cat((full[:3], full[8:])), [0, 2], [(0, 3), (8, 13)])
    manager = EngramOffloadManager([shard, shard], 8, device)
    ptrs = tuple(rows.data_ptr() for rows in manager.device_rows)
    graphs, graph_outputs = {}, {}
    if graph_mode:
        for bucket in (4, 8):
            manager.prepare([torch.tensor([[0, 3, 8]])] * 2, bucket)
            manager.wait_ready()
            # Warmup and capture consume the same fixed device addresses.
            for _ in range(3):
                manager.device_rows[0][:bucket] + manager.device_rows[1][:bucket]
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_outputs[bucket] = manager.device_rows[0][:bucket] + manager.device_rows[1][:bucket]
            manager.mark_consumed()
            graphs[bucket] = graph
    snapshots, expected = [], []
    for step in range(64):
        bucket = (4, 8)[step % 2]
        count = step % (bucket + 1)
        ids = torch.tensor([[i % 3, 3, 8 + (i + step) % 5] for i in range(count)], dtype=torch.int64).reshape(count, 3)
        if count and step % 3 == 0:
            ids[0, 0] = -1
        want = torch.zeros((bucket, 2, 256), dtype=torch.bfloat16)
        shard.gather_into(ids, want[:count])
        expected.append(want * 2)
        manager.prepare([ids, ids], bucket)
        manager.wait_ready()
        if graph_mode:
            graphs[bucket].replay()
            result = graph_outputs[bucket]
        else:
            result = manager.device_rows[0][:bucket] + manager.device_rows[1][:bucket]
        snapshots.append(result.clone())
        manager.mark_consumed()
        assert tuple(rows.data_ptr() for rows in manager.device_rows) == ptrs
    # No result read or global device fence in the loop. The host ring only
    # fences a slot's previous DMA, while events protect graph input reuse.
    manager.close()
    assert shard.weight is None and shard._pinned_owner is None
    for actual, want in zip(snapshots, expected):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)


def test_lifecycle_and_pinned_checkpoint_load(tmp_path):
    path = tmp_path / "table.safetensors"
    save_file({"embedding": torch.ones((8, 256), dtype=torch.bfloat16)}, path)
    shard = EngramTableShard.from_safetensors(path, "embedding", [0], [(0, 8)], rows_per_copy=3)
    assert shard.weight.is_pinned()
    manager = EngramOffloadManager([shard], 2, torch.device("npu"))
    with pytest.raises(RuntimeError, match="prepared"):
        manager.wait_ready()
    with pytest.raises(ValueError, match="matrices"):
        manager.prepare([torch.tensor(0)], 1)
    manager.prepare([torch.tensor([[0]])], 2)
    with pytest.raises(RuntimeError, match="consumed"):
        manager.prepare([torch.tensor([[1]])], 2)
    with pytest.raises(RuntimeError, match="wait_ready"):
        manager.mark_consumed()
    with pytest.raises(RuntimeError, match="consumed"):
        manager.close()
    manager.wait_ready()
    other = torch.npu.Stream()
    with torch.npu.stream(other), pytest.raises(RuntimeError, match="same compute stream"):
        manager.mark_consumed()
    manager.mark_consumed()
    manager.close()
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.prepare([torch.tensor([[0]])], 1)
