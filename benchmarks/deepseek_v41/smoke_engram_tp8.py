# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-rank host lookup -> pinned DMA -> HCCL head gather -> graph replay.

Run with torch.distributed.run --standalone --nproc-per-node=8. This is a
component smoke test, not a throughput benchmark or full-model validation.
"""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.ops.engram_offload import EngramOffloadManager, EngramTableShard


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    world = dist.get_world_size()
    if world != 8:
        raise ValueError("This validation requires TP8")
    device = torch.device("npu", rank)
    bucket, heads, dim, rows_per_head = 8, 24, 256, 13
    local_heads = heads // world
    full = torch.arange(heads * rows_per_head * dim).reshape(-1, dim).remainder(127).bfloat16()
    columns = list(range(rank * local_heads, (rank + 1) * local_heads))
    ranges = [(head * rows_per_head, (head + 1) * rows_per_head) for head in columns]
    host = torch.empty((local_heads * rows_per_head, dim), dtype=torch.bfloat16, pin_memory=True)
    host.copy_(full[ranges[0][0] : ranges[-1][1]])
    shard = EngramTableShard(host, columns, ranges)
    manager = EngramOffloadManager([shard, shard], bucket, device)
    gathered = torch.empty((world * bucket, local_heads, dim), dtype=torch.bfloat16, device=device)

    def run():
        local = manager.device_rows[0] + manager.device_rows[1]
        dist.all_gather_into_tensor(gathered, local)
        return gathered.reshape(world, bucket, local_heads, dim).permute(1, 0, 2, 3).reshape(bucket, heads, dim)

    initial = torch.arange(heads, dtype=torch.int64)[None] * rows_per_head
    manager.prepare([initial, initial], bucket)
    manager.wait_ready()
    for _ in range(3):
        run()
    torch.npu.synchronize()
    dist.barrier()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    manager.mark_consumed()
    snapshots, expected = [], []
    for step in range(32):
        count = step % (bucket + 1)
        offsets = torch.arange(heads, dtype=torch.int64)
        hashes = offsets[None] * rows_per_head + (offsets[None] + torch.arange(count)[:, None] + step) % rows_per_head
        want = torch.zeros((bucket, heads, dim), dtype=torch.bfloat16)
        want[:count] = full[hashes] * 2
        manager.prepare([hashes, hashes], bucket)
        manager.wait_ready()
        graph.replay()
        snapshots.append(result.clone())
        expected.append(want)
        manager.mark_consumed()
    manager.close()
    for actual, want in zip(snapshots, expected):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world,
                    "steps": len(snapshots),
                    "graph": True,
                    "layers": 2,
                    "heads": heads,
                    "head_dim": dim,
                    "exact": True,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
