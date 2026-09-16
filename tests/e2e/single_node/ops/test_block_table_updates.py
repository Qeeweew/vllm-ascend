# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_interface import AscendV41MainCacheSpec
from vllm_ascend.worker.block_table import MultiGroupBlockTable


def make_tables():
    specs = [
        AscendV41MainCacheSpec(block_size=32, num_kv_heads=1, head_size=512, dtype=torch.bfloat16),
        AscendV41MainCacheSpec(block_size=64, num_kv_heads=1, head_size=512, dtype=torch.bfloat16),
        CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32),
    ]
    groups = [KVCacheGroupSpec(layer_names=[f"cache.{i}"], kv_cache_spec=spec) for i, spec in enumerate(specs)]
    with patch(
        "vllm_ascend.worker.block_table.get_dcp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)
    ):
        return MultiGroupBlockTable(
            max_num_reqs=4,
            max_model_len=256,
            max_num_batched_tokens=32,
            pin_memory=True,
            device=torch.device("npu:0"),
            block_sizes=[32, 64, 8],
            max_num_blocks=[8, 4, 1],
            kv_cache_groups=groups,
        )


def test_staged_updates_replay_reorder_and_coalesce_without_reupload():
    torch.npu.set_device(0)
    tables = make_tables()
    writer = tables._updates
    assert writer is not None
    pointers = [table.block_table.gpu.data_ptr() for table in tables.block_tables]
    for row in range(4):
        tables.add_row(([row + 1], [row + 7], [row + 12]), row)
    tables.commit_block_table(2)
    assert writer.step == 1
    assert all(set(table._dirty_ranges) == {2, 3} for table in tables.block_tables)
    tables.commit_block_table(2)
    assert writer.step == 1
    tables.commit_block_table(4)
    assert writer.step == 2
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(device="npu:0")):
        outputs = [table.block_table.gpu.clone() for table in tables.block_tables]
    snapshots = []
    for step in range(10):
        # Multiple writes overlap in the same step; only the final CPU value
        # may reach the graph-bound table, with no racing scatter programs.
        tables.add_row(([step + 1, step + 2], [step + 3], [step + 4]), 0)
        tables.append_row(([step + 5], [step + 6], []), 0)
        tables.swap_row(0, 1)
        tables.clear_row(2)
        tables.move_row(1, 2)
        before = writer.step
        tables.commit_block_table(4)
        assert writer.step == before + 1
        tables.commit_block_table(4)
        assert writer.step == before + 1
        graph.replay()
        snapshots.append(
            ([value.clone() for value in outputs], [t.block_table.cpu.clone() for t in tables.block_tables])
        )
        assert [table.block_table.gpu.data_ptr() for table in tables.block_tables] == pointers
    torch.npu.synchronize()
    for actual, expected in snapshots:
        for value, want in zip(actual, expected):
            torch.testing.assert_close(value.cpu(), want, rtol=0, atol=0)
    tables.clear()
    tables.commit_block_table(4)
    for table in tables.block_tables:
        assert not table.block_table.gpu.count_nonzero().item()


def test_new_page_uploads_only_changed_interval():
    torch.npu.set_device(0)
    tables = make_tables()
    tables.add_row(([1], [2], [3]), 0)
    tables.commit_block_table(1)
    tables.append_row(([4], [], []), 0)
    assert tables[0]._dirty_ranges == {0: (1, 2)}
    assert tables[1]._dirty_ranges == tables[2]._dirty_ranges == {}
    writer = tables._updates
    slot = writer.step % len(writer.host)
    tables.commit_block_table(1)
    torch.npu.synchronize()
    headers = writer.host_arrays[slot][: writer.header_size].reshape(4, writer.max_writes)
    assert headers[:, 0].tolist() == [0, 0, 1, 1]
    assert tables[0].block_table.gpu[0, :2].cpu().tolist() == [1, 4]
