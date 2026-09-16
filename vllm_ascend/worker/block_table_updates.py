# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch changed block-table ranges into one upload and the MRV2 writer."""

import torch


class BlockTableUpdates:
    """Keep graph-bound device tables stable while updating changed ranges.

    Commits and model execution use the runner's compute stream. Two pinned
    slots are fenced after DMA, so later host mutations cannot race a copy.
    Writes to the same row are coalesced before launching the upstream writer.
    """

    def __init__(self, tables, device):
        self.tables = tables
        self.max_writes = sum(table.block_table.cpu.shape[0] for table in tables)
        self.header_size = 4 * self.max_writes
        size = self.header_size + sum(table.block_table.cpu.numel() for table in tables)
        self.host = tuple(torch.empty(size, dtype=torch.int32, device="cpu", pin_memory=True) for _ in range(2))
        self.host_arrays = tuple(tensor.numpy() for tensor in self.host)
        self.device_buffer = torch.empty(size, dtype=torch.int32, device=device)
        self.group_ids, self.indices, self.starts, self.cu_lens = (
            self.device_buffer[: self.header_size].view(4, self.max_writes).unbind()
        )
        self.contents = self.device_buffer[self.header_size :]
        self.output_ptrs = torch.tensor(
            [table.block_table.gpu.data_ptr() for table in tables], dtype=torch.uint64, device=device
        )
        self.output_strides = torch.tensor(
            [table.block_table.gpu.stride(0) for table in tables], dtype=torch.int64, device=device
        )
        self.free = tuple(torch.npu.Event() for _ in self.host)
        self.used = [False] * len(self.host)
        self.step = 0
        for table in tables:
            table._dirty_ranges = {}

    def commit(self, num_reqs):
        writes = [
            (group, table, row, start, end)
            for group, table in enumerate(self.tables)
            for row, (start, end) in table._dirty_ranges.items()
            if row < num_reqs
        ]
        if not writes:
            return
        slot = self.step % len(self.host)
        if self.used[slot]:
            self.free[slot].synchronize()
        packed = self.host_arrays[slot]
        headers = packed[: self.header_size].reshape(4, self.max_writes)
        length = 0
        for index, (group, table, row, start, end) in enumerate(writes):
            count = end - start
            packed[self.header_size + length : self.header_size + length + count] = table.block_table.np[row, start:end]
            length += count
            headers[:, index] = group, row, start, length
        size = self.header_size + length
        self.device_buffer[:size].copy_(self.host[slot][:size], non_blocking=True)
        self.free[slot].record()
        self.used[slot] = True
        self.step += 1
        # Reuse vLLM MRV2's existing multi-group scatter, without UVA or
        # separate transfers for its four small metadata arrays.
        from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel

        _apply_write_kernel[(len(writes),)](
            self.output_ptrs,
            self.output_strides,
            self.indices,
            self.starts,
            self.contents,
            self.cu_lens,
            self.group_ids,
            BLOCK_SIZE=1024,
            MULTI_GROUP=True,
        )
        for _, table, row, _, _ in writes:
            del table._dirty_ranges[row]
