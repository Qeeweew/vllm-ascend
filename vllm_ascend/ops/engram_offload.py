# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host Engram head shards and explicit preparation before NPU graph replay.

Lookup depends only on token hashes. Projection and residual-dependent gating
remain in the model. Host lookup runs before replay; fixed-address pinned H2D
copies are captured in a separate upload graph.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

DEAD_HASH_ID = -1
HOST_BUFFER_COUNT = 2


class EngramTableShard:
    """Concatenated CPU rows for this TP rank's disjoint hash-head buckets."""

    def __init__(
        self,
        weight: torch.Tensor,
        head_indices: Sequence[int],
        head_ranges: Sequence[tuple[int, int]],
    ) -> None:
        if weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError("Engram host weights must be CPU BF16 [rows,head_dim]")
        if not weight.is_contiguous() or not head_indices or len(head_indices) != len(head_ranges):
            raise ValueError("Engram shard requires contiguous rows and matching nonempty head ranges")
        if len(set(head_indices)) != len(head_indices) or min(head_indices) < 0:
            raise ValueError("Engram head indices must be unique nonnegative integers")
        cursor = 0
        local_starts = []
        for start, end in head_ranges:
            if start < 0 or end <= start:
                raise ValueError("Invalid Engram bucket range")
            local_starts.append(cursor)
            cursor += end - start
        if cursor != weight.shape[0]:
            raise ValueError("Host storage size does not match the selected hash-head buckets")
        self.weight = weight
        self._pinned_owner = None
        self._closed = False
        self.head_indices = tuple(head_indices)
        self.head_ranges = tuple(head_ranges)
        self._columns = torch.tensor(self.head_indices, dtype=torch.int64)
        self._global_starts = torch.tensor([start for start, _ in head_ranges], dtype=torch.int64)
        self._global_ends = torch.tensor([end for _, end in head_ranges], dtype=torch.int64)
        self._local_starts = torch.tensor(local_starts, dtype=torch.int64)

    @classmethod
    def from_safetensors(
        cls,
        filename: str | Path,
        tensor_name: str,
        head_indices: Sequence[int],
        head_ranges: Sequence[tuple[int, int]],
        *,
        pin_memory: bool = True,
        rows_per_copy: int = 65536,
        numa_node: int | None = None,
        device: torch.device | None = None,
    ) -> "EngramTableShard":
        """Load directly into final host storage, never materializing a full table.

        pin_memory=False is an explicit bounded-pinned mode: only transfer
        staging is pinned. Allocation errors never silently select that mode.
        Explicit numa_node/device uses a registered mmap populated directly by
        checkpoint chunks before registration. The default allocator is unchanged.
        """
        if (numa_node is None) != (device is None):
            raise ValueError("Engram NUMA loading requires both numa_node and device")
        if numa_node is not None and not pin_memory:
            raise ValueError("Engram NUMA registered tables require pin_memory=True")
        if rows_per_copy <= 0 or not head_ranges:
            raise ValueError("rows_per_copy and number of heads must be positive")
        with safe_open(filename, framework="pt", device="cpu") as reader:
            source = reader.get_slice(tensor_name)
            shape = source.get_shape()
            if source.get_dtype() != "BF16" or len(shape) != 2:
                raise ValueError("Convert the Engram table to a BF16 matrix before offloading")
            if any(start < 0 or end <= start or end > shape[0] for start, end in head_ranges):
                raise ValueError("Engram bucket range exceeds the checkpoint table")
            rows = sum(end - start for start, end in head_ranges)

            def populate(weight):
                cursor = 0
                for start, end in head_ranges:
                    for row in range(start, end, rows_per_copy):
                        count = min(rows_per_copy, end - row)
                        weight[cursor : cursor + count].copy_(source[row : row + count])
                        cursor += count

            owner = None
            try:
                if numa_node is None:
                    weight = torch.empty((rows, shape[1]), dtype=torch.bfloat16, device="cpu", pin_memory=pin_memory)
                    if pin_memory and not weight.is_pinned():
                        raise RuntimeError("Engram host allocation is not pinned")
                    populate(weight)
                else:
                    # Worker-only optional allocation path; no global policy change.
                    from vllm_ascend.ops.engram_pinned_host import EngramPinnedHostTensor

                    owner = EngramPinnedHostTensor(
                        (rows, shape[1]), numa_node=numa_node, device=device, initialize=populate
                    )
                    weight = owner.tensor
                shard = cls(weight, head_indices, head_ranges)
                shard._pinned_owner = owner
                return shard
            except Exception:
                if owner is not None:
                    owner.close()
                raise

    def close(self) -> None:
        if self._closed:
            return
        if self._pinned_owner is not None:
            self._pinned_owner.close()
            self._pinned_owner = None
        self.weight = None
        self._closed = True

    def gather_into(self, hash_ids: torch.Tensor, output: torch.Tensor) -> None:
        """Gather local heads, zero DEAD entries and reject wrong bucket IDs."""
        if self._closed:
            raise RuntimeError("Engram host table shard is closed")
        if hash_ids.device.type != "cpu" or hash_ids.dtype != torch.int64 or hash_ids.ndim != 2:
            raise ValueError("Host lookup requires CPU int64 [tokens,all_hash_heads]")
        if hash_ids.shape[1] <= max(self.head_indices):
            raise ValueError("Missing hash-head columns")
        expected = (hash_ids.shape[0], len(self.head_indices), self.weight.shape[1])
        if output.device.type != "cpu" or output.dtype != torch.bfloat16 or tuple(output.shape) != expected:
            raise ValueError(f"Lookup destination must be CPU BF16 {expected}")
        if not output.is_contiguous():
            raise ValueError("Lookup destination must be contiguous")
        selected = hash_ids.index_select(1, self._columns)
        dead = selected == DEAD_HASH_ID
        valid = (selected >= self._global_starts) & (selected < self._global_ends)
        if not (dead | valid).all():
            raise ValueError("Engram hash ID is outside its head's bucket range")
        local = selected - self._global_starts + self._local_starts
        local.masked_fill_(dead, 0)
        torch.index_select(self.weight, 0, local.flatten(), out=output.view(-1, self.weight.shape[1]))
        output.masked_fill_(dead.unsqueeze(-1), 0)


class EngramOffloadManager:
    """Own stable device inputs and a bounded host staging ring.

    A runner step is prepare(hashes, bucket) -> wait_ready() -> model/replay ->
    mark_consumed(). Calling prepare from capture, omitting mark_consumed, or
    using hashes from placeholder token history is not supported.
    """

    def __init__(
        self,
        shards: Sequence[EngramTableShard],
        max_tokens: int,
        device: torch.device,
        capture_sizes: Sequence[int] = (),
    ) -> None:
        if not shards or max_tokens <= 0 or device.type != "npu":
            raise ValueError("Engram offload needs host shards, a positive token capacity, and an NPU")
        self.shards = tuple(shards)
        self.max_tokens = max_tokens
        self.device = torch.device("npu", torch.npu.current_device()) if device.index is None else device
        device = self.device
        self.device_rows = tuple(
            torch.zeros(
                (max_tokens, len(shard.head_indices), shard.weight.shape[1]), dtype=torch.bfloat16, device=device
            )
            for shard in shards
        )
        self._host_rows = tuple(
            tuple(
                torch.zeros(rows.shape, dtype=torch.bfloat16, device="cpu", pin_memory=True)
                for rows in self.device_rows
            )
            for _ in range(HOST_BUFFER_COUNT)
        )
        self.device_masks = tuple(torch.zeros(max_tokens, dtype=torch.bool, device=device) for _ in range(2))
        self._host_masks = tuple(
            tuple(torch.zeros(max_tokens, dtype=torch.bool, device="cpu", pin_memory=True) for _ in range(2))
            for _ in range(HOST_BUFFER_COUNT)
        )
        with torch.npu.device(device):
            self._copy_stream = torch.npu.Stream(device=device)
            self._host_free = tuple(torch.npu.Event() for _ in range(HOST_BUFFER_COUNT))
            self._host_used = [False] * HOST_BUFFER_COUNT
            self._ready = torch.npu.Event()
            self._consumed = torch.npu.Event()
            self._consumed.record(torch.npu.current_stream(device))
            self._upload_sizes = sorted({max_tokens, *(size for size in capture_sizes if 0 < size <= max_tokens)})
            self._upload_graphs = {}
            # Every captured source address belongs to one persistent host
            # slot. Capture once at startup, never in a serving step.
            self._copy_stream.wait_stream(torch.npu.current_stream(device))
            for slot in range(HOST_BUFFER_COUNT):
                for size in self._upload_sizes:
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=self._copy_stream):
                        for dst, src in zip(self.device_rows, self._host_rows[slot]):
                            dst[:size].copy_(src[:size], non_blocking=True)
                        for dst, src in zip(self.device_masks, self._host_masks[slot]):
                            dst[:size].copy_(src[:size], non_blocking=True)
                    self._upload_graphs[slot, size] = graph
            self._copy_stream.synchronize()
        self._step = 0
        self._prepared = False
        self._waited = False
        self._closed = False

    def prepare(self, hash_ids: Sequence[torch.Tensor], bucket_tokens: int) -> tuple[torch.Tensor, ...]:
        if self._closed or self._prepared:
            raise RuntimeError("Engram manager is closed or the previous step was not marked consumed")
        with torch.npu.device(self.device):
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Host Engram preparation must execute outside graph capture/replay")
        if len(hash_ids) != len(self.shards) or not 0 < bucket_tokens <= self.max_tokens:
            raise ValueError("Invalid Engram layer count or token bucket")
        if any(ids.ndim != 2 for ids in hash_ids):
            raise ValueError("Engram hashes must be matrices")
        tokens = hash_ids[0].shape[0]
        if tokens > bucket_tokens or any(ids.shape[0] != tokens for ids in hash_ids):
            raise ValueError("All Engram layers must describe the same tokens within the bucket")
        staging = self.acquire_staging(bucket_tokens)
        for shard, ids, rows in zip(self.shards, hash_ids, staging):
            shard.gather_into(ids, rows[:tokens])
            rows[tokens:bucket_tokens].zero_()
        return self.submit_staging(bucket_tokens, tokens)

    def acquire_staging(self, bucket_tokens: int) -> tuple[torch.Tensor, ...]:
        if self._closed or self._prepared:
            raise RuntimeError("Engram manager is closed or the previous step was not marked consumed")
        if not 0 < bucket_tokens <= self.max_tokens:
            raise ValueError("Invalid Engram token bucket")
        host_slot = self._step % HOST_BUFFER_COUNT
        if self._host_used[host_slot]:
            # Wait only for the previous DMA reading this slot, not the model.
            self._host_free[host_slot].synchronize()
        return self._host_rows[host_slot]

    def submit_staging(
        self,
        bucket_tokens: int,
        tokens: int,
        token_mask: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self._closed or self._prepared:
            raise RuntimeError("Engram manager is closed or the previous step was not marked consumed")
        if not 0 <= tokens <= bucket_tokens <= self.max_tokens or bucket_tokens == 0:
            raise ValueError("Invalid Engram token bucket")
        host_slot = self._step % HOST_BUFFER_COUNT
        for index, (dst, source) in enumerate(zip(self._host_masks[host_slot], (token_mask, image_token_mask))):
            values = dst.numpy()
            values[:tokens] = (index == 0) if source is None else source.numpy()
            values[tokens:bucket_tokens] = False
        upload_size = next(size for size in self._upload_sizes if size >= bucket_tokens)
        with torch.npu.stream(self._copy_stream):
            # Device inputs cannot be overwritten while the prior graph reads.
            self._copy_stream.wait_event(self._consumed)
            self._upload_graphs[host_slot, upload_size].replay()
            self._host_free[host_slot].record(self._copy_stream)
            self._ready.record(self._copy_stream)
        self._host_used[host_slot] = True
        self._prepared = True
        self._waited = False
        self._step += 1
        return tuple(rows[:bucket_tokens] for rows in self.device_rows)

    def wait_ready(self) -> None:
        if not self._prepared:
            raise RuntimeError("No Engram rows have been prepared for this step")
        self._consumer_stream = torch.npu.current_stream(self.device)
        self._consumer_stream.wait_event(self._ready)
        self._waited = True

    def mark_consumed(self) -> None:
        if not self._prepared or not self._waited:
            raise RuntimeError("Engram consumption must follow prepare and wait_ready")
        if torch.npu.current_stream(self.device) != self._consumer_stream:
            raise RuntimeError("Engram wait and model consumption must use the same compute stream")
        self._consumed.record(self._consumer_stream)
        self._prepared = False

    def close(self) -> None:
        if self._closed:
            return
        if self._prepared:
            raise RuntimeError("Mark the final model invocation consumed before closing Engram offload")
        self._copy_stream.synchronize()
        self._consumed.synchronize()
        self._upload_graphs.clear()
        for shard in self.shards:
            shard.close()
        self._closed = True

    def shutdown(self) -> None:
        """Terminate even an unfinished step, after fencing every device stream.

        A failed forward may never record its consumed event, and a failed
        prepare may already have enqueued DMA. Neither normal close nor the
        previous consumed event alone can establish safety in those cases.
        Synchronization/unregistration errors propagate with owners retained
        for a later shutdown retry; this is never a serving-step operation.
        """
        if self._closed:
            return
        with torch.npu.device(self.device):
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Engram shutdown must remain outside graph capture")
            torch.npu.synchronize(self.device)
        # Only a successful device-wide fence permits abandoning pending use.
        self._prepared = False
        self._waited = False
        self.close()
