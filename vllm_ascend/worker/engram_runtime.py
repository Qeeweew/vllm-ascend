# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-side Engram handoff, always outside model capture and replay."""

from collections.abc import Sequence

import torch
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.ops.engram_offload import EngramOffloadManager
from vllm_ascend.worker.engram_history import EngramRequestHistory


class EngramRuntime:
    """Join actual device inputs, CPU history/hash, and pinned table staging.

    The runner seeds/drops requests through ``history`` and calls
    ``begin_prepare`` after final device-side token/position corrections, then
    ``finish_prepare`` after submitting metadata work. ``wait_ready`` goes
    immediately before model/replay; ``mark_consumed`` goes immediately after.
    A pinned D2H snapshot resolves asynchronous CPU placeholders. Its event
    precedes metadata work, allowing CPU history/hash to overlap that work.
    ``prepare`` retains the synchronous handoff for standalone callers.
    """

    def __init__(self, history: EngramRequestHistory, offload: EngramOffloadManager):
        if len(history.hasher.layout.layer_ids) != len(offload.shards):
            raise ValueError("Engram history and offload layer counts differ")
        self.history = history
        self.offload = offload
        self.token_mask = torch.zeros(offload.max_tokens, dtype=torch.bool, device=offload.device)
        self.image_token_mask = torch.zeros_like(self.token_mask)
        # IDs, positions, optional mask, and at most max_tokens request
        # boundaries. Storage is reused only after the snapshot event completes.
        capacity = 4 * offload.max_tokens + 1
        self._snapshot_device = torch.empty(capacity, dtype=torch.int64, device=offload.device)
        self._snapshot_host = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        self._snapshot_ready = torch.npu.Event()
        self._snapshot_pending = None
        self._prepared = False
        self._closed = False

    @property
    def snapshot_pending(self) -> bool:
        return self._snapshot_pending is not None

    def prepare(self, request_ids, input_ids, positions, query_start_loc, bucket_tokens, *, token_mask=None):
        self.begin_prepare(request_ids, input_ids, positions, query_start_loc, bucket_tokens, token_mask=token_mask)
        return self.finish_prepare()

    def begin_prepare(
        self,
        request_ids: Sequence[str],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        bucket_tokens: int,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        """Enqueue a snapshot of final rows before unrelated metadata work.

        All inputs are on the compute NPU and boundaries contain only real
        requests. ``input_ids``/``positions`` may include graph padding. Vision
        prompts reuse the validated mask seeded by the model's request hook.
        Explicit device masks remain supported and must agree with the prompt.
        """
        device = self.offload.device
        if self._prepared or self._closed or self.snapshot_pending:
            raise RuntimeError("Engram runtime is closed or the previous step was not consumed")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Engram runtime preparation must remain outside graph capture")
        if not 0 < bucket_tokens <= self.offload.max_tokens:
            raise ValueError("Engram token bucket exceeds staging capacity")
        if input_ids.ndim != 1 or positions.shape != input_ids.shape or input_ids.numel() < bucket_tokens:
            raise ValueError("Engram needs flat final token/position buffers covering the graph bucket")
        if len(request_ids) > self.offload.max_tokens:
            raise ValueError("Engram request count exceeds snapshot capacity")
        if query_start_loc.shape != (len(request_ids) + 1,):
            raise ValueError("Engram boundaries must describe real requests without padded request rows")
        for tensor in (input_ids, positions, query_start_loc):
            if tensor.device != device or tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError("Engram snapshot inputs must be integer tensors on the compute NPU")
        if token_mask is not None and (
            token_mask.device != device or token_mask.dtype != torch.bool or token_mask.shape != input_ids.shape
        ):
            raise ValueError("Engram token mask must match the final device input IDs")
        fields = [input_ids[:bucket_tokens], positions[:bucket_tokens], query_start_loc]
        if token_mask is not None:
            fields.append(token_mask[:bucket_tokens])
        # Keep the DMA on the producer stream so final token corrections are
        # ordered before it. Waiting on this event later does NOT wait for the
        # preparation graph submitted after it on the same stream.
        with record_function_or_nullcontext("v41::engram_snapshot_submit"):
            size = sum(field.numel() for field in fields)
            packed = self._snapshot_device[:size]
            torch.cat(fields, out=packed)
            self._snapshot_host[:size].copy_(packed, non_blocking=True)
            self._snapshot_ready.record(torch.npu.current_stream(device))
        self._snapshot_pending = (tuple(request_ids), bucket_tokens, token_mask is not None, size)

    def finish_prepare(self) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Wait only for the snapshot, then hash and submit pinned row DMA."""
        if self._closed or self._prepared or not self.snapshot_pending:
            raise RuntimeError("Engram has no pending snapshot or the previous step was not consumed")
        request_ids, bucket_tokens, has_mask, size = self._snapshot_pending
        with record_function_or_nullcontext("v41::engram_snapshot_d2h"):
            self._snapshot_ready.synchronize()
        snapshot = self._snapshot_host[:size]
        boundaries = snapshot[2 * bucket_tokens : 2 * bucket_tokens + len(request_ids) + 1]
        count = int(boundaries[-1])
        if count < 0 or count > bucket_tokens:
            raise ValueError("Engram final query boundary exceeds the token bucket")
        mask = snapshot[-bucket_tokens:].bool()[:count] if has_mask else None
        with record_function_or_nullcontext("v41::engram_hash"):
            batch = self.history.prepare(
                request_ids,
                snapshot[:count],
                snapshot[bucket_tokens : bucket_tokens + count],
                boundaries,
                token_mask=mask,
                use_seeded_prompt_mask=mask is None,
            )
        with record_function_or_nullcontext("v41::engram_gather_h2d"):
            rows = self.offload.prepare(batch.hash_ids, bucket_tokens)
        # A blocking copy keeps the small pageable CPU mask alive until DMA
        # completes. Large embedding rows use the independent pinned ring.
        self.token_mask[:count].copy_(batch.token_mask)
        self.token_mask[count:bucket_tokens].zero_()
        self.image_token_mask[:count].copy_(batch.image_token_mask)
        self.image_token_mask[count:bucket_tokens].zero_()
        self._snapshot_pending = None
        self._prepared = True
        return rows, self.token_mask[:bucket_tokens]

    def wait_ready(self) -> None:
        self.offload.wait_ready()

    def mark_consumed(self) -> None:
        self.offload.mark_consumed()
        self._prepared = False

    def close(self) -> None:
        if self.snapshot_pending:
            raise RuntimeError("Engram snapshot must be finished before close; use shutdown on failure")
        self.offload.close()
        self._closed = True

    def shutdown(self) -> None:
        """Checked termination cleanup, including an unfinished model step."""
        if self._closed:
            return
        if self.snapshot_pending:
            self._snapshot_ready.synchronize()
            self._snapshot_pending = None
        self.offload.shutdown()
        self._prepared = False
        self._closed = True
