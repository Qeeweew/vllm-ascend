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

    The runner seeds/drops requests through ``history`` and calls ``prepare``
    after the final device-side token/position corrections. ``wait_ready`` goes
    immediately before model/replay; ``mark_consumed`` goes immediately after.
    A single packed D2H snapshot resolves asynchronous CPU placeholders. Its
    synchronization cost must be included in end-to-end decode profiling.
    """

    def __init__(self, history: EngramRequestHistory, offload: EngramOffloadManager):
        if len(history.hasher.layout.layer_ids) != len(offload.shards):
            raise ValueError("Engram history and offload layer counts differ")
        self.history = history
        self.offload = offload
        self.token_mask = torch.zeros(offload.max_tokens, dtype=torch.bool, device=offload.device)
        self.image_token_mask = torch.zeros_like(self.token_mask)
        self._prepared = False
        self._closed = False

    def prepare(
        self,
        request_ids: Sequence[str],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        bucket_tokens: int,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Snapshot final packed rows; ignore graph padding after the last query.

        All inputs are on the compute NPU and boundaries contain only real
        requests. ``input_ids``/``positions`` may include graph padding. Vision
        prompts reuse the validated mask seeded by the model's request hook.
        Explicit device masks remain supported and must agree with the prompt.
        """
        device = self.offload.device
        if self._prepared or self._closed:
            raise RuntimeError("Engram runtime is closed or the previous step was not consumed")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Engram runtime preparation must remain outside graph capture")
        if not 0 < bucket_tokens <= self.offload.max_tokens:
            raise ValueError("Engram token bucket exceeds staging capacity")
        if input_ids.ndim != 1 or positions.shape != input_ids.shape or input_ids.numel() < bucket_tokens:
            raise ValueError("Engram needs flat final token/position buffers covering the graph bucket")
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
        # One D2H synchronization, independent of request count. Never read
        # input_batch.token_ids_cpu for asynchronously sampled/decode tokens.
        with record_function_or_nullcontext("v41::engram_snapshot_d2h"):
            snapshot = torch.cat([field.to(torch.int64) for field in fields]).cpu()
        boundaries = snapshot[2 * bucket_tokens : 2 * bucket_tokens + len(request_ids) + 1]
        count = int(boundaries[-1])
        if count < 0 or count > bucket_tokens:
            raise ValueError("Engram final query boundary exceeds the token bucket")
        mask = snapshot[-bucket_tokens:].bool()[:count] if token_mask is not None else None
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
        self._prepared = True
        return rows, self.token_mask[:bucket_tokens]

    def wait_ready(self) -> None:
        self.offload.wait_ready()

    def mark_consumed(self) -> None:
        self.offload.mark_consumed()
        self._prepared = False

    def close(self) -> None:
        self.offload.close()
        self._closed = True

    def shutdown(self) -> None:
        """Checked termination cleanup, including an unfinished model step."""
        if self._closed:
            return
        self.offload.shutdown()
        self._prepared = False
        self._closed = True
