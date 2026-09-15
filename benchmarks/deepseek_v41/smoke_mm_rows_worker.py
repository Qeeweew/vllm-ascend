# SPDX-License-Identifier: Apache-2.0
"""Optional synchronous Engram-content diagnostics, separate from repeat controls."""

import hashlib

import torch
from smoke_mm_runner_worker import V41MMSmokeWorker


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class V41MMRowsSmokeWorker(V41MMSmokeWorker):
    def start_mm_smoke(self):
        result = super().start_mm_smoke()
        self.engram_content_steps = []
        self.pending_engram_content = None
        runtime = self.model_runner.engram_runtime
        original_prepare = runtime.prepare
        original_wait_ready = runtime.wait_ready

        def observe_prepare(request_ids, input_ids, positions, boundaries, bucket_tokens, **kwargs):
            # Deliberately additional blocking D2H copies: this separate
            # diagnostic must never be interpreted as a performance run.
            entry = {
                "request_ids": list(request_ids),
                "input_ids": input_ids.cpu().tolist(),
                "positions": positions.cpu().tolist(),
                "query_start_loc": boundaries.cpu().tolist(),
                "bucket_tokens": bucket_tokens,
            }
            rows, mask = original_prepare(request_ids, input_ids, positions, boundaries, bucket_tokens, **kwargs)
            self.pending_engram_content = (entry, rows, mask)
            return rows, mask

        def observe_wait_ready():
            original_wait_ready()
            entry, rows, mask = self.pending_engram_content
            entry["rows_sha256"] = [tensor_digest(row) for row in rows]
            entry["engram_mask_sha256"] = tensor_digest(mask)
            entry["image_mask_sha256"] = tensor_digest(runtime.image_token_mask[: entry["bucket_tokens"]])
            self.engram_content_steps.append(entry)
            self.pending_engram_content = None

        runtime.prepare = observe_prepare
        runtime.wait_ready = observe_wait_ready
        return result

    def inspect_mm_smoke(self):
        result = super().inspect_mm_smoke()
        result["engram_content_steps"] = list(getattr(self, "engram_content_steps", []))
        return result
