# SPDX-License-Identifier: Apache-2.0
"""Optional B1 candidate selector; native dispatch remains the default.

Gather and score are pure AscendC vector kernels. The matrix multiplication
between them remains an independent CANN operation. Caller-owned workspace
keeps the large buffers stable through graph capture/replay.
"""

import torch


class CandidateIndexerB1:
    """One CR1 decode query with 32 INT8 heads and 2048 eight-position blocks."""

    def __init__(self, max_context: int, device: torch.device | str):
        if not isinstance(max_context, int) or not 1 <= max_context <= 2**27:
            raise ValueError("max_context must be a static bound in [1, 2**27]")
        self.max_context = max_context
        self.max_blocks = (max_context + 7) // 8
        self.count = max(512, min(16384, self.max_blocks * 8))
        self.key = torch.empty((1, self.count, 128), dtype=torch.bfloat16, device=device)
        self.scale = torch.empty(self.count, dtype=torch.float32, device=device)
        self.positions = torch.empty(self.count, dtype=torch.int32, device=device)
        self.qk = torch.empty((1, 32, self.count), dtype=torch.float32, device=device)
        self.scores = torch.empty((1, self.count), dtype=torch.float32, device=device)

    def __call__(self, query, key_cache, weights, query_scale, key_scale_cache, metadata, candidates):
        if query.shape != (1, 32, 128) or query.dtype != torch.int8:
            raise ValueError("Candidate B1 query must be INT8 [1,32,128]")
        if candidates.shape != (1, 1, 2048) or candidates.dtype != torch.int32:
            raise ValueError("Candidate B1 block IDs must be INT32 [1,1,2048]")
        if weights.shape != (1, 32) or query_scale.shape != (1, 32):
            raise ValueError("Candidate B1 weights and query scales must be [1,32]")
        if weights.dtype != torch.float16 or query_scale.dtype != torch.float16:
            raise ValueError("Candidate B1 weights and query scales must be FP16")
        if key_cache.ndim != 4 or key_cache.shape[1] % 8:
            raise ValueError("Candidate B1 cache page size must be divisible by 8")
        if metadata.cmp_residual_k is not None:
            raise ValueError("Candidate B1 supports CR1 only")
        if metadata.block_table.shape[0] != 1 or metadata.seqused_k.shape != (1,):
            raise ValueError("Candidate B1 requires one request")
        if metadata.block_table.shape[1] * key_cache.shape[1] < self.max_context:
            raise ValueError("Page table capacity is smaller than the declared context bound")
        # Exact FP32 block IDs, including negative/too-large sentinels. Sorting
        # permits local duplicate elimination without a dynamic-size unique().
        blocks = candidates.flatten().clamp(-1, self.max_blocks).float().sort(descending=True).values
        torch.ops._C_ascend.indexer_v41_candidate_gather(
            key_cache,
            key_scale_cache,
            blocks,
            metadata.block_table,
            metadata.seqused_k,
            metadata.cu_seqlens_q,
            self.key,
            self.scale,
            self.positions,
        )
        torch.bmm(query.bfloat16(), self.key.transpose(1, 2), out_dtype=torch.float32, out=self.qk)
        torch.ops._C_ascend.indexer_v41_candidate_score(
            self.qk, weights, query_scale, self.scale, self.positions, self.scores
        )
        return self.select_scores()[:, None], torch.empty(0, dtype=torch.int32, device=query.device)

    def select_scores(self):
        """Final selection stage, also exposed to the independent profiler."""
        selected = self.scores.topk(512, sorted=False).indices
        original = self.positions[None, :].gather(1, selected)
        valid = original >= 0
        if self.max_context <= 2**24:
            # The static bound makes FP32 position selection and sorting exact.
            # Gather encodes every invalid lane as position -1, so selection
            # need not compare scores against a scalar negative infinity.
            ordered = torch.where(valid, original.float(), float(2**24)).sort(-1).values
            indices = torch.where(ordered < 2**24, ordered, -1).int()
        else:
            ordered = torch.where(valid, original, 2**31 - 1).sort(-1).values
            indices = torch.where(ordered < 2**31 - 1, ordered, -1)
        return indices
