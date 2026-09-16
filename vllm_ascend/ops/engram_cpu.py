# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent metadata for native host hashing and direct table lookup."""

import numpy as np
import torch


class EngramCpuLookup:
    def __init__(self, hasher, shards):
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        self.hasher = hasher
        self.tables = tuple(shard.weight for shard in shards)
        self.head_indices = torch.tensor([shard.head_indices for shard in shards], dtype=torch.int64, device="cpu")
        self.local_starts = torch.stack([shard._local_starts for shard in shards])

    def gather_into(
        self,
        input_ids,
        query_start_loc,
        start_positions,
        lookback_ids,
        *,
        token_mask,
        lookback_mask,
        outputs,
        bucket_tokens,
    ):
        hasher = self.hasher
        torch.ops._C_ascend.engram_hash_gather_cpu(
            input_ids,
            torch.from_numpy(np.asarray(query_start_loc, dtype=np.int64)),
            torch.from_numpy(np.asarray(start_positions, dtype=np.int64)),
            lookback_ids,
            hasher.token_map,
            hasher.multipliers,
            hasher.primes,
            self.head_indices,
            self.local_starts,
            self.tables,
            outputs,
            hasher.pad_id,
            bucket_tokens,
            token_mask,
            lookback_mask,
        )

    def close(self):
        self.tables = ()
