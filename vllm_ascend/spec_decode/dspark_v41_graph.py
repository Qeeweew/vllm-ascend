# SPDX-License-Identifier: Apache-2.0
"""Independent fixed-buffer context and query graphs for the V4.1 drafter."""

from bisect import bisect_left

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.dsa_v41 import AscendV41CacheMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.dsa_v41 import DSPARK_MAX_QUERY_TOKENS


def graph_buckets(capacity: int) -> tuple[int, ...]:
    if capacity < 1:
        raise ValueError("DSpark graph capacity must be positive")
    return tuple(sorted({capacity, *(1 << i for i in range(capacity.bit_length()))}))


def select_bucket(size: int, buckets: tuple[int, ...]) -> int:
    index = bisect_left(buckets, size)
    if size < 1 or index == len(buckets):
        raise ValueError("DSpark input exceeds its prepared graph capacity")
    return buckets[index]


class DSparkV41GraphRunner:
    """Capture two graph families; all runtime state is copied on one stream.

    The target's uniform width is K+1, while draft queries have width K. Neither
    target descriptors nor the first request's context length can key both
    computations. Captured query metadata reads fixed raw device buffers; its
    Python request count is always the bucket capacity, including empty tails.
    """

    def __init__(self, proposer):
        self.proposer = proposer
        config = proposer.vllm_config
        parallel = config.parallel_config
        if (
            not 0 < proposer.num_query_per_req <= DSPARK_MAX_QUERY_TOKENS
            or not proposer.sample_from_anchor
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or proposer.dcp_size != 1
            or proposer._enable_probabilistic_draft_probs
            or config.lora_config is not None
        ):
            raise ValueError("V4.1 DSpark graphs require K1..8, DP1/CP1, greedy drafting and no LoRA")
        self.context_buckets = graph_buckets(proposer.max_num_tokens)
        self.query_buckets = graph_buckets(proposer.max_batch_size)
        self.aux = torch.zeros(
            (proposer.max_num_tokens, proposer.hidden_size * 3), dtype=proposer.dtype, device=proposer.device
        )
        self.cu_q = torch.zeros(proposer.max_batch_size + 1, dtype=torch.int32, device=proposer.device)
        self.lengths = torch.zeros(proposer.max_batch_size, dtype=torch.int32, device=proposer.device)
        self.sample_indices = torch.arange(proposer.max_query_tokens, dtype=torch.int32, device=proposer.device)
        self.tables: dict[int, torch.Tensor] = {}
        self.context = ACLGraphWrapper(self._run_context, config, CUDAGraphMode.FULL, use_eagle=True)
        self.query = ACLGraphWrapper(self._run_query, config, CUDAGraphMode.FULL, use_eagle=True)
        # Both graphs have no host task-parameter updates. The draft wrapper's
        # stream-ordered replay path is valid, without its generic FIA barrier.
        self.ready = False

    def initialize_metadata(self):
        p = self.proposer
        for group in p.draft_attn_groups:
            builder = group.get_metadata_builder()
            if not isinstance(builder, AscendV41CacheMetadataBuilder):
                raise ValueError("V4.1 graph requires a V4.1 builder for every draft cache group")
            self.tables[group.kv_cache_group_id] = torch.full(
                (p.max_batch_size, builder.max_blocks), -1, dtype=torch.int32, device=p.device
            )

    def stage_aux(self, states, rows):
        if states.ndim != 2 or states.shape[1] != self.aux.shape[1] or states.shape[0] < rows:
            raise ValueError("DSpark graph requires all three target auxiliary states")
        bucket = select_bucket(rows, self.context_buckets)
        self.aux[:rows].copy_(states[:rows])
        self.aux[rows:bucket].zero_()

    def _run_context(self, rows):
        p = self.proposer
        combined = p.model.combine_hidden_states(self.aux[:rows])
        slots = [p._per_group_context_slot_mapping_buffers[gid][:rows] for gid in p._layer_group_idx]
        p.model.precompute_and_store_context_kv(combined, p._context_positions_buffer[:rows], slots)
        # ACLGraphWrapper accepts tensors or tensor containers as outputs.
        # Context results live in the caller-owned KV caches.
        return ()

    def _run_query(self, batch):
        p = self.proposer
        tokens = batch * p.num_query_per_req
        host_cu = torch.arange(batch + 1, dtype=torch.int32) * p.num_query_per_req
        metadata = {}
        for group in p.draft_attn_groups:
            gid = group.kv_cache_group_id
            common = AscendCommonAttentionMetadata(
                query_start_loc=self.cu_q[: batch + 1],
                query_start_loc_cpu=host_cu,
                seq_lens=self.lengths[:batch],
                num_reqs=batch,
                num_actual_tokens=tokens,
                max_query_len=p.num_query_per_req,
                max_seq_len=p.draft_model_config.max_model_len,
                block_table_tensor=self.tables[gid][:batch],
                slot_mapping=p._per_group_query_slot_mapping_buffers[gid][:tokens],
                positions=p.positions[:tokens],
                causal=False,
            )
            value = group.get_metadata_builder().build_for_drafting(common, draft_index=1)
            metadata.update((name, value) for name in group.layer_names)
        context = get_forward_context()
        context.attn_metadata = metadata
        context.draft_attn_metadatas = [metadata]
        context.moe_layer_index = 0
        return p._run_merged_draft(
            num_input_tokens=tokens,
            batch_size=batch,
            token_indices_to_sample=self.sample_indices[:tokens],
            target_positions=p.positions[:tokens],
            inputs_embeds=None,
            multi_steps_attn_metadata=[metadata],
            num_tokens=tokens,
        )

    def _call(self, wrapper, size, mode):
        p = self.proposer
        tokens = size if wrapper is self.context else size * p.num_query_per_req
        descriptor = BatchDescriptor(num_tokens=tokens, uniform=True)
        with set_ascend_forward_context(
            {},
            p.vllm_config,
            num_tokens=tokens,
            num_actual_tokens=tokens,
            aclgraph_runtime_mode=mode,
            batch_descriptor=descriptor,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            return wrapper(size)

    def capture(self):
        """Run only during startup capture, with no live request cache writes."""
        if self.ready:
            return
        if not self.tables:
            raise RuntimeError("DSpark graph capture requires initialized draft cache groups")
        p = self.proposer
        self.aux.zero_()
        self.cu_q.zero_()
        self.lengths.zero_()
        p.positions.fill_(-1)
        p._context_positions_buffer.zero_()
        p.input_ids.fill_(p.parallel_drafting_token_id)
        p._dspark_seed_buffer.zero_()
        p.token_indices_to_sample[: p.max_query_tokens].copy_(
            torch.arange(p.max_query_tokens, dtype=torch.int32, device=p.device)
        )
        for slots in p._per_group_context_slot_mapping_buffers.values():
            slots.fill_(-1)
        for slots in p._per_group_query_slot_mapping_buffers.values():
            slots.fill_(-1)
        for table in self.tables.values():
            table.fill_(-1)
        # Largest first permits graph-pool reuse. Context produces only cache
        # writes; query returns caller-owned proposal storage.
        for wrapper, buckets in ((self.context, self.context_buckets), (self.query, self.query_buckets)):
            for size in reversed(buckets):
                for _ in range(2):
                    self._call(wrapper, size, CUDAGraphMode.NONE)
                self._call(wrapper, size, CUDAGraphMode.FULL)
        torch.npu.synchronize()
        self.ready = True

    def propose(
        self,
        num_speculative_tokens,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        target_model_batch_desc,
        sampling_metadata,
        mm_embed_inputs=None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output=None,
        num_scheduled_tokens=0,
        num_rejected_tokens_gpu=None,
    ):
        """Stage current requests, replay both graphs, return only active rows."""
        if not self.ready:
            raise RuntimeError("V4.1 DSpark context/query graphs must be captured before requests")
        p = self.proposer
        if num_speculative_tokens != p.num_query_per_req:
            raise ValueError("V4.1 DSpark draft length must match the configured graph query width")
        batch = common_attn_metadata.num_reqs
        query_bucket = select_bucket(batch, self.query_buckets)
        tokens, _, common, _ = p.set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            common_attn_metadata,
            num_rejected_tokens_gpu,
            req_scheduled_tokens,
            long_seq_metadata,
            num_prefill_reqs,
            num_decode_reqs,
        )
        context_rows = p._dflash_num_context
        context_bucket = select_bucket(context_rows, self.context_buckets)
        p._context_positions_buffer[context_rows:context_bucket].zero_()
        for slots in p._per_group_context_slot_mapping_buffers.values():
            slots[context_rows:context_bucket].fill_(-1)
        query_tokens = query_bucket * p.num_query_per_req
        p.input_ids[tokens:query_tokens].fill_(p.parallel_drafting_token_id)
        p.positions[tokens:query_tokens].fill_(-1)
        self.cu_q[: batch + 1].copy_(common.query_start_loc[: batch + 1])
        self.cu_q[batch + 1 : query_bucket + 1].fill_(tokens)
        self.lengths[:batch].copy_(common.seq_lens[:batch])
        self.lengths[batch:query_bucket].zero_()
        for group in p.draft_attn_groups:
            gid = group.kv_cache_group_id
            source = p._per_group_block_table_buffers[gid]
            table = self.tables[gid][:query_bucket]
            if source.shape[0] < batch or source.shape[1] > table.shape[1]:
                raise ValueError("DSpark graph page table exceeds prepared capacity")
            table.fill_(-1)
            table[:batch, : source.shape[1]].copy_(source[:batch])
            p._per_group_query_slot_mapping_buffers[gid][tokens:query_tokens].fill_(-1)
        self._call(self.context, context_bucket, CUDAGraphMode.FULL)
        return self._call(self.query, query_bucket, CUDAGraphMode.FULL)[:batch]
