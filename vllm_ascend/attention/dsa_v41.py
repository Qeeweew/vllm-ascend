# SPDX-License-Identifier: Apache-2.0
"""Device metadata for V4.1 cache groups, independent of V4 metadata.

Builders run before model invocation/replay. They copy changing data into
fixed-address buffers; native attention/indexer scheduling runs once per
cache-group build, never once per consuming model layer. One builder owns
one in-flight batch; concurrent or speculative overlapping steps need their
own builder/buffer slot.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import KVCacheLayout

from vllm_ascend.core.kv_cache_interface import (
    AscendV41IndexerCacheSpec,
    AscendV41MainCacheSpec,
    AscendV41SWACacheSpec,
    get_kv_cache_compression_ratio,
)
from vllm_ascend.ops.dsa_v41 import AscendDSAV41Metadata, AscendDSAV41Ops, build_dspark_v41_swa_indices

if TYPE_CHECKING:
    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Metadata


@dataclass
class AscendV41CacheMetadata(AttentionMetadata):
    role: str
    compress_ratio: int
    physical_block_size: int
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_kv: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    token_to_req_indices: torch.Tensor
    schedule: torch.Tensor
    seqused_cmp_kv: torch.Tensor | None = None
    cmp_residual_kv: torch.Tensor | None = None
    # Host execution classification for downstream MoE dispatch. These are
    # never used to derive device cache slots or query boundaries.
    num_prefills: int = 0
    num_decode_tokens: int = 0
    draft_swa_indices: torch.Tensor | None = None
    draft_swa_lengths: torch.Tensor | None = None


def make_v41_attention_metadata(
    swa: AscendV41CacheMetadata, main: AscendV41CacheMetadata | None = None
) -> AscendDSAV41Metadata:
    """Assemble layer-local SWA with the configured shared main-KV source.

    The caller resolves source-layer identity; metadata does not infer it from
    layer order or reuse a mutable global last-source pointer. SWA pages/table
    must retain every key for the current query chunk plus its preceding 127
    tokens. Full logical-page columns, including expired-page placeholders, are
    required: do not pass a table rebased to the first live window page.
    """
    if swa.role != "swa" or (main is not None and main.role != "main"):
        raise ValueError("V4.1 attention requires SWA and optional main-cache metadata")
    if main is not None and main.cu_seqlens_q.shape != swa.cu_seqlens_q.shape:
        raise ValueError("V4.1 source and consumer must use the same request batch")
    if main is not None and swa.draft_swa_indices is not None:
        raise ValueError("DSpark explicit SWA metadata cannot use a compressed cache")
    return AscendDSAV41Metadata(
        swa.cu_seqlens_q,
        swa.seqused_kv,
        swa.block_table,
        swa.schedule if main is None else main.schedule,
        None if main is None else main.block_table,
        None if main is None else main.seqused_cmp_kv,
        None if main is None else main.cmp_residual_kv,
        swa.draft_swa_indices,
        swa.draft_swa_lengths,
    )


def make_v41_indexer_metadata(index: AscendV41CacheMetadata) -> "AscendIndexerV41Metadata":
    # Cache/model imports are intentionally lazy to avoid a model/backend cycle.
    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Metadata

    if index.role != "index":
        raise ValueError("V4.1 indexer requires index-cache metadata")
    assert index.seqused_cmp_kv is not None
    return AscendIndexerV41Metadata(
        index.cu_seqlens_q, index.seqused_cmp_kv, index.block_table, index.schedule, index.cmp_residual_kv
    )


class AscendV41CacheMetadataBuilder(AttentionMetadataBuilder[AscendV41CacheMetadata]):
    _cudagraph_support = AttentionCGSupport.ALWAYS

    @staticmethod
    def _execution_counts(common):
        if common.num_reqs == 0 or common.num_actual_tokens == 0:
            return 0, 0
        prefilling = getattr(common, "is_prefilling", None)
        starts = getattr(common, "query_start_loc_cpu", None)
        # Missing host request state is not evidence of decode. In particular,
        # max_query_len == 1 also describes a one-token prompt/prefill tail.
        if (
            prefilling is None
            or starts is None
            or prefilling.device.type != "cpu"
            or starts.device.type != "cpu"
            or prefilling.numel() < common.num_reqs
            or starts.numel() < common.num_reqs + 1
        ):
            return common.num_reqs, 0
        lengths = starts[1 : common.num_reqs + 1] - starts[: common.num_reqs]
        is_prefill = (prefilling[: common.num_reqs] | (lengths > 1)) & (lengths > 0)
        return int(is_prefill.sum()), int(lengths.masked_fill(is_prefill, 0).sum())

    def build_for_cudagraph_capture(self, common_attn_metadata):
        metadata = self.build(0, common_attn_metadata)
        # Capture's synthetic batch can inherit real request prefill flags.
        # Only uniform single-token decode capture may select the decode MoE.
        if common_attn_metadata.max_query_len == 1:
            metadata.num_prefills = 0
            metadata.num_decode_tokens = common_attn_metadata.num_actual_tokens
        return metadata

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if isinstance(kv_cache_spec, AscendV41SWACacheSpec):
            self.role = "swa"
        elif isinstance(kv_cache_spec, AscendV41MainCacheSpec):
            self.role = "main"
        elif isinstance(kv_cache_spec, AscendV41IndexerCacheSpec):
            self.role = "index"
        else:
            raise TypeError("V4.1 metadata requires an explicit V4.1 cache spec")
        self.compress_ratio = get_kv_cache_compression_ratio(kv_cache_spec)
        self.logical_block_size = kv_cache_spec.block_size
        self.physical_block_size = kv_cache_spec.physical_block_size
        if self.physical_block_size % 16 or not 16 <= self.physical_block_size <= 1024:
            raise ValueError("V4.1 physical cache blocks must be multiples of 16 in [16,1024]")
        scheduler = vllm_config.scheduler_config
        self.max_tokens = scheduler.max_num_batched_tokens
        self.max_requests = scheduler.max_num_seqs
        self.max_sequence = vllm_config.model_config.max_model_len
        self.max_blocks = (self.max_sequence + self.logical_block_size - 1) // self.logical_block_size
        config = vllm_config.model_config.hf_config
        self.num_heads = config.num_attention_heads // vllm_config.parallel_config.tensor_parallel_size
        self.positions = torch.empty(self.max_tokens, dtype=torch.int64, device=device)
        self.token_indices = torch.arange(self.max_tokens, dtype=torch.int32, device=device)
        self.requests = torch.empty(self.max_tokens, dtype=torch.int32, device=device)
        self.slots = torch.empty(self.max_tokens, dtype=torch.int64, device=device)
        self.cu_q = torch.empty(self.max_requests + 1, dtype=torch.int32, device=device)
        self.lengths = torch.empty(self.max_requests, dtype=torch.int32, device=device)
        self.cmp_lengths = torch.empty_like(self.lengths)
        self.residual = torch.empty_like(self.lengths)
        self.table = torch.full((self.max_requests, self.max_blocks), -1, dtype=torch.int32, device=device)
        self.schedule = torch.zeros(1024, dtype=torch.int32, device=device)
        self.draft_swa_indices: torch.Tensor | None = None
        self.draft_swa_lengths: torch.Tensor | None = None

    def enable_dspark_device_metadata(self, max_query_tokens: int) -> None:
        """Opt a dedicated CR0/SWA draft builder into fixed-K5 visibility.

        Call before capture. Target builders never infer this mode from the
        common metadata's causal flag. Cache tensors may bind after this call;
        their actual physical capacity is checked on each build.
        """
        if self.role != "swa":
            raise ValueError("DSpark draft metadata requires a SWA-only CR0 cache group")
        if not 0 < max_query_tokens <= self.max_tokens:
            raise ValueError("DSpark query capacity must fit the preallocated metadata capacity")
        if self.draft_swa_indices is not None:
            if self.draft_swa_indices.shape[0] != max_query_tokens:
                raise ValueError("DSpark metadata capacity cannot change after preparation")
            return
        if self.positions.device.type == "npu" and torch.npu.is_current_stream_capturing():
            raise RuntimeError("Enable DSpark metadata before graph capture")
        self.draft_swa_indices = torch.empty(
            (max_query_tokens, 1, 256), dtype=torch.int32, device=self.positions.device
        )
        self.draft_swa_lengths = torch.empty((max_query_tokens, 1), dtype=torch.int32, device=self.positions.device)

    def _refresh_slots(self, positions, cu_q, table, requests, slots):
        if not table.shape[0]:
            requests.zero_()
            slots.fill_(-1)
            return
        token_indices = self.token_indices[: slots.numel()]
        torch.searchsorted(cu_q[1:], token_indices, right=True, out_int32=True, out=requests)
        valid_token = (token_indices < cu_q[-1]) & (positions >= 0)
        valid = valid_token
        if self.role != "swa":
            valid = valid & ((positions + 1) % self.compress_ratio == 0)
        page = positions // self.logical_block_size
        valid = valid & (page < table.shape[1])
        safe_requests = requests.clamp(0, table.shape[0] - 1).long()
        safe_page = page.clamp(0, table.shape[1] - 1)
        physical = table.flatten().index_select(0, safe_requests * table.shape[1] + safe_page)
        offset = (positions % self.logical_block_size) // self.compress_ratio
        slots.copy_(physical.long() * self.physical_block_size + offset)
        slots.masked_fill_(~valid | (physical < 0), -1)
        requests.masked_fill_(~valid_token, -1)

    def _refresh_schedule(self, cu_q, lengths, cmp_lengths, residual, batch, draft_lengths=None):
        if batch == 0:
            self.schedule.zero_()
            return
        if self.role == "index":
            schedule = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                num_heads_q=32,
                num_heads_k=1,
                head_dim=128,
                topk=512,
                quant_mode=2,
                cu_seqlens_q=cu_q,
                seqused_k=cmp_lengths,
                cmp_residual_k=residual,
                batch_size=batch,
                max_seqlen_q=self.max_tokens,
                max_seqlen_k=self.max_sequence,
                layout_q="TND",
                layout_k="PA_BBND",
                mask_mode=3,
                cmp_ratio=self.compress_ratio,
                device=str(lengths.device),
            )
        else:
            ratio = 0 if self.role == "swa" else self.compress_ratio
            draft = draft_lengths is not None
            draft_kwargs = {"ori_topk": 256, "ori_topk_length": draft_lengths} if draft else {}
            schedule = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=512,
                cu_seqlens_q=cu_q,
                seqused_ori_kv=lengths,
                seqused_cmp_kv=cmp_lengths,
                cmp_residual_kv=residual,
                batch_size=batch,
                max_seqlen_q=self.max_tokens,
                max_seqlen_ori_kv=self.max_sequence,
                max_seqlen_cmp_kv=self.max_sequence // ratio if ratio else 0,
                cmp_topk=AscendDSAV41Ops.TOPK if ratio else 0,
                cmp_ratio=ratio,
                ori_mask_mode=0 if draft else 4,
                cmp_mask_mode=3,
                ori_win_left=132 if draft else 127,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=bool(ratio),
                **draft_kwargs,
            )
        self.schedule.copy_(schedule)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Refresh buffers from DEVICE boundaries/positions, including padding.

        The common block table indexes ORIGINAL token pages. ``slot_mapping`` from
        the common metadata supplies only the token bucket size: its values are
        ignored because generic/V4 compressed-slot conventions can differ.
        """
        common = common_attn_metadata
        tokens, batch = common.slot_mapping.numel(), common.num_reqs
        draft_indices = draft_lengths = None
        if self.draft_swa_indices is not None:
            if getattr(common, "causal", None) is not False:
                raise ValueError("DSpark draft metadata requires common.causal=False")
            if tokens > self.draft_swa_indices.shape[0]:
                raise ValueError("DSpark query bucket exceeds its prepared capacity")
            draft_indices = self.draft_swa_indices[:tokens]
            draft_lengths = self.draft_swa_lengths[:tokens]
        if tokens > self.max_tokens or batch > self.max_requests:
            raise ValueError("V4.1 batch exceeds its preallocated metadata capacity")
        if common.positions is None or common.positions.numel() < tokens:
            raise ValueError("V4.1 metadata requires a device position for every bucket token")
        if common.block_table_tensor.shape[0] < batch or common.block_table_tensor.shape[1] > self.max_blocks:
            raise ValueError("V4.1 logical block table does not fit its configured context capacity")
        required_sequence = getattr(common, "max_seq_len", 0)
        if draft_indices is not None:
            # The proposer keeps K5 virtual queries even at the context end.
            # Only their in-range prefix/query keys require logical pages.
            required_sequence = min(required_sequence, self.max_sequence)
        if required_sequence > common.block_table_tensor.shape[1] * self.logical_block_size:
            raise ValueError("V4.1 requires full logical block-table columns, not a rebased sliding-window table")
        positions, requests, slots = self.positions[:tokens], self.requests[:tokens], self.slots[:tokens]
        cu_q, lengths, table = self.cu_q[: batch + 1], self.lengths[:batch], self.table[:batch]
        positions.copy_(common.positions[:tokens])
        cu_q.copy_(common.query_start_loc[: batch + 1])
        lengths.copy_(common.seq_lens[:batch])
        table.fill_(-1)
        table[:, : common.block_table_tensor.shape[1]].copy_(common.block_table_tensor[:batch])
        self._refresh_slots(positions, cu_q, table, requests, slots)
        cmp_lengths = self.cmp_lengths[:batch] if self.role != "swa" else None
        residual = self.residual[:batch] if self.role != "swa" and self.compress_ratio == 2 else None
        if cmp_lengths is not None:
            torch.div(lengths, self.compress_ratio, rounding_mode="floor", out=cmp_lengths)
        if residual is not None:
            torch.remainder(lengths, self.compress_ratio, out=residual)
        if draft_indices is not None:
            cache = self.vllm_config.compilation_config.static_forward_context[self.layer_names[0]].kv_cache
            if (
                not isinstance(cache, torch.Tensor)
                or cache.ndim != 4
                or cache.shape[0] == 0
                or cache.shape[1:] != (self.physical_block_size, 1, 512)
                or cache.dtype != torch.bfloat16
                or cache.device != table.device
            ):
                raise ValueError("DSpark SWA cache must be bound before building draft metadata")
            if batch:
                build_dspark_v41_swa_indices(
                    table,
                    cu_q,
                    lengths,
                    page_size=self.physical_block_size,
                    num_cache_blocks=cache.shape[0],
                    indices_output=draft_indices,
                    lengths_output=draft_lengths,
                    max_model_len=self.max_sequence,
                )
            else:
                draft_indices.fill_(-1)
                draft_lengths.zero_()
            slots.masked_fill_(slots >= cache.shape[0] * self.physical_block_size, -1)
            padded_query = (positions < 0) | (positions >= self.max_sequence) | (draft_lengths[:, 0] == 0)
            slots.masked_fill_(padded_query, -1)
            requests.masked_fill_(padded_query, -1)
            # Derive visibility above from the original virtual prefix + K5,
            # then bound native scheduling without mutating common.seq_lens.
            lengths.clamp_(0, self.max_sequence)
            # Empty request slots must not claim a nonzero native KV interval
            # at the duplicated query boundary of the next active request.
            lengths.masked_fill_(cu_q[1:] == cu_q[:-1], 0)
        self._refresh_schedule(cu_q, lengths, cmp_lengths, residual, batch, draft_lengths)
        num_prefills, num_decode_tokens = self._execution_counts(common)
        return AscendV41CacheMetadata(
            self.role,
            self.compress_ratio,
            self.physical_block_size,
            positions,
            cu_q,
            lengths,
            table,
            slots,
            requests,
            self.schedule,
            cmp_lengths,
            residual,
            num_prefills,
            num_decode_tokens,
            draft_indices,
            draft_lengths,
        )


class AscendV41CacheImpl:
    @staticmethod
    def update_graph_params(*args, **kwargs):
        # The runner invokes this hook for every backend during full graph
        # replay. V4.1 has no mutable task-group parameters: build() refreshes
        # all fixed-address tensor contents before replay, including schedules.
        pass


class AscendV41CacheBackend(AttentionBackend):
    """Storage metadata backend; attention execution belongs to the V4.1 op."""

    @staticmethod
    def get_name():
        return "ASCEND_V41_CACHE"

    @staticmethod
    def get_builder_cls():
        return AscendV41CacheMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return AscendV41CacheImpl

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls):
        return True

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 512]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Compact cache-state views require contiguous rows inside each layer.
        # Placing the block axis outside layers breaks that storage contract.
        return (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC)
