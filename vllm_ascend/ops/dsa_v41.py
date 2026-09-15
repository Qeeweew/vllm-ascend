# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 BF16 shared-KV attention on Ascend 910B.

Projection, normalization, cache writes and RoPE are deliberately separate.
Q and both caches enter after RoPE; compressed row j uses position j * CR
(group first). The output still needs inverse query RoPE before wo_a.

BF16 cache storage is an accuracy baseline, not an interpretation of the
CUDA packed cache. The standalone model reference uses FP8 SWA and FP4
group16/E4M3 main KV; current vLLM CUDA main-cache storage instead depends
on the backend (MXFP8, older mixed FP8/BF16, or plain rows). The indexer's
group32/E8M0 format is different again.
"""

from dataclasses import dataclass

import torch


def build_dspark_v41_swa_indices(
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    *,
    page_size: int,
    num_cache_blocks: int,
    indices_output: torch.Tensor,
    lengths_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write fixed-K5 noncausal visibility into caller-owned device buffers.

    Official DSpark uses [max(prefix_length - 128, 0), sequence_length) for
    every query in the block: 128 prefix tokens plus all five draft queries.
    Unlike the old V4 helper's physical slots, arch22 SparseFlashMla consumes
    LOGICAL token IDs and applies the block table itself. Invalid page entries
    become -1, without compacting later valid columns. Lengths cover the full
    column span, including holes. Padding rows have length zero and -1 IDs.

    Active requests must have five query rows; empty request slots are allowed.
    Offsets must be nondecreasing, and active rows must fit the output capacity.
    No runtime tensor values are read on host. Intermediate tensors are device
    operations; the output addresses remain stable across graph replay.
    """
    if block_table.ndim != 2 or min(block_table.shape) == 0:
        raise ValueError("DSpark block table must be nonempty [requests, pages]")
    batch = block_table.shape[0]
    if cu_seqlens_q.shape != (batch + 1,) or seqused_kv.shape != (batch,):
        raise ValueError("DSpark query offsets and lengths must match the request count")
    if indices_output.ndim != 3 or indices_output.shape[1:] != (1, 256):
        raise ValueError("DSpark indices_output must be [T,1,256]")
    if lengths_output.shape != indices_output.shape[:2]:
        raise ValueError("DSpark lengths_output must be [T,1]")
    tensors = (block_table, cu_seqlens_q, seqused_kv, indices_output, lengths_output)
    if any(t.dtype != torch.int32 or t.device != block_table.device for t in tensors):
        raise ValueError("DSpark metadata and output buffers must be INT32 on one device")
    if not indices_output.is_contiguous() or not lengths_output.is_contiguous():
        raise ValueError("DSpark output buffers must be contiguous")
    if page_size % 16 or not 16 <= page_size <= 1024 or num_cache_blocks <= 0:
        raise ValueError("DSpark requires a valid BF16 cache page size and positive block count")
    query_lengths = cu_seqlens_q[1:].long() - cu_seqlens_q[:-1].long()
    prefix_lengths = seqused_kv.long() - query_lengths
    starts = (prefix_lengths - 128).clamp_min(0)
    visible_lengths = (seqused_kv.long() - starts).clamp(0, 133)
    request_valid = (query_lengths == 5) & (prefix_lengths >= 0)
    request_valid &= seqused_kv <= block_table.shape[1] * page_size
    columns = torch.arange(256, device=block_table.device)
    positions = starts[:, None] + columns[None, :]
    logical_pages = positions // page_size
    physical_pages = block_table.gather(1, logical_pages.clamp(0, block_table.shape[1] - 1))
    valid = request_valid[:, None] & (columns[None, :] < visible_lengths[:, None])
    valid &= (logical_pages < block_table.shape[1]) & (physical_pages >= 0) & (physical_pages < num_cache_blocks)
    logical_indices = torch.where(valid, positions, -1).int()

    rows = torch.arange(indices_output.shape[0], device=block_table.device, dtype=torch.int32)
    request_ids = torch.searchsorted(cu_seqlens_q[1:].contiguous(), rows, right=True).clamp_max(batch - 1)
    row_valid = (rows >= cu_seqlens_q[0]) & (rows < cu_seqlens_q[-1])
    row_valid &= request_valid[request_ids]
    indices_output[:, 0].copy_(torch.where(row_valid[:, None], logical_indices[request_ids], -1))
    lengths_output[:, 0].copy_(torch.where(row_valid, visible_lengths[request_ids], 0).int())
    return indices_output, lengths_output


@dataclass
class AscendDSAV41Metadata:
    """Device buffers whose contents must be refreshed before graph replay.

    The SWA table maps ORIGINAL logical token pages (including old pages) to
    physical cache blocks. A ring table may alias expired pages only when no
    query in the current chunk can still attend them. In particular, retain
    the current prefill chunk plus its preceding 127 tokens until attention
    has finished. The compressed table maps compressed logical token pages.

    ``seqused_cmp_kv`` and ``cmp_residual_kv`` are derived buffers, not views:
    updating ``seqused_kv`` alone is insufficient. Rebuild/copy ALL scheduling
    and length buffers in place, preserving addresses across graph replay.
    """

    cu_seqlens_q: torch.Tensor
    seqused_kv: torch.Tensor
    swa_block_table: torch.Tensor
    schedule: torch.Tensor
    cmp_block_table: torch.Tensor | None = None
    seqused_cmp_kv: torch.Tensor | None = None
    cmp_residual_kv: torch.Tensor | None = None
    draft_swa_indices: torch.Tensor | None = None
    draft_swa_lengths: torch.Tensor | None = None


class AscendDSAV41Ops:
    """One joint SWA+CSA softmax, with one denominator-only sink per head.

    Uses the in-tree SparseFlashMla AscendC implementation. Its 910B path
    supports CR0/1/2 with implicit causal SWA and explicit compressed indices;
    simultaneous explicit SWA and compressed indices are not supported.
    """

    HEAD_DIM = 512
    TOPK = 512
    WINDOW = 128

    def __init__(self, compress_ratio: int, num_heads: int = 8) -> None:
        if compress_ratio not in (0, 1, 2):
            raise ValueError("V4.1 attention compress_ratio must be 0, 1 or 2")
        if num_heads not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("V4.1 local query heads must be a power of two up to 64")
        self.compress_ratio = compress_ratio
        self.num_heads = num_heads

    def build_metadata(
        self,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        swa_block_table: torch.Tensor,
        *,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        cmp_block_table: torch.Tensor | None = None,
        draft_swa_indices: torch.Tensor | None = None,
        draft_swa_lengths: torch.Tensor | None = None,
    ) -> AscendDSAV41Metadata:
        """Lengths/bounds count original tokens; no device-to-host reads."""
        if bool(self.compress_ratio) != (cmp_block_table is not None):
            raise ValueError("CR1/2 requires a compressed block table; CR0 forbids it")
        if cu_seqlens_q.shape != (seqused_kv.numel() + 1,):
            raise ValueError("cu_seqlens_q must contain batch_size + 1 offsets")
        if any(t.dtype != torch.int32 for t in (cu_seqlens_q, seqused_kv, swa_block_table)):
            raise ValueError("V4.1 attention lengths and tables must be INT32")
        if cmp_block_table is not None and cmp_block_table.dtype != torch.int32:
            raise ValueError("V4.1 compressed block table must be INT32")
        draft = draft_swa_indices is not None
        if draft != (draft_swa_lengths is not None):
            raise ValueError("Draft SWA indices and lengths must be provided together")
        if draft:
            if self.compress_ratio:
                raise ValueError("Explicit DSpark SWA indices require CR0")
            if draft_swa_indices.ndim != 3 or draft_swa_indices.shape[1:] != (1, 256):
                raise ValueError("Draft SWA indices must be [T,1,256]")
            if draft_swa_lengths.shape != draft_swa_indices.shape[:2]:
                raise ValueError("Draft SWA lengths must be [T,1]")
            if any(
                t.dtype != torch.int32 or t.device != seqused_kv.device for t in (draft_swa_indices, draft_swa_lengths)
            ):
                raise ValueError("Draft SWA buffers must be INT32 on the metadata device")
        draft_kwargs = {"ori_topk": 256, "ori_topk_length": draft_swa_lengths} if draft else {}
        cmp_lengths = seqused_kv // self.compress_ratio if self.compress_ratio else None
        residual = seqused_kv % self.compress_ratio if self.compress_ratio == 2 else None
        schedule = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
            num_heads_q=self.num_heads,
            num_heads_kv=1,
            head_dim=self.HEAD_DIM,
            cu_seqlens_q=cu_seqlens_q,
            seqused_ori_kv=seqused_kv,
            seqused_cmp_kv=cmp_lengths,
            cmp_residual_kv=residual,
            batch_size=seqused_kv.numel(),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_ori_kv=max_seqlen_kv,
            max_seqlen_cmp_kv=max_seqlen_kv // self.compress_ratio if self.compress_ratio else 0,
            cmp_topk=self.TOPK if self.compress_ratio else 0,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=0 if draft else 4,
            cmp_mask_mode=3,
            ori_win_left=self.WINDOW + 5 - 1 if draft else self.WINDOW - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=bool(self.compress_ratio),
            **draft_kwargs,
        )
        return AscendDSAV41Metadata(
            cu_seqlens_q,
            seqused_kv,
            swa_block_table,
            schedule,
            cmp_block_table,
            cmp_lengths,
            residual,
            draft_swa_indices,
            draft_swa_lengths,
        )

    def forward(
        self,
        query: torch.Tensor,
        swa_cache: torch.Tensor,
        sinks: torch.Tensor,
        metadata: AscendDSAV41Metadata,
        *,
        cmp_cache: torch.Tensor | None = None,
        cmp_indices: torch.Tensor | None = None,
        return_softmax_lse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [T,H,512] attention and optional [1,T,H] FP32 logsumexp.

        Compressed indices are logical row IDs [T,1,512], increasing and unique,
        with -1 padding only at the end. The selector must enforce per-query group
        completion. Caches are BF16 [blocks,block_size,1,512], with block_size a
        multiple of 16; both caches share K=V and use a single KV head. The caller
        must write current cache rows before calling this operation.
        """
        tokens = query.shape[0]
        draft = metadata.draft_swa_indices is not None
        if draft:
            if self.compress_ratio or metadata.draft_swa_lengths is None:
                raise ValueError("Draft SWA metadata requires CR0 and explicit lengths")
            if metadata.draft_swa_indices.shape != (tokens, 1, 256) or metadata.draft_swa_lengths.shape != (tokens, 1):
                raise ValueError("Draft SWA buffers must match the query token count")
        elif metadata.draft_swa_lengths is not None:
            raise ValueError("Draft SWA lengths require explicit indices")
        if query.shape != (tokens, self.num_heads, self.HEAD_DIM) or query.dtype != torch.bfloat16:
            raise ValueError("V4.1 attention query must be BF16 [T,local_heads,512]")
        if sinks.shape != (self.num_heads,) or sinks.dtype != torch.float32:
            raise ValueError("V4.1 attention sinks must be FP32 [local_heads]")
        if self.compress_ratio:
            if cmp_cache is None or cmp_indices is None or metadata.cmp_block_table is None:
                raise ValueError("CR1/2 requires compressed cache, indices and block table")
            if cmp_indices.shape != (tokens, 1, self.TOPK) or cmp_indices.dtype != torch.int32:
                raise ValueError("V4.1 compressed indices must be INT32 [T,1,512]")
        elif cmp_cache is not None or cmp_indices is not None or metadata.cmp_block_table is not None:
            raise ValueError("CR0 forbids compressed cache, indices and block table")
        for cache in (swa_cache, cmp_cache):
            if cache is None:
                continue
            if cache.ndim != 4 or cache.shape[2:] != (1, self.HEAD_DIM) or cache.dtype != torch.bfloat16:
                raise ValueError("V4.1 attention cache must be BF16 [blocks,block_size,1,512]")
            if cache.shape[1] % 16 or not 16 <= cache.shape[1] <= 1024:
                raise ValueError("V4.1 attention cache block_size must be a multiple of 16 in [16,1024]")
        if tokens == 0:
            lse_shape = (1, 0, self.num_heads) if return_softmax_lse else (0,)
            return torch.empty_like(query), torch.empty(lse_shape, dtype=torch.float32, device=query.device)
        draft_kwargs = (
            {"ori_sparse_indices": metadata.draft_swa_indices, "ori_topk_length": metadata.draft_swa_lengths}
            if draft
            else {}
        )
        return torch.ops._C_ascend.npu_sparse_flash_mla(
            query,
            ori_kv=swa_cache,
            cmp_kv=cmp_cache,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=metadata.swa_block_table,
            cmp_block_table=metadata.cmp_block_table,
            cu_seqlens_q=metadata.cu_seqlens_q,
            seqused_ori_kv=metadata.seqused_kv,
            seqused_cmp_kv=metadata.seqused_cmp_kv,
            cmp_residual_kv=metadata.cmp_residual_kv,
            sinks=sinks,
            metadata=metadata.schedule,
            softmax_scale=self.HEAD_DIM**-0.5,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=0 if draft else 4,
            cmp_mask_mode=3,
            ori_win_left=self.WINDOW + 5 - 1 if draft else self.WINDOW - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=return_softmax_lse,
            **draft_kwargs,
        )
