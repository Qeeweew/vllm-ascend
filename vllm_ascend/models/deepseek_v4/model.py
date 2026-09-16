# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import islice

import torch
import torch.nn.functional as F
import vllm.envs as envs
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache as VllmDeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.dsa_attn_kv_plan import get_dsv4_attn_kv_dtype
from vllm_ascend.core.kv_cache_interface import (
    AscendSlidingWindowMLASpec,
    AscendV41IndexerCacheSpec,
    AscendV41MainCacheSpec,
    AscendV41SWACacheSpec,
)
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.models.deepseek_v4.compressor import Compressor, CompressorV41, CompressorV41Metadata
from vllm_ascend.models.deepseek_v4.indexer import (
    AscendIndexerV41Metadata,
    AscendIndexerV41Ops,
    DeepseekV4Indexer,
    DeepseekV41IndexerProjections,
)
from vllm_ascend.ops.cache_v41 import write_index_cache_v41, write_main_cache_v41
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.ops.dsa_v41 import AscendDSAV41Metadata, AscendDSAV41Ops
from vllm_ascend.ops.engram_gate import engram_gate
from vllm_ascend.ops.mhc_v41 import mhc_collapse, mhc_post, mhc_pre_delayed
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.ops.v41_rope_cache import v41_index_cache_store, v41_main_cache_store, v41_rope
from vllm_ascend.patch.worker.patch_deepseek_v41_mm import (
    DeepseekV41VLDummyInputsBuilder,
    DeepseekV41VLMultiModalProcessor,
    DeepseekV41VLProcessingInfo,
)
from vllm_ascend.utils import (
    enable_custom_op,
    enable_dsa_cp,
    extract_dsv4_layer_index,
    get_dsv4_compress_ratio,
)
from vllm_ascend.worker.engram_image_mask import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, V41EngramImageSpans
from vllm_ascend.worker.v2.pp_utils import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_utils import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)

sequence_parallel_chunk = sp_shard


class DeepseekV41AttentionProjections(nn.Module):
    """BF16 attention projections with explicit FP32 adjacent-pair RoPE.

    Cache insertion and sparse attention are separate operations. Normalized
    low-rank Q is also the indexer's input; projected per-head Q has no second
    RMSNorm in V4.1. Output projection removes query RoPE before grouped wo_a.
    """

    def __init__(
        self,
        config,
        compress_ratio: int,
        max_position: int,
        prefix: str,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        if compress_ratio not in (0, 1, 2) or config.num_attention_heads % tp or config.o_groups % tp:
            raise ValueError("Unsupported V4.1 attention ratio or TP partition")
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        ascend_config = get_ascend_config()
        self.enable_fused_rope = getattr(ascend_config, "enable_v41_rope", False)
        self.enable_fused_cache_store = getattr(ascend_config, "enable_v41_cache_store", False)
        self.local_heads = config.num_attention_heads // tp
        self.local_groups = config.o_groups // tp
        self.q_rank = config.q_lora_rank
        self.o_rank = config.o_lora_rank
        self.group_width = config.num_attention_heads * self.head_dim // config.o_groups
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_rank, self.head_dim],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            disable_tp=True,
            return_bias=False,
            prefix=f"{prefix}.fused_wqa_wkv",
        )
        self.q_norm = RMSNorm(self.q_rank, config.rms_norm_eps, dtype=torch.bfloat16)
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=torch.bfloat16)
        self.wq_b = ColumnParallelLinear(
            self.q_rank,
            config.num_attention_heads * self.head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = ColumnParallelLinear(
            self.group_width,
            config.o_groups * self.o_rank,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        # wo_a is grouped, so the forward below explicitly reads its ND weight.
        # The Ascend loader may additionally transpose it into grouped BMM layout.
        self.wo_a.skip_weight_nz_conversion = True
        self.wo_b = RowParallelLinear(
            config.o_groups * self.o_rank,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.attn_sink = nn.Parameter(torch.empty(self.local_heads, dtype=torch.float32), requires_grad=False)
        if max_position < 1 or self.rope_dim % 2 or self.rope_dim > self.head_dim:
            raise ValueError("Invalid V4.1 rotary dimensions or context limit")
        if rope_cache is not None:
            if any(t.shape != (max_position, self.rope_dim // 2) or t.dtype != torch.float32 for t in rope_cache):
                raise ValueError("Shared V4.1 RoPE cache must match context, rotary width and FP32 dtype")
            self.register_buffer("rope_cos", rope_cache[0], persistent=False)
            self.register_buffer("rope_sin", rope_cache[1], persistent=False)
            return
        parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        key = "compress" if compress_ratio else "main"
        parameters = parameters.get(key, parameters)
        theta = config.compress_rope_theta if compress_ratio else config.rope_theta
        factor = parameters.get("factor", 1.0) if compress_ratio else 1.0
        original = parameters.get("original_max_position_embeddings", max_position)
        inv = ComplexExpRotaryEmbedding.precompute_freqs_cis(
            self.rope_dim,
            max_position,
            original,
            theta,
            factor,
            parameters.get("beta_fast", 32),
            parameters.get("beta_slow", 1),
        )
        if compress_ratio and not parameters.get("apply_yarn_scaling", True):
            inv = 1.0 / (factor * theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim))
        angles = torch.arange(max_position, dtype=torch.float32)[:, None] * inv[None]
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def rotate(self, value: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        """Rotate the last RoPE dimensions, preserving BF16 rounding boundaries."""
        if getattr(self, "enable_fused_rope", False):
            output = torch.empty_like(value, memory_format=torch.contiguous_format)
            return v41_rope(value, positions, self.rope_cos, self.rope_sin, output, inverse=inverse)
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        if value.ndim == 3:
            cos, sin = cos[:, None], sin[:, None]
        if inverse:
            sin = -sin
        pairs = value[..., -self.rope_dim :].float().unflatten(-1, (-1, 2))
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        return torch.cat((value[..., : -self.rope_dim], rotated.to(value.dtype)), dim=-1)

    def project_inputs(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, *, rotate_kv: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qr, kv = self.fused_wqa_wkv(hidden_states).split([self.q_rank, self.head_dim], dim=-1)
        qr = self.q_norm(qr.contiguous())
        kv = self.kv_norm(kv.contiguous())
        query = self.wq_b(qr).view(-1, self.local_heads, self.head_dim)
        return qr, self.rotate(query, positions), self.rotate(kv, positions) if rotate_kv else kv

    def project_output(self, attention: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        output = self.rotate(attention, positions, inverse=True)
        grouped = output.reshape(-1, self.local_groups, self.group_width).transpose(0, 1)
        weight = self.wo_a.weight
        if weight.ndim == 3:
            # AscendColumnParallelLinear's loader has already transformed ND
            # [groups * rank, width] into [groups, width, rank]. Reinterpreting
            # that storage as the original layout silently scrambles wo_a.
            if weight.shape != (self.local_groups, self.group_width, self.o_rank):
                raise ValueError("Unexpected V4.1 grouped wo_a weight layout")
        else:
            weight = weight.view(self.local_groups, self.o_rank, self.group_width).transpose(1, 2)
        output = torch.bmm(grouped, weight).transpose(0, 1).flatten(1)
        return self.wo_b(output)


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    """Register one explicit V4.1 storage layout with the cache planner."""

    def __init__(self, prefix: str, spec: KVCacheSpec, vllm_config: VllmConfig):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = torch.empty(0)
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache layer {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, cache):
        if isinstance(self.spec, AscendV41IndexerCacheSpec):
            if not isinstance(cache, (tuple, list)) or len(cache) != 2:
                raise ValueError("V4.1 index cache binding requires key and scale views")
            key, scale = cache
            if key.dtype != torch.int8 or scale.dtype != torch.float16 or scale.shape != key.shape[:-1]:
                raise ValueError("V4.1 index cache binding has an invalid key/scale layout")
            self.kv_cache = (key, scale)
        else:
            if not isinstance(cache, torch.Tensor) or cache.dtype != torch.bfloat16 or cache.shape[2:] != (1, 512):
                raise ValueError("V4.1 main/SWA cache must be BF16 [blocks,physical_rows,1,512]")
            self.kv_cache = cache

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return self.spec

    def get_attn_backend(self):
        from vllm_ascend.attention.dsa_v41 import AscendV41CacheBackend

        return AscendV41CacheBackend

    def forward(self):
        raise RuntimeError("V4.1 cache layers provide storage and are not called directly")


@dataclass
class DeepseekV41AttentionBatch:
    """Per-layer cache views and metadata; shared buffers retain stable addresses.

    Main/index cache tensors refer to the most recent KV source. Top-k and
    candidate buffers are shared across layers and overwritten only by their
    configured sources. A caller must refresh device metadata before replay.
    SWA pages must retain the current chunk and its preceding 127 tokens.
    """

    swa_cache: torch.Tensor
    swa_slots: torch.Tensor
    attention: AscendDSAV41Metadata
    main_cache: torch.Tensor | None = None
    main_slots: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    index_scale_cache: torch.Tensor | None = None
    index_slots: torch.Tensor | None = None
    topk: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    compressor: CompressorV41Metadata | None = None
    indexer: AscendIndexerV41Metadata | None = None


class DeepseekV41Attention(DeepseekV41AttentionProjections):
    """CSA2 source/consumer chain with separate projection and vector kernels."""

    def __init__(
        self,
        config,
        layer_id: int,
        max_position: int,
        prefix: str,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        vllm_config: VllmConfig | None = None,
        topk_buffer: torch.Tensor | None = None,
        candidate_buffer: torch.Tensor | None = None,
        *,
        is_draft_layer: bool = False,
    ):
        ratio = config.compress_ratios[layer_id]
        if is_draft_layer and ratio != 0:
            raise ValueError("V4.1 DSpark draft attention requires CR0")
        super().__init__(config, ratio, max_position, prefix, rope_cache)
        self.prefix = prefix
        self.is_draft_layer = is_draft_layer
        self.compress_ratio = ratio
        self.is_kv_source = not is_draft_layer and layer_id in config.kv_source_layer_ids
        self.is_index_source = not is_draft_layer and layer_id in config.index_source_layer_ids
        self.candidate_source = not is_draft_layer and layer_id == config.candidate_source_layer_id
        self.candidate_consumer = not is_draft_layer and 0 <= config.candidate_source_layer_id < layer_id
        if self.is_kv_source and not self.is_index_source:
            raise ValueError("V4.1 KV sources must also own an indexer")
        self.sparse = AscendDSAV41Ops(ratio, self.local_heads)
        self.compressor = (
            CompressorV41(config.hidden_size, ratio, config.rms_norm_eps, f"{prefix}.compressor")
            if self.is_kv_source
            else None
        )
        self.indexer = (
            DeepseekV41IndexerProjections(config, self.is_kv_source, f"{prefix}.indexer")
            if self.is_index_source
            else None
        )
        candidate_mode = "source" if self.candidate_source else "consumer" if self.candidate_consumer else "off"
        self.selector = (
            AscendIndexerV41Ops(ratio, candidate_mode, trusted_unique_candidates=candidate_mode == "consumer")
            if self.indexer is not None
            else None
        )
        self._topk_buffer = topk_buffer
        self._candidate_buffer = candidate_buffer
        self._cache_context = None
        if vllm_config is not None:
            self._cache_context = vllm_config.compilation_config.static_forward_context
            block_size = vllm_config.cache_config.block_size
            common = dict(block_size=block_size, num_kv_heads=1, head_size_v=0, alignment=None)
            self.swa_cache_layer = DeepseekV41CacheLayer(
                f"{prefix}.swa_cache",
                AscendV41SWACacheSpec(
                    **common,
                    head_size=512,
                    dtype=torch.bfloat16,
                    sliding_window=128,
                    extra_retained_tokens=vllm_config.num_speculative_tokens,
                ),
                vllm_config,
            )
            if ratio:
                source_id = max(source for source in config.kv_source_layer_ids if source <= layer_id)
                self._kv_source_prefix = prefix.replace(f".layers.{layer_id}.", f".layers.{source_id}.")
                if self.is_kv_source:
                    self.main_cache_layer = DeepseekV41CacheLayer(
                        f"{prefix}.main_cache",
                        AscendV41MainCacheSpec(
                            **common,
                            head_size=512,
                            dtype=torch.bfloat16,
                            tokens_per_state=ratio,
                        ),
                        vllm_config,
                    )
                    self.index_cache_layer = DeepseekV41CacheLayer(
                        f"{prefix}.indexer.k_cache",
                        AscendV41IndexerCacheSpec(
                            **common,
                            head_size=128,
                            dtype=torch.int8,
                            tokens_per_state=ratio,
                        ),
                        vllm_config,
                    )

    def _resolve_batch(self, metadata: dict) -> DeepseekV41AttentionBatch:
        from vllm_ascend.attention.dsa_v41 import make_v41_attention_metadata, make_v41_indexer_metadata

        swa = metadata[self.swa_cache_layer.prefix]
        if not self.compress_ratio:
            return DeepseekV41AttentionBatch(
                self.swa_cache_layer.kv_cache,
                swa.slot_mapping,
                make_v41_attention_metadata(swa),
            )
        main_prefix = f"{self._kv_source_prefix}.main_cache"
        index_prefix = f"{self._kv_source_prefix}.indexer.k_cache"
        main, index = metadata[main_prefix], metadata[index_prefix]
        key_cache, scale_cache = self._cache_context[index_prefix].kv_cache
        compressor = None
        if self.compressor is not None:
            if self.compressor.state_cache is not None:
                compressor = metadata[self.compressor.state_cache.prefix]
            else:
                compressor = CompressorV41Metadata(main.slot_mapping, main.cu_seqlens_q, main.token_to_req_indices)
        return DeepseekV41AttentionBatch(
            self.swa_cache_layer.kv_cache,
            swa.slot_mapping,
            make_v41_attention_metadata(swa, main),
            main_cache=self._cache_context[main_prefix].kv_cache,
            main_slots=main.slot_mapping,
            index_cache=key_cache,
            index_scale_cache=scale_cache,
            index_slots=index.slot_mapping,
            topk=self._topk_buffer,
            candidates=self._candidate_buffer,
            compressor=compressor,
            indexer=make_v41_indexer_metadata(index) if self.indexer is not None else None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        batch: DeepseekV41AttentionBatch | None = None,
    ) -> torch.Tensor:
        if batch is None:
            context = get_forward_context()
            metadata = context.attn_metadata
            if metadata is None and getattr(context, "in_profile_run", False):
                # Initial memory profiling precedes cache allocation. Exercise
                # all projection GEMMs without attempting an unbound cache
                # read. Actual graph capture requires built cache metadata.
                qr, query, _ = self.project_inputs(hidden_states, positions)
                if self.compressor is not None:
                    self.compressor.project(hidden_states)
                    self.indexer.project_key(torch.zeros_like(query[:, 0]))
                if self.indexer is not None:
                    index_query, _ = self.indexer.project_query(hidden_states, qr)
                    self.selector.quantize(self.rotate(index_query, positions))
                return self.project_output(torch.zeros_like(query), positions)
            if isinstance(metadata, dict) and self._cache_context is not None:
                batch = self._resolve_batch(metadata)
            elif isinstance(metadata, dict) and isinstance(metadata.get(self.prefix), DeepseekV41AttentionBatch):
                batch = metadata[self.prefix]
            else:
                raise ValueError("V4.1 attention requires its own cache/metadata preparation")
        if self.is_draft_layer != (batch.attention.draft_swa_indices is not None):
            raise ValueError("V4.1 draft attention requires explicit noncausal metadata; target attention forbids it")
        tokens = hidden_states.shape[0]
        if self.enable_fused_cache_store:
            qr, query, swa = self.project_inputs(hidden_states, positions, rotate_kv=False)
            v41_main_cache_store(swa, positions, batch.swa_slots, self.rope_cos, self.rope_sin, batch.swa_cache)
        else:
            qr, query, swa = self.project_inputs(hidden_states, positions)
            write_main_cache_v41(batch.swa_cache, swa, batch.swa_slots)
        if self.compressor is not None:
            if batch.compressor is None or batch.main_slots is None or batch.index_slots is None:
                raise ValueError("V4.1 KV source requires compressor metadata and main/index write slots")
            projected = self.compressor.project(hidden_states)
            latent = torch.empty((tokens, 512), dtype=torch.bfloat16, device=hidden_states.device)
            self.compressor(projected, positions, batch.compressor, latent)
            key = self.indexer.project_key(latent)
            if self.enable_fused_cache_store:
                v41_index_cache_store(
                    key,
                    positions,
                    batch.index_slots,
                    self.rope_cos,
                    self.rope_sin,
                    batch.index_cache,
                    batch.index_scale_cache,
                    compress_ratio=self.compress_ratio,
                )
                v41_main_cache_store(
                    latent,
                    positions,
                    batch.main_slots,
                    self.rope_cos,
                    self.rope_sin,
                    batch.main_cache,
                    compress_ratio=self.compress_ratio,
                )
            else:
                group_positions = (positions // self.compress_ratio) * self.compress_ratio
                key, scale = self.selector.quantize(self.rotate(key, group_positions))
                write_index_cache_v41(
                    batch.index_cache,
                    batch.index_scale_cache,
                    key,
                    scale,
                    batch.index_slots,
                    positions=positions,
                    compress_ratio=self.compress_ratio,
                )
                main = self.rotate(latent, group_positions)
                write_main_cache_v41(
                    batch.main_cache,
                    main,
                    batch.main_slots,
                    positions=positions,
                    compress_ratio=self.compress_ratio,
                )
        if self.indexer is not None:
            if batch.indexer is None or batch.topk is None:
                raise ValueError("V4.1 index source requires indexer metadata and shared top-k buffer")
            index_query, weights = self.indexer.project_query(hidden_states, qr)
            index_query, scale = self.selector.quantize(self.rotate(index_query, positions))
            topk, candidates = self.selector.select_topk(
                index_query,
                weights,
                scale,
                batch.index_cache,
                batch.index_scale_cache,
                batch.indexer,
                batch.candidates[:tokens] if self.candidate_consumer else None,
            )
            batch.topk[:tokens].copy_(topk)
            if self.candidate_source:
                if batch.candidates is None:
                    raise ValueError("V4.1 candidate source requires a shared candidate buffer")
                batch.candidates[:tokens].copy_(candidates)
        attention, _ = self.sparse.forward(
            query,
            batch.swa_cache,
            self.attn_sink,
            batch.attention,
            cmp_cache=batch.main_cache if self.compress_ratio else None,
            cmp_indices=batch.topk[:tokens] if self.compress_ratio else None,
        )
        return self.project_output(attention, positions)


class EngramV41(nn.Module):
    """Device half of Engram; its large embedding table belongs to the host.

    The runner stages each rank's hash-head rows before invoking the model or
    replaying its graph. This component gathers heads, projects, and gates the
    HC stream. It performs no Python lookup or host transfer during forward.
    """

    def __init__(self, config, prefix: str):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hash_heads = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.head_dim = config.engram_head_dim
        if self.hash_heads % self.tp_size:
            raise ValueError("Engram hash heads must divide evenly across TP ranks")
        if (config.hc_mult, config.hidden_size) != (4, 5120):
            raise ValueError("V4.1 Engram gate requires HC4 and hidden size 5120")
        self.local_heads = self.hash_heads // self.tp_size
        self.eps = config.rms_norm_eps
        self.wkv = ReplicatedLinear(
            self.hash_heads * self.head_dim,
            (config.hc_mult + 1) * config.hidden_size,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            return_bias=False,
            prefix=f"{prefix}.wkv",
        )
        self.q_weight = nn.Parameter(torch.empty(4, 5120, dtype=torch.bfloat16), requires_grad=False)
        self.k_weight = nn.Parameter(torch.empty(4, 5120, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, staged_rows: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        if staged_rows.shape != (tokens, self.local_heads, self.head_dim):
            raise ValueError("Engram staging must contain this rank's heads for the current token bucket")
        rows = tensor_model_parallel_all_gather(staged_rows, dim=1) if self.tp_size > 1 else staged_rows
        kv = self.wkv(rows.flatten(1))
        return engram_gate(hidden_states, kv, self.q_weight, self.k_weight, token_mask, self.eps)


class AscendDeepseekV4SWACache(VllmDeepseekV4SWACache):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__(head_dim, window_size, torch.uint8, prefix, cache_config)
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        self.dtype = dtype

        self.block_size = DSV4_BLOCK_SIZES[cache_config.block_size][0][1]

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        self.dtype = get_dsv4_attn_kv_dtype(vllm_config)
        if self.dtype == torch.float8_e4m3fn:
            vllm_config.cache_config.cache_dtype = "float8_e4m3fn"
        cached_head_size = self.head_dim + 128 if self.dtype == torch.float8_e4m3fn else self.head_dim
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=cached_head_size,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version="deepseek_v4",
            alignment=None,
        )

    def forward(self): ...

    def get_attn_backend(self):
        from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend

        return AscendDSASWABackend


def precompute_freqs_cis_cpu(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis.to(x.device)).flatten(-2)
    y.copy_(x)
    return y


def get_spec_layer_idx_from_weight_name(config: DeepseekV2Config | DeepseekV3Config, weight_name: str) -> int | None:
    if weight_name.startswith("mtp."):
        return 0
    return None


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_draft_layer: bool = False,
        image_sentinel_lo: int = 129257,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)

        self.ep_group = get_ep_group().device_group
        # vLLM creates the EP communication group even for TP-only MoE.
        # Expert ownership must follow the configured partition, not that
        # group's physical size: TP shards matrices while retaining all IDs.
        self.ep_rank = get_ep_group().rank_in_group if parallel_config.enable_expert_parallel else 0
        self.ep_size = self.ep_group.size() if parallel_config.enable_expert_parallel else 1
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. Only silu is supported for now.")

        self.gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False, quant_config=None, prefix=f"{prefix}.gate"
        )
        self.gate.precast_fp32_weight = True

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        self.is_fusion_moe_shared_experts_enabled = getattr(get_ascend_config(), "mix_placement", False)
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.hash = layer_idx < getattr(config, "num_hash_layers", 0) and not is_draft_layer
        self.gate.bias_vl = None
        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(
                    config.n_routed_experts,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        if self.hash:
            # Use zeros instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            # Keep scaling outside the router path so the order matches
            # DeepSeek V4: normalize top-k weights, then scale routed output.
            # AITER applies routed_scaling_factor internally.
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=image_sentinel_lo,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts if self.is_fusion_moe_shared_experts_enabled else 0,
            hash_indices_table=self.gate.tid2eid,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if hidden_states_fp32 is not None:
            hidden_states_fp32 = hidden_states_fp32.view(-1, hidden_dim)

        modality_kwargs = {} if image_token_mask is None else {"image_token_mask": image_token_mask}
        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoEFactory class
            router_input = hidden_states if hidden_states_fp32 is None else hidden_states_fp32
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_input,
                input_ids=input_ids,
                **modality_kwargs,
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_input = hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
            router_logits = F.linear(router_input, self.gate.weight)
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
                **modality_kwargs,
            )

        fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
        if fused_moe_out_is_tuple:
            shared_output, final_hidden_states = fused_moe_out
            if self.shared_experts is None:
                assert shared_output is None

            if hidden_states.dtype != torch.float16:
                if not self.is_rocm_aiter_moe_enabled:
                    if self.shared_experts is not None:
                        assert shared_output is not None
                        final_hidden_states = muls_add_triton(
                            final_hidden_states, shared_output, self.routed_scaling_factor
                        )
                    else:
                        final_hidden_states *= self.routed_scaling_factor
            elif self.shared_experts is not None:
                assert shared_output is not None
                final_hidden_states = muls_add_triton(
                    shared_output, final_hidden_states, 1.0 / self.routed_scaling_factor
                )
        else:
            final_hidden_states = fused_moe_out

        if not self.is_sequence_parallel and self.tp_size > 1 and fused_moe_out_is_tuple:
            # Legacy tuple outputs are reduced here. Tensor outputs from the
            # upstream MoERunner have already gone through its final reduction.
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(1 + torch.floor(positions / original_max_position_embeddings))
    # Broadcast over num_heads and head_dim
    return scaling[..., None, None]


class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        max_position_embeddings: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        reduce_results: bool = True,
        need_gather_q_kv: bool = False,
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        config_layer_idx = extract_dsv4_layer_index(config, prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = config.num_attention_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps
        self.norm_eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.enable_dsa_cp = enable_dsa_cp()

        attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads
        self.attn_sink = nn.Parameter(torch.empty(attn_sink_heads, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
            return_bias=False,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_norm_without_weight = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
        wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
        self.wq_b = wq_b_cls(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )

        self.wkv = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.kv_norm = RMSNorm(self.head_dim, self.norm_eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * config.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_a",
            return_bias=False,
        )
        # Every DSA o_proj path consumes wo_a.weight directly via
        # npu_transpose_batchmatmul / npu_transpose_quant_batchmatmul,
        # so the weight must remain ND.
        self.wo_a.skip_weight_nz_conversion = True
        self.wo_b = RowParallelLinear(
            self.n_groups * config.o_lora_rank,
            self.dim,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
            return_bias=False,
        )
        self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)

        if self.compress_ratio > 1:
            config.rope_parameters["rope_theta"] = config.compress_rope_theta
            rope_groups = ["default", f"c{self.compress_ratio}"]
        else:
            config.rope_parameters["rope_theta"] = config.rope_theta
            rope_groups = ["default"]
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=config.rope_parameters["rope_theta"],
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            rope_groups=rope_groups,
        )

        self.compressor: Compressor | None = None
        self.indexer: DeepseekV4Indexer | None = None

        use_index_cache = getattr(config, "use_index_cache", False)

        # IndexCache: decide whether this layer reuses topk from a previous
        # indexer-bearing layer. Refer: https://arxiv.org/abs/2603.12201
        # Only meaningful when this layer actually owns an Indexer (c4) and
        # IndexCache is enabled via hf-overrides. MTP layers are excluded
        # because spec_decode shares topk_indices_buffer at the model level
        # only, leaving impl-level references stale.
        skip_topk = False
        if self.compress_ratio == 4 and use_index_cache and ".mtp." not in prefix:
            compress_ratios = getattr(config, "compress_ratios", None) or []
            indexer_seq_idx = sum(1 for r in compress_ratios[:config_layer_idx] if r == 4)
            pattern = getattr(config, "index_topk_pattern", None)
            freq = getattr(config, "index_topk_freq", 1)
            if pattern is None:
                skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0
            else:
                assert pattern[0] == "F", "index_topk_pattern must start with 'F'"
                if 0 <= indexer_seq_idx < len(pattern):
                    skip_topk = pattern[indexer_seq_idx] == "S"

        if self.compress_ratio > 1:
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                head_dim=self.head_dim,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
            )  # Compressor(4, 128)

            if self.compress_ratio == 4:
                self.indexer = DeepseekV4Indexer(
                    vllm_config,
                    config,
                    self.compress_ratio,
                    skip_topk=skip_topk,
                    use_index_cache=use_index_cache,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    prefix=f"{prefix}.indexer",
                    topk_indices_buffer=topk_indices_buffer,
                )

        k_dtype = get_dsv4_attn_kv_dtype(vllm_config)
        swa_cache_layer = AscendDeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=k_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        dsa_modules = DSAModules(
            wq_a=self.wq_a,
            q_norm=self.q_norm,
            q_norm_without_weight=self.q_norm_without_weight,
            wq_b=self.wq_b,
            wkv=self.wkv,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            indexer=self.indexer,
            compressor=self.compressor,
            swa_cache_layer=swa_cache_layer,
        )

        self.dsa_attn = AscendDeepseekSparseAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            eps=self.eps,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            dsa_modules=dsa_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            # prefix=f'{prefix}.attn',
            prefix=f"{prefix}",
            need_gather_q_kv=need_gather_q_kv,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.dsa_attn(positions, hidden_states, llama_4_scaling)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters["original_max_position_embeddings"]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = enable_dsa_cp()  # TODO: delete this when enable_dsa_cp is sunset.

        attn_cls = DeepseekV4Attention

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
            need_gather_q_kv=self.use_sequence_parallel_moe and self.enable_dsa_cp,
        )

        self.mlp = DeepseekV4MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def rms_norm_cast(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize once and provide the exact FP32 routing input."""
        if enable_custom_op():
            return torch.ops._C_ascend.npu_rms_norm_cast(
                hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, hidden_states.float()

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        y = torch.ops._C_ascend.npu_hc_pre_v2(
            x, hc_fn, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps
        )
        return y

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0), residual.unsqueeze(dim=0), post.unsqueeze(dim=0), comb.unsqueeze(dim=0)
        )
        return y.squeeze(dim=0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states.clone()
        full_num_tokens = positions.shape[0]
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        attn_kwargs = {"positions": positions, "hidden_states": hidden_states, "llama_4_scaling": llama_4_scaling}
        hidden_states = self.self_attn(**attn_kwargs)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        hidden_states, hidden_states_fp32 = self.rms_norm_cast(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        return hidden_states, residual


class DeepseekV41DecoderLayer(nn.Module):
    """V4.1 block with explicit CSA2/MoE components and delayed mHC state."""

    def __init__(self, config, attention: nn.Module, moe: nn.Module, engram: EngramV41 | None = None):
        super().__init__()
        self.self_attn = attention
        self.mlp = moe
        self.engram = engram
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        for sublayer in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{sublayer}_fn", nn.Parameter(torch.empty(24, 20480, dtype=torch.float32), requires_grad=False)
            )
            self.register_parameter(
                f"hc_{sublayer}_base", nn.Parameter(torch.empty(24, dtype=torch.float32), requires_grad=False)
            )
            self.register_parameter(
                f"hc_{sublayer}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        engram_rows: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.engram is not None:
            if engram_rows is None or token_mask is None:
                raise ValueError("Engram rows and token mask must be prepared before the model invocation")
            hidden_states = self.engram(hidden_states, engram_rows, token_mask)
        residual = hidden_states
        collapsed, post, comb, next_pre = mhc_pre_delayed(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
            self.norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )
        output = self.self_attn(positions=positions, hidden_states=self.input_layernorm(collapsed))
        hidden_states = mhc_post(output, residual, post, comb)
        residual = hidden_states
        collapsed, post, comb, pre_mix = mhc_pre_delayed(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            next_pre,
            self.norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )
        normalized = self.post_attention_layernorm(collapsed)
        output = self.mlp(
            normalized, input_ids=input_ids, hidden_states_fp32=normalized.float(), image_token_mask=image_token_mask
        )
        return mhc_post(output, residual, post, comb), pre_mix


class DeepseekV41Model(nn.Module, EagleModelMixin):
    """V4.1 text backbone with delayed mixes and externally staged Engram rows.

    This TP8 backbone uses its own attention/cache contract. Multimodal input
    embedding preparation and the outer causal-LM weight loader are separate.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config = vllm_config.model_config.hf_config
        parallel = vllm_config.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.tensor_parallel_size != 8:
            raise ValueError("The V4.1 910B backbone currently requires TP8 and PP1")
        if parallel.use_sequence_parallel_moe or parallel.enable_expert_parallel:
            raise ValueError("V4.1 staged Engram rows currently require replicated token rows and TP MoE")
        if config.hc_mult != 4:
            raise ValueError("V4.1 backbone requires four hyper-connection streams")
        self.config = config
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.engram_layer_ids = tuple(config.engram_layer_ids)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList()
        maximum = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer("topk_indices", torch.full((maximum, 1, 512), -1, dtype=torch.int32), persistent=False)
        self.register_buffer(
            "candidate_blocks", torch.full((maximum, 1, 2048), -1, dtype=torch.int32), persistent=False
        )
        # Share only within this model, avoiding mutable process-global state.
        # Construct these on the final device so later .to(device) is a no-op
        # and does not duplicate the two cache tensors per layer.
        rope_caches = {}
        for layer_id in range(config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{layer_id}"
            rope_kind = bool(config.compress_ratios[layer_id])
            attention = DeepseekV41Attention(
                config,
                layer_id,
                vllm_config.model_config.max_model_len,
                f"{layer_prefix}.self_attn",
                rope_cache=rope_caches.get(rope_kind),
                vllm_config=vllm_config,
                topk_buffer=self.topk_indices,
                candidate_buffer=self.candidate_blocks,
            )
            rope_caches.setdefault(rope_kind, (attention.rope_cos, attention.rope_sin))
            moe = DeepseekV4MoE(
                config,
                parallel,
                vllm_config.quant_config,
                prefix=f"{layer_prefix}.mlp",
                image_sentinel_lo=getattr(config, "image_token_id", 129264),
            )
            engram = EngramV41(config, f"{layer_prefix}.engram") if layer_id in self.engram_layer_ids else None
            self.layers.append(DeepseekV41DecoderLayer(config, attention, moe, engram))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if any(type(layer) is not int or not 1 <= layer <= len(self.layers) for layer in layers):
            raise ValueError("V4.1 auxiliary layer IDs must be one-based backbone layer indices")
        super()._set_aux_hidden_state_layers(layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        engram_rows: tuple[torch.Tensor, ...] | None = None,
        engram_token_mask: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("V4.1 PP intermediates are not enabled")
        if self.engram_layer_ids and (engram_rows is None or len(engram_rows) != len(self.engram_layer_ids)):
            raise ValueError("V4.1 requires staged rows for each configured Engram layer")
        hidden = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        if image_token_mask is None:
            image_token_mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        if (
            image_token_mask.dtype != torch.bool
            or image_token_mask.shape != (hidden.shape[0],)
            or image_token_mask.device != hidden.device
        ):
            raise ValueError("V4.1 image token mask must be bool with one entry per packed token on the model device")
        hidden = hidden[:, None, :].expand(-1, 4, -1).contiguous()
        pre = torch.zeros((hidden.shape[0], 4), dtype=torch.float32, device=hidden.device)
        pre[:, 0] = 1
        rows_by_layer = dict(zip(self.engram_layer_ids, engram_rows or ()))
        aux_hidden_states = []
        for layer_id, layer in enumerate(self.layers):
            hidden, pre = layer(
                positions,
                hidden,
                pre,
                input_ids=input_ids,
                engram_rows=rows_by_layer.get(layer_id),
                token_mask=engram_token_mask,
                image_token_mask=image_token_mask,
            )
            if layer_id + 1 in self.aux_hidden_state_layers:
                # Official V4.1 DSpark takes post-FFN HC means, before the
                # next layer's Engram injection and before final collapse/norm.
                aux_hidden_states.append(hidden.mean(dim=1))
        hidden = self.norm(mhc_collapse(hidden, pre))
        return (hidden, aux_hidden_states) if aux_hidden_states else hidden


@support_torch_compile
class DeepseekV4Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False
    # vLLM #50514 validates and relays the model's existing PP aux payload.
    supports_aux_hidden_states_over_pp = True
    AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        # Expose at model level so spec_decode/llm_base_proposer can share
        # this buffer with the MTP draft via attribute replacement.
        self.topk_indices_buffer = topk_indices_buffer

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(vllm_config, prefix, topk_indices_buffer=topk_indices_buffer),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        self.make_empty_intermediate_tensors = make_pp_empty_intermediate_tensors(
            self,
            make_empty_intermediate_tensors,
        )

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_norm = RMSNorm(hc_dim, eps=config.rms_norm_eps, has_weight=False, dtype=torch.float32)

        # Pre-hc_head residual stream buffer for the speculative draft
        # (MTP / DSpark / DFlash). Only needed when the decoder consumes
        # target-model hidden states; allocating it unconditionally would
        # permanently cost max_num_batched_tokens * hc_dim per rank.
        # Aligned with upstream DeepSeekV4 (see vllm PR #50312).
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        self._mtp_hidden_buffer = (
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
            if get_pp_group().is_last_rank and needs_mtp_hidden_states
            else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        x_norm = self.hc_norm(x)
        mixes = torch.nn.functional.linear(x_norm, hc_fn)
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None
        aux_hidden_states = get_pp_transport_tensors(
            intermediate_tensors,
            PPTransportDataType.AUX_HIDDEN_STATES,
        )

        if self.use_sequence_parallel_moe:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(forward_context.is_padding, hidden_states)
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)  # TODO: support PP with dsacp.

        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling_config = None
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        if pp_group.is_first_rank:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)  # (b, s, h) -> (b, s, c, h)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
                input_ids=input_ids,
            )
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states.mean(dim=1)
                if self.use_sequence_parallel_moe:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)

        if not pp_group.is_last_rank:
            intermediate_tensors = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )
            return add_pp_transport_tensors(
                intermediate_tensors,
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )

        if self.use_sequence_parallel_moe:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)

        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class DeepseekV2MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekV4MoE]
    """
    List of MoE MLP layers in the model.
    """

    def extract_moe_parameters(self, example_moe: DeepseekV4MoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class AscendDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts, SupportsLoRA, SupportsEagle3):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls = DeepseekV4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # Set MoE hyperparameters
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV4DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if getattr(get_ascend_config(), "mix_placement", False) else 0),
            num_redundant_experts=0,
        )

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model._set_aux_hidden_state_layers(layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        rocm_aiter_moe_shared_expert_enabled = getattr(get_ascend_config(), "mix_placement", False)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if rocm_aiter_moe_shared_expert_enabled else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # TODO:
            if not name.startswith("model"):
                name = f"model.{name}"

            if ".w1." in name:
                name = name.replace(".w1.", ".gate_proj.")
            if ".w2." in name:
                name = name.replace(".w2.", ".down_proj.")
            if ".w3." in name:
                name = name.replace(".w3.", ".up_proj.")

            if "model.head." in name and "model.lm_head." not in name:
                name = name.replace("model.head.", "lm_head.")
            if "model.lm_head." in name:
                name = name.replace("model.lm_head.", "lm_head.")
            if "embed." in name and "embed_token." not in name:
                name = name.replace("embed.", "embed_tokens.")
            if "attn" in name and "self_attn" not in name:
                name = name.replace(".attn.", ".self_attn.")
            if ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")
            if ".ffn_norm." in name:
                name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
            if ".attn_norm." in name:
                name = name.replace(".attn_norm.", ".input_layernorm.")
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            if "rotary_emb.inv_freq" in name:
                continue
            if ".gate.bias_vl" in name:
                # The parameter keeps the checkpoint name on Ascend. It is
                # passed to the hash router as its vision-only correction
                # bias, while text rows continue to use tid2eid.
                pass
            elif ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            # Hash-router layers route text tokens through ``tid2eid`` and keep
            # ``e_score_correction_bias`` unset, but the checkpoint still ships
            # a router bias for them. Skip it instead of raising a KeyError.
            if name.endswith(".gate.e_score_correction_bias") and name not in params_dict:
                continue

            if "sink" in name:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                if enable_dsa_cp():
                    param.data.copy_(loaded_weight)
                else:
                    # Handle attention sinks (distributed across ranks)
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                    param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            is_fusion_moe_shared_experts_layer = rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        if split_dim == 0:
                            weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                        else:
                            weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params


class AscendDeepseekV41ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts, SupportsEagle3):
    """V4.1 TP8 text entry with checkpoint-preserving INT4 expert loading.

    Engram embedding matrices are deliberately excluded from device parameter
    loading and are loaded by the runner's host-offload initialization. Vision
    and speculative towers belong to their separate model wrappers.
    """

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        weight_format = getattr(self.config, "ascend_weight_format", {})
        if (
            weight_format.get("group_size") != 32
            or weight_format.get("signed_scale") is not True
            or weight_format.get("checkpoint_packing") != "offset_binary_q_plus_8"
            or weight_format.get("engram_dtype") != "bfloat16"
        ):
            raise ValueError("Convert V4.1 weights to Ascend signed-scale INT4 group32/BF16 before loading")
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekV41Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.num_moe_layers = self.config.num_hidden_layers
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.expert_weights = []
        self.moe_mlp_layers = [layer.mlp for layer in self.model.layers]
        self.moe_layers = [layer.experts for layer in self.moe_mlp_layers]
        self.extract_moe_parameters(self.moe_mlp_layers[-1] if self.moe_mlp_layers else None)

    def create_engram_runtime(self):
        # Worker-only initialization, after device parameters have been loaded.
        import json
        from pathlib import Path

        from safetensors import safe_open
        from transformers import AutoTokenizer

        from vllm_ascend.ops.engram_hash import HostEngramHasher
        from vllm_ascend.ops.engram_offload import EngramOffloadManager, EngramTableShard
        from vllm_ascend.worker.engram_history import EngramRequestHistory
        from vllm_ascend.worker.engram_runtime import EngramRuntime

        if not self.config.engram_layer_ids:
            return None
        root = Path(self.vllm_config.model_config.model)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        table_names = [f"layers.{layer_id}.engram.embed.weight" for layer_id in self.config.engram_layer_ids]
        # Validate every table before pinning any large allocation. Inspecting
        # safetensors slices only reads headers; it does not load table rows.
        for name, rows in zip(table_names, self.config.engram_num_embeddings, strict=True):
            if name not in index:
                raise ValueError(f"Missing converted Engram table {name}")
            with safe_open(root / index[name], framework="pt", device="cpu") as reader:
                table = reader.get_slice(name)
                if table.get_dtype() != "BF16" or table.get_shape() != [rows, self.config.engram_head_dim]:
                    raise ValueError(
                        f"Converted Engram table {name} must be BF16 [{rows}, {self.config.engram_head_dim}]"
                    )
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            trust_remote_code=self.vllm_config.model_config.trust_remote_code,
        )
        hasher = HostEngramHasher.from_tokenizer(self.config, tokenizer)
        rank, world = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        device = next(self.model.parameters()).device
        numa_nodes = get_ascend_config().engram_numa_nodes
        placement = {} if numa_nodes is None else {"numa_node": numa_nodes[rank], "device": device}
        shards = []
        try:
            for layer in range(len(hasher.layout.layer_ids)):
                name = table_names[layer]
                heads, ranges = hasher.layout.head_shard(layer, rank, world)
                shards.append(EngramTableShard.from_safetensors(root / index[name], name, heads, ranges, **placement))
            manager = EngramOffloadManager(
                shards,
                self.vllm_config.scheduler_config.max_num_batched_tokens,
                device,
            )
        except Exception:
            for shard in shards:
                shard.close()
            raise
        try:
            return EngramRuntime(EngramRequestHistory(hasher), manager)
        except Exception:
            manager.close()
            raise

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return tuple(layer + 1 for layer in self.config.dspark_target_layer_ids)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return AscendDeepseekV4ForCausalLM.get_expert_mapping(self)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        direct = set()

        def remaining_weights():
            for name, weight in weights:
                native = name.removeprefix("model.")
                if (
                    native.startswith(("vision.", "aligner.", "mtp."))
                    or native in {"image_start", "image_end", "image_newline"}
                    or ".engram.embed." in native
                ):
                    continue
                name = native if native.startswith("lm_head.") else f"model.{native}"
                name = name.replace(".attn.", ".self_attn.")
                # These replicated projections load their component directly
                # into a fused ND matrix; no full-matrix concat temporary.
                fusions = (
                    (".self_attn.wq_a.weight", ".self_attn.fused_wqa_wkv.weight", 0),
                    (".self_attn.wkv.weight", ".self_attn.fused_wqa_wkv.weight", 1),
                    (".compressor.wkv.weight", ".compressor.fused_wkv_wgate.weight", 0),
                    (".compressor.wgate.weight", ".compressor.fused_wkv_wgate.weight", 1),
                )
                for source, destination, shard in fusions:
                    if name.endswith(source):
                        target = name.removesuffix(source) + destination
                        parameter = params[target]
                        parameter.weight_loader(parameter, weight, shard)
                        direct.add(target)
                        break
                else:
                    # The existing expert-aware path handles packed INT4,
                    # group scales and shape metadata with TP-aware callbacks.
                    yield name, weight

        loaded = AscendDeepseekV4ForCausalLM.load_weights(self, remaining_weights())
        return loaded | direct


class AscendV41VisionRMSNorm(nn.Module):
    """Released V4.1 vision norm: FP32 arithmetic, epsilon 1e-6."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return (self.weight.float() * value).to(x.dtype)


def _v41_vision_cos_sin(n_h: int, n_w: int, head_dim: int, theta: float, device: torch.device):
    # Explicit device ownership avoids the reference's global default-device
    # and process-global LRU cache assumptions. Reused by every tower block.
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _v41_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class AscendV41VisionSDPA(nn.Module):
    """One-image numerical path; input and output use [1,N,H,D]."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Match the released [H,N,D] SDPA dispatch. Adding a batch dimension
        # can select a different BF16 CPU implementation and rounding order.
        return (
            F.scaled_dot_product_attention(
                q[0].transpose(0, 1),
                k[0].transpose(0, 1),
                v[0].transpose(0, 1),
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )


class AscendV41VisionAttention(nn.Module):
    def __init__(
        self,
        config,
        dtype: torch.dtype,
        attention_factory: Callable[[int, int], nn.Module] | None,
    ):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // self.n_heads
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim, dtype=dtype)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim, dtype=dtype)
        # An injected AscendMMEncoderAttention uses full bidirectional FIA.
        # It is constructed before any forward/capture, not lazily in a graph.
        self.attention = (
            AscendV41VisionSDPA() if attention_factory is None else attention_factory(self.n_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        tokens = x.shape[0]
        q, k, v = (value.view(tokens, self.n_heads, self.head_dim) for value in self.wqkv(x).chunk(3, dim=-1))
        q, k = _v41_vision_rotary(q, cos, sin), _v41_vision_rotary(k, cos, sin)
        output = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        return self.wo(output.reshape(tokens, -1))


class AscendV41VisionMLP(nn.Module):
    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False, dtype=dtype)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class AscendV41VisionBlock(nn.Module):
    def __init__(
        self,
        config,
        dtype: torch.dtype,
        attention_factory: Callable[[int, int], nn.Module] | None,
    ):
        super().__init__()
        self.norm1 = AscendV41VisionRMSNorm(config.vision_dim)
        self.attn = AscendV41VisionAttention(config, dtype, attention_factory)
        self.norm2 = AscendV41VisionRMSNorm(config.vision_dim)
        self.mlp = AscendV41VisionMLP(config, dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class AscendV41VisionPatchEmbed(nn.Module):
    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim, dtype=dtype)

    def forward(self, patches: torch.Tensor):
        return self.proj(patches.flatten(1))


class AscendV41VisionTower(nn.Module):
    """V4.1 ViT over one image, independent of LM/Engram and CUDA models.

    Parameters are replicated. No TP collectives or implicit image sharding
    occur here. A wrapper owns multimodal scheduling and can inject an NPU
    encoder attention factory accepting (num_heads, head_dim) at construction.
    Local checkpoint names match the released ``vision.`` subtree exactly.
    """

    def __init__(
        self,
        config,
        *,
        dtype: torch.dtype = torch.bfloat16,
        attention_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("V4.1 vision uses BF16 weights or an FP32 reference")
        if (
            config.vision_dim <= 0
            or config.vision_n_heads <= 0
            or config.vision_dim % config.vision_n_heads
            or (config.vision_dim // config.vision_n_heads) % 4
            or config.vision_n_layers <= 0
            or config.vision_patch_size <= 0
            or config.vision_inter_dim <= 0
            or not math.isfinite(config.vision_rope_theta)
            or config.vision_rope_theta <= 0
        ):
            raise ValueError("Invalid V4.1 vision geometry or 2D RoPE configuration")
        self.patch_size = config.vision_patch_size
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = AscendV41VisionPatchEmbed(config, dtype)
        self.blocks = nn.ModuleList(
            [AscendV41VisionBlock(config, dtype, attention_factory) for _ in range(config.vision_n_layers)]
        )
        self.norm = AscendV41VisionRMSNorm(config.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        if n_h <= 0 or n_w <= 0 or patches.shape != (n_h * n_w, 3, self.patch_size, self.patch_size):
            raise ValueError("V4.1 patches must exactly cover a positive [height,width] grid")
        if patches.dtype != self.patch_embed.proj.weight.dtype or patches.device != self.patch_embed.proj.weight.device:
            raise ValueError("V4.1 patches must have the tower's weight dtype and device")
        x = self.patch_embed(patches)
        cos, sin = _v41_vision_cos_sin(n_h, n_w, self.head_dim, self.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class AscendV41VisionAligner(nn.Module):
    """Bottom/right pad, channel-major unfold, GELU, then LLM-width rows."""

    def __init__(self, config, *, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("V4.1 aligner uses BF16 weights or an FP32 reference")
        if config.vision_downsample_ratio <= 0 or config.vision_dim <= 0 or config.hidden_size <= 0:
            raise ValueError("Invalid V4.1 aligner dimensions")
        self.downsample_ratio = config.vision_downsample_ratio
        self.vision_dim = config.vision_dim
        self.w1 = nn.Linear(self.vision_dim * self.downsample_ratio**2, config.hidden_size, dtype=dtype)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        if n_h <= 0 or n_w <= 0 or x.shape != (n_h * n_w, self.vision_dim):
            raise ValueError("V4.1 aligner features must exactly cover the patch grid")
        if x.dtype != self.w1.weight.dtype or x.device != self.w1.weight.device:
            raise ValueError("V4.1 aligner features must have the weight dtype and device")
        ratio = self.downsample_ratio
        value = x.view(n_h, n_w, self.vision_dim).permute(2, 0, 1)
        value = F.pad(value, (0, -n_w % ratio, 0, -n_h % ratio))
        value = F.unfold(value.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(value), approximate="none"))


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV41VLMultiModalProcessor,
    info=DeepseekV41VLProcessingInfo,
    dummy_inputs=DeepseekV41VLDummyInputsBuilder,
)
class AscendDeepseekV41ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsEagle3):
    """V4.1 composition: eager replicated encoder and causal text LM.

    The processor protocol is the four fields provided by the local
    ``patch_deepseek_v41_mm`` module. Initial admission supports one image per
    request, complete image prefill, no encoder graph,
    no speculation and no PP. A batch may contain images from several requests.
    """

    requires_raw_input_tokens = True
    supports_encoder_tp_data = False

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality != "image":
            raise ValueError(f"Unsupported V4.1 modality: {modality!r}")
        return "<｜deepseek_image｜>"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.get_multimodal_config()
        self.image_limit = self.multimodal_config.get_limit_per_prompt("image")
        if "image" not in self.multimodal_config.limit_per_prompt and not self.multimodal_config.language_model_only:
            self.image_limit = 1
        if self.image_limit not in (0, 1):
            raise ValueError("Initial V4.1 multimodal support requires image limit 0 or 1 per request")
        if self.image_limit:
            compilation = vllm_config.compilation_config
            if compilation.cudagraph_mm_encoder or compilation.compile_mm_encoder:
                raise ValueError("V4.1 images require eager encoder execution; disable encoder compilation and graphs")
            if not vllm_config.scheduler_config.disable_chunked_mm_input:
                raise ValueError("V4.1 image admission currently requires disable_chunked_mm_input=True")
        if self.multimodal_config.enable_mm_embeds:
            raise ValueError(
                "V4.1 requires processor-owned image spans; external multimodal embeddings are not enabled"
            )
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.speculative_config is not None
            or self.multimodal_config.mm_encoder_tp_mode == "data"
            or self.multimodal_config.mm_encoder_only
        ):
            raise ValueError("V4.1 wrapper requires PP1, no speculation and replicated eager vision")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("V4.1 wrapper requires BF16 dense and vision weights")
        if self.image_limit and self.config.vision_n_layers <= 0:
            raise ValueError("Images require a V4.1 vision tower")
        self.vision = self.aligner = None
        self._tower_model_names = []
        for name in ("image_start", "image_end", "image_newline"):
            self.register_parameter(name, None)
        if self.image_limit:
            # Worker-only dependency: constructing an NPU encoder must not
            # load the encoder graph manager during text-model import.
            from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

            with self._mark_tower_model(vllm_config, {"image"}):
                self.vision = AscendV41VisionTower(
                    self.config,
                    attention_factory=lambda heads, dim: AscendMMEncoderAttention(heads, dim),
                )
                self.aligner = AscendV41VisionAligner(self.config)
                for name in ("image_start", "image_end", "image_newline"):
                    setattr(self, name, nn.Parameter(torch.empty(self.config.hidden_size, dtype=torch.bfloat16)))
        with self._mark_language_model(vllm_config):
            self.language_model = AscendDeepseekV41ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

    def get_language_model(self):
        return self.language_model

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_default_aux_hidden_state_layers()

    def create_engram_runtime(self):
        return self.language_model.create_engram_runtime()

    def engram_prompt_mask(self, request) -> torch.Tensor:
        spans = V41EngramImageSpans.from_request(request, image_token_id=self.config.image_token_id)
        if len(spans.image_spans) > self.image_limit:
            raise ValueError("V4.1 request exceeds its admitted image limit")
        return spans.prompt_keep_mask()

    def _validate_image_inputs(self, patches, vit_grid, llm_grid, types):
        if not self.image_limit:
            raise ValueError("The V4.1 image tower is disabled by image limit 0")
        for name, grid in (("vit_grid", vit_grid), ("llm_grid", llm_grid)):
            if (
                not isinstance(grid, torch.Tensor)
                or grid.device.type != "cpu"
                or grid.dtype not in (torch.int32, torch.int64)
                or grid.ndim != 2
                or grid.shape[1] != 2
                or grid.shape[0] == 0
            ):
                raise ValueError(f"V4.1 {name} must be a nonempty CPU integer [images,2] tensor")
        if vit_grid.shape != llm_grid.shape:
            raise ValueError("V4.1 patch and aligner grids must describe the same images")
        if (
            not isinstance(types, torch.Tensor)
            or types.device.type != "cpu"
            or types.dtype not in (torch.int32, torch.int64)
            or types.ndim != 1
        ):
            raise ValueError("V4.1 roles must be a one-dimensional CPU integer tensor")
        grids = []
        patch_count = span_offset = 0
        ratio = self.config.vision_downsample_ratio
        for (height, width), (rows, columns) in zip(vit_grid.tolist(), llm_grid.tolist(), strict=True):
            if min(height, width) <= 0 or (rows, columns) != (
                (height + ratio - 1) // ratio,
                (width + ratio - 1) // ratio,
            ):
                raise ValueError("V4.1 aligner grid must be the ceil-downsampled positive patch grid")
            expected = [IMAGE_START] + ([IMAGE] * columns + [IMAGE_NEW_LINE]) * rows + [IMAGE_END]
            if len(expected) > self.config.vision_max_n_token:
                raise ValueError("V4.1 image span exceeds the configured token budget")
            actual = types[span_offset : span_offset + len(expected)]
            if actual.tolist() != expected:
                raise ValueError("V4.1 image roles must exactly cover START, IMAGE/NEW_LINE rows and END")
            grids.append((height, width, rows, columns))
            patch_count += height * width
            span_offset += len(expected)
        size = self.config.vision_patch_size
        if not isinstance(patches, torch.Tensor) or patches.shape != (patch_count, 3, size, size):
            raise ValueError("V4.1 patches must exactly cover every supplied image grid")
        if span_offset != types.numel():
            raise ValueError("V4.1 roles contain positions outside the supplied image spans")
        return grids

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if not kwargs:
            return ()
        if set(kwargs) != {"patches", "vit_grid", "llm_grid", "types"}:
            raise ValueError("V4.1 images require patches, vit_grid, llm_grid and types")
        patches = kwargs["patches"]
        grids = self._validate_image_inputs(patches, kwargs["vit_grid"], kwargs["llm_grid"], kwargs["types"])
        device = self.image_start.device
        if device.type == "npu" and torch.npu.is_current_stream_capturing():
            raise RuntimeError("V4.1 image encoding and merging must remain outside decoder graph capture")
        patches = patches.to(device=device, dtype=self.aligner.w1.weight.dtype)
        spans = []
        offset = 0
        for height, width, rows, columns in grids:
            count = height * width
            features = self.vision(patches[offset : offset + count], height, width)
            aligned = self.aligner(features, height, width)
            span = aligned.new_empty((rows * (columns + 1) + 2, self.config.hidden_size))
            span[0], span[-1] = self.image_start, self.image_end
            middle = span[1:-1].view(rows, columns + 1, -1)
            middle[:, :columns] = aligned.view(rows, columns, -1)
            middle[:, columns] = self.image_newline
            spans.append(span)
            offset += count
        return tuple(spans)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return embeddings
        if is_multimodal is None or is_multimodal.dtype != torch.bool or is_multimodal.shape != input_ids.shape:
            raise ValueError("V4.1 embedding merge requires an explicit boolean mask for the complete image spans")
        if input_ids.device.type == "npu" and torch.npu.is_current_stream_capturing():
            raise RuntimeError("V4.1 image embedding merge must remain outside decoder graph capture")
        # This validation runs only in eager image prefill. The CPU prompt
        # provider separately validates ranges/roles before Engram seeding.
        if not torch.all(input_ids[is_multimodal] == self.config.image_token_id):
            raise ValueError("Every V4.1 image embedding position must preserve its raw image token ID")
        return _merge_multimodal_embeddings(embeddings, multimodal_embeddings, is_multimodal)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("V4.1 requires raw token IDs even when merged inputs_embeds are supplied")
        image_token_mask = kwargs.get("image_token_mask")
        if (
            not isinstance(image_token_mask, torch.Tensor)
            or image_token_mask.dtype != torch.bool
            or image_token_mask.shape != input_ids.shape
            or image_token_mask.device != input_ids.device
        ):
            raise ValueError("V4.1 MM forward requires an explicit bool image_token_mask matching raw token IDs")
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = {
            name: parameter for name, parameter in self.named_parameters() if not name.startswith("language_model.")
        }
        loaded_mm = set()

        def language_weights():
            for name, value in weights:
                child_name = name.removeprefix("language_model.")
                native = child_name.removeprefix("model.")
                if native.startswith(("vision.", "aligner.")) or native in {
                    "image_start",
                    "image_end",
                    "image_newline",
                }:
                    if not self.image_limit:
                        continue
                    if native not in parameters or native in loaded_mm:
                        raise ValueError(f"Unexpected or duplicate V4.1 vision weight: {native}")
                    parameter = parameters[native]
                    if value.dtype != torch.bfloat16 or value.shape != parameter.shape:
                        raise ValueError(f"V4.1 vision weight {native} must have BF16 checkpoint dtype and exact shape")
                    default_weight_loader(parameter, value)
                    loaded_mm.add(native)
                else:
                    yield child_name, value

        loaded_lm = self.language_model.load_weights(language_weights())
        missing = parameters.keys() - loaded_mm
        if missing:
            raise ValueError(f"Missing V4.1 vision weights: {sorted(missing)}")
        return loaded_mm | {f"language_model.{name}" for name in loaded_lm}
