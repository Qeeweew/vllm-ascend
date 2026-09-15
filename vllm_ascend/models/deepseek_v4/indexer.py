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
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.attention.dsa_attn_kv_plan import is_a5_bf16_kv_enabled
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata, Compressor
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.indexer_v41_candidate import CandidateIndexerB1
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import (
    npu_stream_switch,
    vllm_version_is,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata


def hadamard_linear(x: torch.Tensor, hadamard: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    dim_padded = 2 ** math.ceil(math.log2(dim))
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    return F.linear(x, hadamard), x_shape, dim


def hadamard_scale(out: torch.Tensor, x_shape: tuple[int, ...], dim: int, scale: float = 1.0) -> torch.Tensor:
    """Scale and reshape the output of hadamard_linear."""
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    out, x_shape, dim = hadamard_linear(x, hadamard)
    return (out * dim**-0.5)[..., :dim].reshape(*x_shape)


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8DynamicLinearMethod)


def _is_mxfp8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8MXFP8DynamicLinearMethod``."""
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    if isinstance(quant_method, AscendW8A8MXFP8DynamicLinearMethod):
        return True
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8MXFP8DynamicLinearMethod)


class AscendDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__(head_dim, dtype, prefix, cache_config, compress_ratio)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
            self.dtype = torch.float8_e4m3fn
            if not is_a5_bf16_kv_enabled(vllm_config):
                vllm_config.cache_config.cache_dtype = "float8_e4m3fn"

        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        storage_block_size = DSV4_BLOCK_SIZES[vllm_config.cache_config.block_size][0][0]
        # vLLM #51718 replaced MLAAttentionSpec.compress_ratio with
        # AttentionSpec.tokens_per_state on main.
        ratio_kwargs = (
            {"compress_ratio": self.compress_ratio}
            if vllm_version_is("0.28.0")
            else {"tokens_per_state": self.compress_ratio}
        )
        return AscendMLAAttentionSpec(
            block_size=storage_block_size * self.compress_ratio,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            model_version="deepseek_v4",
            cache_dtype_str=self.cache_config.cache_dtype,
            scale_dim=1 if self.head_dim == 128 else 0,
            scale_dtype=torch.float
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else torch.float16,
            **ratio_kwargs,
        )

    def forward(self): ...

    def get_attn_backend(self):
        # Keep these imports lazy to avoid a model-inspection circular import.
        if self.compress_ratio == 4:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC4Backend

            return AscendDSAC4Backend
        if self.compress_ratio == 128:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC128Backend

            return AscendDSAC128Backend
        raise ValueError(f"Unsupported DeepSeek V4 indexer compression ratio: {self.compress_ratio}")


@dataclass(frozen=True)
class AscendIndexerMetadata:
    compressor: AscendCompressorMetadata


@dataclass(frozen=True)
class IndexerOverlapPlan:
    """Main-attention compressor work scheduled around Indexer selection."""

    compute_attention_compressed_kv: typing.Callable[[], tuple[torch.Tensor, torch.Tensor]]
    scatter_attention_compressed_kv: typing.Callable[[torch.Tensor, torch.Tensor], None]
    aux_stream: torch.npu.Stream | None = None


class AscendIndexerOps:
    def __init__(self, index_topk: int) -> None:
        from vllm_ascend.device.device_op import DeviceOperator

        self.device_operator = DeviceOperator
        self.index_topk = index_topk

    def unpack_dsa_indexer_kv_cache(self, kv_cache: tuple[torch.Tensor, ...]):
        return self.device_operator.unpack_dsa_indexer_kv_cache(kv_cache)

    def quantize_query(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.device_operator.indexer_quantize_query(query)

    def quantize_key_and_update_cache(
        self,
        key: torch.Tensor,
        key_cache: torch.Tensor,
        full_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
    ):
        return self.device_operator.indexer_quant_scatter_part1(
            key,
            key_cache,
            full_cache,
            slot_mapping,
        )

    def update_scale_cache(
        self,
        key_scale: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self.device_operator.dsa_indexer_scatter_scale_part3(
            key_scale,
            scale_cache,
            slot_mapping,
        )

    def select_topk(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        metadata: typing.Any,
    ) -> torch.Tensor:
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(metadata.qli_metadata))
        topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query=query,
            key=key_cache,
            weights=self.device_operator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=self.device_operator.prepare_dsa_indexer_query_scale(query_scale),
            key_dequant_scale=self.device_operator.prepare_dsa_indexer_key_scale(scale_cache),
            topk=self.index_topk,
            quant_mode=self.device_operator.get_dsa_indexer_quant_mode(),
            cu_seqlens_q=metadata.qli_cu_seqlens_q,
            seqused_k=metadata.qli_seqused_k,
            cmp_residual_k=metadata.qli_cmp_residual_k,
            block_table=metadata.block_table,
            metadata=metadata.qli_metadata,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            return_value=0,
        )
        return topk_idxs

    def quantize_update_cache_and_select_topk(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        weights: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        full_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        metadata: typing.Any,
    ) -> torch.Tensor:
        query, query_scale, _, _ = self.device_operator.indexer_quant_scatter(
            query,
            key,
            key_cache,
            scale_cache,
            full_cache,
            slot_mapping,
        )
        return self.select_topk(
            query,
            weights,
            query_scale,
            key_cache,
            scale_cache,
            metadata,
        )


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int,
        skip_topk: bool,
        use_index_cache: bool,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.skip_topk = skip_topk
        self.use_index_cache = use_index_cache

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )

        self.cv_wq_b = CVLinearWrapper(self.wq_b)
        self.topk_indices_buffer = topk_indices_buffer
        if self.skip_topk and self.topk_indices_buffer is None:
            raise ValueError("skip_topk requires topk_indices_buffer")
        self.ops = AscendIndexerOps(index_topk=self.index_topk)
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        k_dtype = (
            torch.float8_e4m3fn
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else torch.int8
        )

        if self.compress_ratio == 4:
            # TODO(cmq): change the dtype of cache
            self.k_cache = AscendDeepseekV4IndexerCache(
                head_dim=self.head_dim,
                dtype=k_dtype,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                head_dim=self.head_dim,
                rotate=True,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
            )  # Compressor(4, 128)

    @staticmethod
    def _get_indexer_cache_metadata(
        metadata: AscendIndexerMetadata,
    ) -> tuple[typing.Any, torch.Tensor]:
        cache_metadata = metadata.compressor.cache
        cache_req_metadata = cache_metadata.req_metadata
        hadamard = cache_metadata.hadamard
        assert cache_req_metadata is not None
        assert hadamard is not None
        return cache_req_metadata, hadamard

    def update_cache(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
    ) -> None:
        """Update Indexer caches without projecting queries or selecting TopK."""
        if hidden_states.shape[0] == 0:
            return

        state_cache, key_cache, scale_cache, full_cache = self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        _, hadamard = self._get_indexer_cache_metadata(metadata)
        compressor = self.compressor
        assert compressor is not None
        key, slot_mapping = compressor(
            hidden_states=hidden_states,
            state_cache=state_cache,
            metadata=metadata.compressor,
        )
        if key.shape[0] == 0:
            return
        if compressor.rotate:
            key = rotate_activation(key, hadamard)
        _, key_scale = self.ops.quantize_key_and_update_cache(
            key,
            key_cache,
            full_cache,
            slot_mapping,
        )
        if key_scale is not None:
            self.ops.update_scale_cache(
                key_scale,
                scale_cache,
                slot_mapping,
            )

    def _get_cached_topk_indices(self, num_tokens: int, offset: int = 0) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("topk_indices_buffer is required to read cached TopK indices")
        topk_indices = self.topk_indices_buffer[offset : offset + num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_cached_topk_indices(self, topk_indices: torch.Tensor, offset: int = 0) -> None:
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        topk_indices_buffer = self.topk_indices_buffer[offset : offset + num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            if topk_indices_to_cache.shape[1] != 1:
                raise ValueError("TopK indices must have a singleton head dimension")
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        topk_indices_buffer.copy_(topk_indices_to_cache)

    def forward(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        overlap_plan: IndexerOverlapPlan,
        *,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        cache_metadata, _ = self._get_indexer_cache_metadata(metadata)
        cos = cache_metadata.cos[layer_name][:num_tokens]
        sin = cache_metadata.sin[layer_name][:num_tokens]
        aux_stream = overlap_plan.aux_stream
        if self.skip_topk:
            topk_indices = self._get_cached_topk_indices(num_tokens)
        elif aux_stream is not None:
            indexer_q = self._cv_compute_query_and_update_cache_multistream(
                hidden_states,
                qr,
                kv_cache,
                metadata,
                cos,
                sin,
                aux_stream,
                qr_pertoken_scale,
            )
            compressed_kv, compress_slot_mapping = overlap_plan.compute_attention_compressed_kv()
            topk_indices = self._select_topk_multistream(
                hidden_states,
                indexer_q,
                kv_cache,
                metadata,
                aux_stream,
                lambda: overlap_plan.scatter_attention_compressed_kv(
                    compressed_kv,
                    compress_slot_mapping,
                ),
            )
        else:
            topk_indices = self._select_topk_serial(
                hidden_states,
                qr,
                kv_cache,
                metadata,
                cos,
                sin,
                qr_pertoken_scale,
                write_cache=write_cache,
            )

        if write_cache and (self.skip_topk or aux_stream is None):
            compressed_kv, compress_slot_mapping = overlap_plan.compute_attention_compressed_kv()
            overlap_plan.scatter_attention_compressed_kv(compressed_kv, compress_slot_mapping)

        if self.use_index_cache:
            self._update_cached_topk_indices(topk_indices)
        return topk_indices

    def _cv_compute_query_and_update_cache_multistream(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        aux_stream: torch.npu.Stream,
        qr_pertoken_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Indexer query and update its cache.

        The internal multistream strategy keeps the original four-part layout:
        - Part0: Main pre-compute qr_quant[V] + compressor[C/mixed] + kv_hadamard[V]
        - Part1: Main matmul[C] ∥ Aux kv_quant[V] + scatter_k_cache[AIV]
        - Part2: Main rope[V] (serial)
        - Part3: Main q_hadamard[C] ∥ Aux scatter_scale_cache[AIV]
        """
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        _, hadamard = self._get_indexer_cache_metadata(metadata)
        main_stream = torch.npu.current_stream()
        compressor = self.compressor
        assert compressor is not None

        # ===== Part0: Pre-compute on main =====
        # Reuse the prolog's pre-quantized qr when this layer's scheme
        # matches (W8A8 fused quant / MXFP8 split-quant).
        if qr_pertoken_scale is not None and (_is_w8a8_dynamic(self.wq_b) or _is_mxfp8_dynamic(self.wq_b)):
            qr_quant_ready = qr
            qr_scale_ready = qr_pertoken_scale
        else:
            qr_quant_ready, qr_scale_ready = self.cv_wq_b.quantize(qr)

        kv, slot_mapping_indexer = compressor(
            hidden_states=hidden_states,
            state_cache=indexer_state_cache,
            metadata=metadata.compressor,
        )
        if kv.numel() == 0:
            kv = None
        elif compressor.rotate:
            kv = rotate_activation(kv, hadamard)

        # ===== Part1: matmul[C] ∥ kv_quant[V] + scatter_k_cache[AIV] =====
        # Record event before main stream operations for aux_stream to wait
        e_kv_ready = main_stream.record_event()

        # Aux: kv_quant + scatter_k_cache (parallel with main matmul + rope)
        if kv is not None:
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_kv_ready)
                kv, kv_scale = self.ops.quantize_key_and_update_cache(
                    kv,
                    indexer_k_cache,
                    indexer_full_cache,
                    slot_mapping_indexer,
                )

        # Main: matmul q from qr (directly submit, V/C different engines dispatch naturally)
        if _is_w8a8_dynamic(self.wq_b) and qr_pertoken_scale is not None:
            q = torch_npu.npu_quant_matmul(
                qr_quant_ready,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_scale_ready,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            q = self.cv_wq_b.matmul(qr_quant_ready, qr_scale_ready)  # qr_matmul

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        q = q.view(-1, self.n_heads, self.head_dim)

        # ===== Part2: rope[V] (main only) =====
        torch.ops._C_ascend.inplace_partial_rotary_mul(  # rope
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        )

        # Wait for aux_stream kv_scatter to complete before proceeding
        if kv is not None:
            main_stream.wait_stream(aux_stream)

        e_rope_done = main_stream.record_event()

        # ===== Part3: q_hadamard[C] ∥ scatter_scale_cache[AIV] =====
        # Note: On A5, indexer_compress_epilog_v2 in Part1 handles both k_cache
        # and scale_cache in one fused operation, so Part3 is skipped
        # (kv_scale is None on A5 from indexer_quant_scatter_part1).
        if kv is not None and kv_scale is not None:
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_rope_done)
                self.ops.update_scale_cache(
                    kv_scale,
                    indexer_scale_cache,
                    slot_mapping_indexer,
                )

        # Main: q_hadamard[Part1 - linear] (directly submit, C/AIV different engines dispatch naturally)
        # Part1: F.linear - parallel with aux_stream kv_scatter
        hidden_size = q.size(-1)
        q_linear, q_shape, q_dim = hadamard_linear(q, hadamard)

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        # Main: q_hadamard[Part2 - scale] (after aux_stream completes)
        # Part2: scale * reshape - dot multiplication
        q = hadamard_scale(q_linear, q_shape, q_dim, scale=hidden_size**-0.5)

        return q

    def _select_topk_multistream(
        self,
        hidden_states: torch.Tensor,
        indexer_q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        aux_stream: torch.npu.Stream,
        scatter_attention_compressed_kv: typing.Callable[[], None],
    ) -> torch.Tensor:
        """Overlap Indexer selection inputs with caller-provided main-stream work."""
        main_stream = torch.npu.current_stream()
        weights_proj_start = main_stream.record_event()
        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(weights_proj_start)
            weights_proj_output = self.weights_proj(hidden_states)
            weights_proj_done = torch.npu.current_stream().record_event()

        q_quant, q_scale = self.ops.quantize_query(indexer_q)
        # Enqueue only independent Vector/AIV work on the current main stream;
        # do not switch streams or launch Cube work that would contend with the
        # auxiliary weights projection.
        scatter_attention_compressed_kv()
        main_stream.wait_event(weights_proj_done)

        (_, indexer_k_cache, indexer_scale_cache, _) = self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        cache_metadata, _ = self._get_indexer_cache_metadata(metadata)
        weights = weights_proj_output * (self.softmax_scale * self.n_heads**-0.5)
        return self.ops.select_topk(
            q_quant,
            weights,
            q_scale,
            indexer_k_cache,
            indexer_scale_cache,
            cache_metadata,
        )

    def _indexer_qkv_prepare(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ):
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        cache_metadata, hadamard = self._get_indexer_cache_metadata(metadata)
        compressor = self.compressor
        assert compressor is not None

        if (
            _is_w8a8_dynamic(self.wq_b)
            and qr_pertoken_scale is not None
            and not get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
        ):
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)  # [T, N, D]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        )

        q = rotate_activation(q, hadamard)
        kv = None
        indexer_slot_mapping = None
        if write_cache:
            kv, indexer_slot_mapping = compressor(
                hidden_states=x,
                state_cache=indexer_state_cache,
                metadata=metadata.compressor,
            )
            if kv.numel() == 0:
                kv = None
            elif compressor.rotate:
                kv = rotate_activation(kv, hadamard)

        return (
            q,
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            cache_metadata,
            indexer_slot_mapping,
        )

    def _select_topk_serial(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ):
        q, kv, ik, isc, ifc, cache_metadata, indexer_slot_mapping = self._indexer_qkv_prepare(
            x,
            qr,
            kv_cache,
            metadata,
            cos,
            sin,
            qr_pertoken_scale,
            write_cache=write_cache,
        )

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

        if write_cache:
            return self.ops.quantize_update_cache_and_select_topk(
                q,
                kv,
                weights,
                ik,
                isc,
                ifc,
                indexer_slot_mapping,
                cache_metadata,
            )

        q, q_scale = self.ops.quantize_query(q)
        return self.ops.select_topk(
            q,
            weights,
            q_scale,
            ik,
            isc,
            cache_metadata,
        )


@dataclass
class AscendIndexerV41Metadata:
    """Device-resident metadata for the independent V4.1 CSA indexer.

    ``seqused_k`` counts compressed positions. CR2 also needs the remainder
    of the original context length; CR1 must pass no remainder tensor. All
    buffers, including ``qli_metadata``, must be refreshed in-place before a
    graph replay when requests/lengths change. Candidate IDs are relative
    blocks of eight compressed positions and are shared only within one step.
    """

    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    block_table: torch.Tensor
    qli_metadata: torch.Tensor
    cmp_residual_k: torch.Tensor | None = None


class DeepseekV41IndexerProjections(nn.Module):
    """Replicated selector projections; cache owners alone have wk/k_norm."""

    def __init__(self, config, owns_k: bool, prefix: str):
        super().__init__()
        if (config.index_n_heads, config.index_head_dim) != (32, 128):
            raise ValueError("V4.1 native indexer requires 32 heads of width 128")
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            32 * 128,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            32,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.weights_proj",
        )
        if owns_k:
            self.wk = ReplicatedLinear(
                512,
                128,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                return_bias=False,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(128, config.rms_norm_eps, dtype=torch.bfloat16)
        self.owns_k = owns_k

    def project_query(self, hidden: torch.Tensor, qr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.wq_b(qr).view(-1, 32, 128)
        weights = (self.weights_proj(hidden).float() * (128**-0.5 * 32**-0.5)).to(torch.float16)
        return query, weights

    def project_key(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.owns_k:
            raise RuntimeError("Only a KV source projects index keys")
        return self.k_norm(self.wk(latent))


class AscendIndexerV41Ops:
    """910B selector for 32 replicated heads; projections/RoPE stay separate.

    INT8 is a new per-128-element activation quantization. It is neither a
    reinterpretation of MXFP4/FP8 bytes nor numerically identical to their
    group scaling. The native score pipeline also rounds QK/1024 and
    weights*query_scale to FP16 before FP32 head reduction. Model quality must
    therefore be evaluated separately from native-kernel correctness.
    """

    def __init__(
        self, compress_ratio: int, candidate_mode: str = "off", *, candidate_max_context: int | None = None
    ) -> None:
        if compress_ratio not in (1, 2):
            raise ValueError("V4.1 indexer compress_ratio must be 1 or 2")
        if candidate_mode not in ("off", "source", "consumer"):
            raise ValueError("candidate_mode must be off, source, or consumer")
        if compress_ratio != 1 and candidate_mode != "off":
            raise ValueError("V4.1 candidate source and consumers require CR1")
        if candidate_max_context is not None:
            if candidate_mode != "consumer":
                raise ValueError("candidate_max_context requires a CR1 candidate consumer")
            if not isinstance(candidate_max_context, int) or not 1 <= candidate_max_context <= 2**27:
                raise ValueError("candidate_max_context must be a static bound in [1, 2**27]")
        self.compress_ratio = compress_ratio
        self.candidate_mode = {"source": 1, "consumer": 2, "off": 3}[candidate_mode]
        self.candidate_max_context = candidate_max_context
        self._candidate_selector: CandidateIndexerB1 | None = None

    def prepare_candidate_workspace(self, device: torch.device | str) -> None:
        """Allocate the optional B1 workspace explicitly before graph capture.

        Runtime CR1 lengths must not exceed candidate_max_context. The caller
        owns that static bound; selection never reads device lengths on host.
        Unprepared selectors retain native dispatch, including during capture.
        """
        if self.candidate_max_context is None:
            return
        device = torch.device(device)
        if device.type == "npu" and device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        if self._candidate_selector is not None:
            if self._candidate_selector.key.device != device:
                raise ValueError("Candidate workspace cannot move devices after preparation")
            return
        if device.type == "npu" and torch.npu.is_current_stream_capturing():
            raise RuntimeError("Prepare candidate workspace before NPU graph capture")
        self._candidate_selector = CandidateIndexerB1(self.candidate_max_context, device)

    @staticmethod
    def quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize already rotated BF16/FP16 vectors, preserving head axes."""
        if value.shape[-1] != 128 or value.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("V4.1 indexer quantize expects BF16/FP16 vectors of width 128")
        quantized, scale = torch_npu.npu_dynamic_quant(value, dst_type=torch.int8)
        return quantized, scale.to(torch.float16)

    def build_metadata(
        self,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        *,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cmp_residual_k: torch.Tensor | None = None,
    ) -> AscendIndexerV41Metadata:
        """Build scheduling data; max_seqlen_k is in ORIGINAL token units."""
        if (self.compress_ratio == 1) != (cmp_residual_k is None):
            raise ValueError("CR1 forbids cmp_residual_k; CR2 requires it")
        schedule = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            topk=512,
            quant_mode=2,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=seqused_k.numel(),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
            device=str(seqused_k.device),
        )
        return AscendIndexerV41Metadata(cu_seqlens_q, seqused_k, block_table, schedule, cmp_residual_k)

    def select_topk(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        metadata: AscendIndexerV41Metadata,
        candidate_blocks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [T,1,512] positions and source-only [T,1,2048] blocks.

        Position IDs are increasing with -1 padding at the end. Source block
        IDs retain score order, matching the native consumer ABI. Inputs must
        retain their addresses/lifetimes throughout graph capture and replay.
        """
        tokens = query.shape[0]
        if query.shape != (tokens, 32, 128) or query.dtype != torch.int8:
            raise ValueError("V4.1 indexer query must be INT8 [T,32,128]")
        if key_cache.ndim != 4 or key_cache.shape[2:] != (1, 128) or key_cache.dtype != torch.int8:
            raise ValueError("V4.1 indexer key cache must be INT8 [blocks,block_size,1,128]")
        if weights.shape != (tokens, 32) or query_scale.shape != weights.shape:
            raise ValueError("V4.1 indexer weights and query scales must have shape [T,32]")
        if scale_cache.shape != key_cache.shape[:-1]:
            raise ValueError("V4.1 indexer key scales must have shape [blocks,block_size,1]")
        if any(t.dtype != torch.float16 for t in (weights, query_scale, scale_cache)):
            raise ValueError("V4.1 indexer weights and scales must be FP16")
        if (self.compress_ratio == 1) != (metadata.cmp_residual_k is None):
            raise ValueError("CR1 forbids cmp_residual_k; CR2 requires it")
        if self.candidate_mode == 2:
            if candidate_blocks is None or candidate_blocks.shape != (tokens, 1, 2048):
                raise ValueError("V4.1 consumer requires source candidate blocks [T,1,2048]")
            if candidate_blocks.dtype != torch.int32:
                raise ValueError("V4.1 candidate block IDs must be INT32")
        elif candidate_blocks is not None:
            raise ValueError("Only a candidate consumer accepts candidate_blocks")
        if not tokens:
            indices = torch.empty((0, 1, 512), dtype=torch.int32, device=query.device)
            shape = (0, 1, 2048) if self.candidate_mode == 1 else (0,)
            return indices, torch.empty(shape, dtype=torch.int32, device=query.device)
        candidate = self._candidate_selector
        if (
            candidate is not None
            and tokens == 1
            and metadata.seqused_k.shape == (1,)
            and metadata.cu_seqlens_q.shape == (2,)
            and metadata.block_table.shape[0] == 1
            and key_cache.shape[1] % 8 == 0
            and metadata.block_table.shape[1] * key_cache.shape[1] >= candidate.max_context
            and candidate.key.device == query.device
        ):
            return candidate(query, key_cache, weights, query_scale, scale_cache, metadata, candidate_blocks)
        indices, _, blocks = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            query=query,
            key=key_cache,
            weights=weights,
            query_dequant_scale=query_scale,
            key_dequant_scale=scale_cache,
            topk=512,
            quant_mode=2,
            candidate_topk_index=candidate_blocks,
            cu_seqlens_q=metadata.cu_seqlens_q,
            seqused_k=metadata.seqused_k,
            cmp_residual_k=metadata.cmp_residual_k,
            block_table=metadata.block_table,
            metadata=metadata.qli_metadata,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
            candidate_mode=self.candidate_mode,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
        )
        # CANN INT32 sort has a costly per-row path on 910B. Relative position
        # IDs below 2**24 are exact FP32 integers, so sorting them in FP32 is
        # lossless. A shape-only cache bound selects the path without a host
        # synchronization; larger address spaces retain the INT32 path.
        sentinel = 2147483647
        if metadata.block_table.shape[1] * key_cache.shape[1] <= 2**24:
            indices = indices.float()
            sentinel = 2**24
        indices = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
        valid_rows = torch.arange(tokens, device=query.device) < metadata.cu_seqlens_q[-1]
        indices = torch.where((indices != sentinel) & valid_rows[:, None, None], indices, -1).to(torch.int32)
        if self.candidate_mode == 1:
            blocks = torch.where(valid_rows[:, None, None], blocks, -1)
        return indices, blocks
