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
import typing
from dataclasses import dataclass

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.compressor import CompressorStateCache
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder, MultipleOf
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.compressor_v41 import compressor_v41
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata


class AscendCompressorStateCache(CompressorStateCache):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        block_size: int,
        prefix: str,
    ):
        super().__init__(state_dim, dtype, compress_ratio, prefix)
        self.compress_ratio = compress_ratio
        self.block_size = block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        from vllm_ascend.models.layer.attention.layer import dsv4_block_sizes

        pads = dsv4_block_sizes(vllm_config)[vllm_config.cache_config.block_size][1]
        page_size_padded = pads[0] if self.state_dim == 2 * 256 and self.compress_ratio == 4 else pads[1]

        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
            page_size_padded=page_size_padded,
        )

    def forward(self): ...

    def get_attn_backend(self):
        # Keep these imports lazy to avoid a model-inspection circular import.
        if self.compress_ratio == 4:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC4StateBackend

            return AscendDSAC4StateBackend
        if self.compress_ratio == 128:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC128StateBackend

            return AscendDSAC128StateBackend
        raise ValueError(f"Unsupported DeepSeek V4 state-cache compression ratio: {self.compress_ratio}")


@dataclass(frozen=True)
class AscendCompressorMetadata:
    """Request metadata for the compressed KV and compressor state caches."""

    cache: typing.Any
    state: typing.Any


@dataclass(frozen=True)
class CompressorV41Metadata:
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req_indices: torch.Tensor


class CompressorV41MetadataBuilder(AttentionMetadataBuilder):
    """Prepare stable ring metadata outside graph replay, using device boundaries.

    Adaptive verification can change query boundaries on device. Searching the
    device query_start_loc avoids using stale CPU request lengths for drafts.
    """

    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.kv_cache_spec, CircularBufferSpec):
            raise TypeError("V4.1 compressor requires a circular state cache")
        self.capacity = self.kv_cache_spec.block_size
        maximum = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_indices = torch.arange(maximum, dtype=torch.int32, device=self.device)
        self.request_indices = torch.empty(maximum, dtype=torch.int32, device=self.device)
        self.slots = torch.empty(maximum, dtype=torch.int64, device=self.device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = common_attn_metadata
        positions = metadata.positions
        if positions is None:
            raise ValueError("V4.1 compressor metadata requires token positions")
        tokens = metadata.slot_mapping.numel()
        if tokens > self.slots.numel():
            raise ValueError("Compressor graph bucket exceeds metadata capacity")
        requests = self.request_indices[:tokens]
        indices = self.token_indices[:tokens]
        slots = self.slots[:tokens]
        table = metadata.block_table_tensor
        if table.shape[0] == 0:
            requests.zero_()
            slots.fill_(-1)
        else:
            torch.searchsorted(metadata.query_start_loc[1:], indices, right=True, out_int32=True, out=requests)
            requests.clamp_(max=table.shape[0] - 1)
            blocks = table[:, 0].index_select(0, requests.long()).long()
            slots.copy_(blocks * self.capacity + positions[:tokens] % self.capacity)
            slots.masked_fill_((indices >= metadata.query_start_loc[-1]) | (positions[:tokens] < 0) | (blocks < 0), -1)
        return CompressorV41Metadata(slots, metadata.query_start_loc, requests)


class CompressorV41Impl:
    @staticmethod
    def update_graph_params(*args, **kwargs):
        # Metadata build refreshes fixed-address ring indices before replay;
        # this vector operator has no mutable graph task-group parameters.
        pass


class CompressorV41Backend(AttentionBackend):
    @staticmethod
    def get_name():
        return "ASCEND_COMPRESSOR_V41"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls):
        return [1024]

    @staticmethod
    def get_builder_cls():
        return CompressorV41MetadataBuilder

    @staticmethod
    def get_impl_cls():
        return CompressorV41Impl


class CompressorV41StateCache(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str):
        super().__init__()
        config = get_current_vllm_config()
        self.prefix = prefix
        self.block_size = max(8, 1 << (config.num_speculative_tokens + 1).bit_length())
        self.kv_cache = torch.tensor([])
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate compressor state cache: {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.dtype != torch.float32 or kv_cache.shape[1:] != (1, self.block_size, 1024):
            raise ValueError("Compressor state must be FP32 [blocks,1,capacity,1024]")
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return CircularBufferSpec(
            block_size=self.block_size, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32
        )

    def get_attn_backend(self):
        return CompressorV41Backend

    def forward(self):
        raise RuntimeError("Compressor state cache is storage, not a callable attention layer")


class CompressorV41(nn.Module):
    """V4.1 projection and pure vector compression, exposed as separate calls.

    A source layer calls project() once, then compresses into the pre-RoPE
    latent shared by main attention and the indexer. Cache insertion and RoPE
    are the caller's responsibility. CR2 requests FP32 GEMM output from BF16
    operands; casting a BF16 result afterwards would already have lost the
    required precision. Checkpoint projection weights are BF16 in both ratios.
    """

    def __init__(self, hidden_size: int, compress_ratio: int, eps: float, prefix: str):
        super().__init__()
        if compress_ratio not in (1, 2):
            raise ValueError("V4.1 compressor ratio must be 1 or 2")
        self.compress_ratio = compress_ratio
        self.eps = eps
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [512] * compress_ratio,
            bias=False,
            return_bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.fused_wkv_wgate.skip_weight_nz_conversion = True
        self.norm = RMSNorm(512, eps, dtype=torch.bfloat16)
        self.state_cache = CompressorV41StateCache(f"{prefix}.state_cache") if compress_ratio == 2 else None
        self.register_buffer("empty_state", torch.empty(0, dtype=torch.float32), persistent=False)

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("V4.1 compressor projection requires BF16 hidden states")
        weight = self.fused_wkv_wgate.weight
        if self.compress_ratio == 2:
            return torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
        return torch.mm(hidden_states, weight.t())

    def forward(
        self, kv_score: torch.Tensor, positions: torch.Tensor, metadata: CompressorV41Metadata, latent_out: torch.Tensor
    ) -> torch.Tensor:
        state = self.empty_state if self.state_cache is None else self.state_cache.kv_cache
        return compressor_v41(
            kv_score,
            positions,
            metadata.slot_mapping,
            metadata.query_start_loc,
            metadata.token_to_req_indices,
            self.norm.weight,
            state,
            latent_out,
            self.compress_ratio,
            self.eps,
        )


class Compressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
        *,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        self.vllm_config = vllm_config
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = config.rms_norm_eps
        self.coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32))
        self.wkv = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.wgate = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else quant_config,
            prefix=f"{prefix}.wgate",
            return_bias=False,
        )

        # The custom compressor op consumes ND weights directly.
        self.wkv.skip_weight_nz_conversion = True
        self.wgate.skip_weight_nz_conversion = True

        # The DSV4 compressor kernel only accepts FP32 norm_weight.
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=torch.float32)

        state_dtype = torch.float32
        # TODO(zyj): change following codes if block_size is configurable & refactor the magic numbers
        if compress_ratio == 4:
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][2],
            )
        elif compress_ratio == 128:
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][3],
            )
        else:
            raise ValueError(
                f"Only support compress_ratio in [4, 128]. Got unsupported compress_ratio: {compress_ratio}"
            )

    def _compute_metadata(
        self,
        metadata: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Imported lazily to avoid a circular import at module load time.
        from vllm_ascend.attention.dsa_v1 import get_or_compute_compressor_metadata

        precomputed = getattr(metadata, "compressor_metadata", None)
        if precomputed is not None:
            group_id = metadata.compressor_metadata_group_id
            assert group_id is not None
            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, group_id)
            return precomputed

        return get_or_compute_compressor_metadata(metadata, self.compress_ratio, self.vllm_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_cache: torch.Tensor,
        metadata: AscendCompressorMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressor_metadata = metadata.cache.req_metadata
        state_metadata = metadata.state.req_metadata
        assert compressor_metadata is not None
        assert state_metadata is not None
        compress_cos, compress_sin, slot_mapping = self._compute_metadata(compressor_metadata)
        compressed_kv = torch.ops._C_ascend.compressor(
            hidden_states,
            self.wkv.weight,
            self.wgate.weight,
            state_cache.squeeze(-2),
            self.ape,
            self.norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table=state_metadata.block_table,
            cu_seqlens=compressor_metadata.query_start_loc,
            seqused=None,
            start_pos=compressor_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=2 if self.overlap else 1,
            norm_eps=self.norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        return compressed_kv, slot_mapping
