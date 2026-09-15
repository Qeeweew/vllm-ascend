# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve V4.1 pinned staging without enabling CUDA UVA on Ascend.

The upstream resolver only initializes Engram on CUDA. Keep other models and
platforms on that resolver; remove this patch when it accepts backend storage
validation. Ascend uses explicit host lookup and H2D staging, not UVA lookup.
"""

from vllm.config import EngramConfig, VllmConfig
from vllm.platforms import current_platform

_ORIGINAL_RESOLVE = VllmConfig._resolve_and_verify_engram_config


def _resolve_and_verify_engram_config(self: VllmConfig) -> None:
    model = self.model_config
    if (
        current_platform.device_type != "npu"
        or model is None
        or model.architecture != "DeepseekV41ForCausalLM"
        or not getattr(model.hf_text_config, "engram_layer_ids", None)
    ):
        return _ORIGINAL_RESOLVE(self)

    parallel = self.parallel_config
    engram = self.engram_config
    if engram is None:
        engram = EngramConfig(cpu_offload=True)
    if not engram.cpu_offload or engram.embedding_across_dp or engram.dp_shared_memory:
        raise ValueError(
            "Ascend V4.1 requires pinned CPU Engram offload with TP-local shards; "
            "embedding_across_dp and dp_shared_memory are not supported"
        )
    if parallel.tensor_parallel_size != 8 or parallel.pipeline_parallel_size != 1:
        raise ValueError("Ascend V4.1 currently requires TP8 and PP1")
    if parallel.data_parallel_size != 1:
        raise ValueError("Ascend V4.1 typed image routing currently requires DP1")
    if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("Ascend V4.1 Engram does not yet support context parallelism")
    if parallel.use_ubatching:
        raise ValueError("Ascend V4.1 Engram does not support DBO or microbatching")
    if parallel.enable_expert_parallel or parallel.use_sequence_parallel_moe or parallel.enable_elastic_ep:
        raise ValueError("Ascend V4.1 Engram requires replicated token rows and TP MoE")
    if self.speculative_config is not None:
        raise ValueError("Ascend V4.1 speculative model integration is not yet enabled")
    load = self.load_config
    if load.load_format not in {"auto", "safetensors", "dummy"}:
        raise ValueError("Ascend V4.1 host Engram requires the safetensors loader (or dummy test weights)")
    strategy = getattr(load, "safetensors_load_strategy", None)
    if strategy is None:
        load.safetensors_load_strategy = "lazy"
    elif strategy not in {"lazy", "prefetch"}:
        # DefaultModelLoader obtains each tensor before the model can skip its
        # host tables. Eager f.read()+load would privately materialize a whole
        # ~183 GiB table shard in every TP worker, before pinned head sharding.
        raise ValueError("Ascend V4.1 requires lazy or prefetch safetensors loading for host Engram tables")
    engram.verify_parallel_config(parallel)
    engram.verify_load_config(self.load_config)
    self.engram_config = engram


VllmConfig._resolve_and_verify_engram_config = _resolve_and_verify_engram_config
