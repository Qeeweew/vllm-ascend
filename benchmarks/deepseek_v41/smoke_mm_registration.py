# SPDX-License-Identifier: Apache-2.0
"""Explicit test-only registration; production registry remains untouched."""


def register_mm_smoke_model():
    # Lazy imports isolate CPU checkpoint preparation from model/plugin setup.
    from vllm import ModelRegistry
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV41ForConditionalGeneration
    from vllm_ascend.patch.worker.patch_deepseek_v41_mm import (
        DeepseekV41VLDummyInputsBuilder,
        DeepseekV41VLMultiModalProcessor,
        DeepseekV41VLProcessingInfo,
    )

    MULTIMODAL_REGISTRY.register_processor(
        DeepseekV41VLMultiModalProcessor,
        info=DeepseekV41VLProcessingInfo,
        dummy_inputs=DeepseekV41VLDummyInputsBuilder,
    )(AscendDeepseekV41ForConditionalGeneration)
    ModelRegistry.register_model("DeepseekV41ForCausalLM", AscendDeepseekV41ForConditionalGeneration)
