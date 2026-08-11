# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from transformers import HunYuanVLProcessor

_STALE_PROCESSOR_MODULES = {
    "HunYuanVLProcessor": "vllm.transformers_utils.processors.hunyuan_vl",
    "HunYuanVLImageProcessor": "vllm.transformers_utils.processors.hunyuan_vl_image",
}

_HUNYUAN_VL_EXTRA_SPECIAL_TOKENS = {
    "image_start_token": "<｜hy_place▁holder▁no▁100｜>",
    "image_end_token": "<｜hy_place▁holder▁no▁101｜>",
    "image_token": "<｜hy_place▁holder▁no▁102｜>",
}
_HUNYUAN_VL_SPECIAL_TOKENS = {
    **_HUNYUAN_VL_EXTRA_SPECIAL_TOKENS,
    "pad_token": "<｜hy_▁pad▁｜>",
}
_HUNYUAN_VL_SPECIAL_TOKEN_IDS = {
    "image_start_token": 120118,
    "image_end_token": 120119,
    "image_token": 120120,
    "pad_token": 120002,
}


def _register_hunyuan_tokenizer_special_tokens(tokenizer: Any) -> None:
    """Restore the named-token schema required by Transformers 5.13."""
    missing_tokens = {
        name: token
        for name, token in _HUNYUAN_VL_EXTRA_SPECIAL_TOKENS.items()
        if tokenizer is not None and getattr(tokenizer, name, None) is None
    }
    if missing_tokens:
        tokenizer._set_model_specific_special_tokens(special_tokens=missing_tokens)

    actual_tokens = {
        name: (getattr(tokenizer, name, None), getattr(tokenizer, f"{name}_id", None))
        for name in _HUNYUAN_VL_SPECIAL_TOKENS
    }
    expected_tokens = {
        name: (token, _HUNYUAN_VL_SPECIAL_TOKEN_IDS[name]) for name, token in _HUNYUAN_VL_SPECIAL_TOKENS.items()
    }
    if actual_tokens != expected_tokens:
        raise ValueError(
            "HunyuanVL tokenizer special-token schema does not match the model vocabulary: "
            f"expected {expected_tokens!r}, got {actual_tokens!r}"
        )


class _HunYuanVLProcessorCompat(HunYuanVLProcessor):
    """Native processor with the legacy HunyuanOCR token schema restored."""

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: Any = None,
        chat_template: Any = None,
        cat_extra_token: bool = True,
        **kwargs: Any,
    ) -> None:
        _register_hunyuan_tokenizer_special_tokens(tokenizer)
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            cat_extra_token=cat_extra_token,
            **kwargs,
        )


def _remove_stale_registry_entries() -> bool:
    """Backport the lazy-registry cleanup from vLLM PR #47867."""
    import vllm.transformers_utils.processors as vllm_processors

    class_to_module = vllm_processors._CLASS_TO_MODULE
    exported_names = vllm_processors.__all__
    entries_to_remove = []
    for class_name, stale_module in _STALE_PROCESSOR_MODULES.items():
        registered_module = class_to_module.get(class_name)
        if registered_module is None:
            continue
        if registered_module != stale_module:
            raise RuntimeError(f"Unexpected vLLM processor registry entry for {class_name}: {registered_module!r}")
        if class_name not in exported_names:
            raise RuntimeError(f"Missing vLLM processor export for {class_name}")

        entries_to_remove.append(class_name)

    for class_name in entries_to_remove:
        del class_to_module[class_name]
        exported_names.remove(class_name)

    return bool(entries_to_remove)


def _patch_hunyuan_processor_loader(hunyuan_vision: Any) -> None:
    """Use the native processor with the complete Hunyuan tokenizer schema."""

    def get_hf_processor(self: Any, **kwargs: object) -> Any:
        kwargs.pop("use_fast", None)
        kwargs.setdefault("backend", "pil")
        return self.ctx.get_hf_processor(_HunYuanVLProcessorCompat, **kwargs)

    hunyuan_vision.HunYuanVLProcessingInfo.get_hf_processor = get_hf_processor


def install_hunyuan_vl_processor_compat() -> None:
    """Align both supported vLLM refs with Transformers 5.13 Hunyuan APIs."""
    _remove_stale_registry_entries()
    from vllm.model_executor.models import hunyuan_vision as main_hunyuan_vision

    _patch_hunyuan_processor_loader(main_hunyuan_vision)
