# SPDX-License-Identifier: Apache-2.0
"""CPU V4.1 image and prompt contract against the local release and tokenizer."""

import copy
import io
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.processing import InputProcessingContext
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_ascend.patch.worker import patch_deepseek_v41_mm as mm

RELEASE_ROOT = Path("/mnt/models/DeepSeek-V4.1-Flash")


@pytest.fixture(scope="module")
def release():
    required = [RELEASE_ROOT / name for name in ("config.json", "tokenizer.json", "inference/image_processor.py")]
    if not all(path.is_file() for path in required):
        pytest.skip("DeepSeek-V4.1 release tokenizer and CPU image reference are unavailable")
    config = DeepseekV41Config(**json.loads(required[0].read_text()))
    oracle = runpy.run_path(str(required[2]))
    tokenizer = AutoTokenizer.from_pretrained(RELEASE_ROOT, local_files_only=True, trust_remote_code=False)
    return config, oracle, tokenizer


def make_image(width, height, mode="RGB"):
    data = (np.arange(width * height * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(height, width, 3)
    return Image.fromarray(data).convert(mode)


def image_record(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"data": buffer.getvalue()}


def processor_for(config, tokenizer, *, allow_multi_image_reference=False):
    multimodal = MultiModalConfig(limit_per_prompt={"image": 8})
    model_config = SimpleNamespace(
        model=str(RELEASE_ROOT),
        hf_config=config,
        dtype=torch.bfloat16,
        max_model_len=16384,
        encoder_config=None,
        multimodal_config=multimodal,
        get_multimodal_config=lambda: multimodal,
    )
    info_type = mm.DeepseekV41VLProcessingInfo
    if allow_multi_image_reference:
        # Exercise general span formatting independently of the initial
        # production admission limit of one image per request.
        class MultiImageReferenceInfo(mm.DeepseekV41VLProcessingInfo):
            def get_supported_mm_limits(self):
                return {"image": None}

        info_type = MultiImageReferenceInfo
    info = info_type(InputProcessingContext(model_config, tokenizer))
    return mm.DeepseekV41VLMultiModalProcessor(info, mm.DeepseekV41VLDummyInputsBuilder(info))


@pytest.mark.parametrize("size", [(1, 1), (37, 53), (544, 544), (1023, 777), (2048, 1536), (4096, 8), (8, 4096)])
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_release_exact_resize_patch_values_and_reading_order(release, size, mode):
    config, oracle, _ = release
    image = make_image(*size, mode=mode)
    actual = mm.DeepseekV41VLImageProcessor(config)(image)
    expected = oracle["load_image"](image_record(image), config)
    assert actual[1:] == expected[1:]
    assert actual[0].dtype == torch.bfloat16 and actual[0].device.type == "cpu"
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    h, w = actual[-2:]
    torch.testing.assert_close(mm.image_token_types(h, w), oracle["image_token_types"](h, w), rtol=0, atol=0)
    assert mm.num_image_tokens(h, w) <= config.vision_max_n_token


@pytest.mark.parametrize("size", [(70, 98), (98, 70), (98, 98), (127, 113)])
def test_odd_patch_and_llm_grids_have_no_v4_padding_or_permutation(release, size):
    source_config, oracle, _ = release
    config = copy.deepcopy(source_config)
    config.vision_min_pixels = 0
    image = make_image(*size)
    actual = mm.DeepseekV41VLProcessor(config)(images=[image])
    expected = oracle["load_image"](image_record(image), config)
    assert actual["vit_grid"].tolist() == [[expected[1], expected[2]]]
    assert actual["llm_grid"].tolist() == [[expected[3], expected[4]]]
    torch.testing.assert_close(actual["patches"], expected[0], rtol=0, atol=0)
    assert actual["types"].tolist() == [mm.IMAGE_START] + ([mm.IMAGE] * expected[4] + [mm.IMAGE_NEW_LINE]) * expected[
        3
    ] + [mm.IMAGE_END]
    assert actual["types"].numel() == expected[3] * (expected[4] + 1) + 2
    assert "perm" not in actual


@pytest.mark.parametrize("size", [(800, 100), (801, 100), (100, 800)])
def test_configured_aspect_cap_exact_boundary_and_stretch_branch(release, size):
    source_config, oracle, _ = release
    config = copy.deepcopy(source_config)
    config.vision_max_wh_ratio = 8
    image = make_image(*size)
    actual = mm.DeepseekV41VLImageProcessor(config)(image)
    expected = oracle["load_image"](image_record(image), config)
    assert actual[1:] == expected[1:]
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)


@pytest.mark.parametrize("prefix", ["", "看图：", "text " * 17])
@pytest.mark.parametrize("adjacent", [False, True])
def test_real_tokenizer_multiimage_full_base_processor_matches_reference_spans(release, prefix, adjacent):
    config, oracle, tokenizer = release
    processor = processor_for(config, tokenizer, allow_multi_image_reference=True)
    placeholder = mm.IMAGE_PLACEHOLDER
    assert tokenizer.encode(placeholder, add_special_tokens=False) == [config.image_token_id]
    images = [make_image(98, 70), make_image(4096, 8, mode="RGBA")]
    prompt = prefix + placeholder + ("" if adjacent else " comparison ") + placeholder + "结束"
    raw_tokens = tokenizer.encode(prompt)
    assert raw_tokens.count(config.image_token_id) == len(images)
    expected_tokens, expected_types, starts = [], [], []
    for token in raw_tokens:
        if token == config.image_token_id:
            image_index = len(starts)
            expected = oracle["load_image"](image_record(images[image_index]), config)
            types = oracle["image_token_types"](expected[3], expected[4])
            starts.append((len(expected_tokens), types.numel()))
            expected_tokens.extend([config.image_token_id] * types.numel())
            expected_types.append(types)
        else:
            expected_tokens.append(token)
    output = processor(prompt, processor.info.parse_mm_data({"image": images}))
    assert output["prompt_token_ids"] == expected_tokens
    ranges = output["mm_placeholders"]["image"]
    assert [(item.offset, item.length) for item in ranges] == starts
    for index, item in enumerate(ranges):
        assert item.get_num_embeds() == item.length  # Delimiters also get learned embeddings.
        assert item.is_embed is None or bool(item.is_embed.all())
        data = output["mm_kwargs"]["image"][index]
        torch.testing.assert_close(data["types"].data, expected_types[index], rtol=0, atol=0)
        assert data["types"].data.device.type == "cpu"
        assert data["vit_grid"].data.device.type == "cpu"
        assert data["llm_grid"].data.device.type == "cpu"
    assert set(output["mm_kwargs"]["image"][0]) == {"patches", "vit_grid", "llm_grid", "types"}


@pytest.mark.parametrize("placeholders,images", [(0, 1), (1, 2), (2, 1)])
def test_placeholder_image_count_mismatch_rejected_in_both_directions(release, placeholders, images):
    config, _, tokenizer = release
    processor = processor_for(config, tokenizer, allow_multi_image_reference=True)
    data = {"image": [make_image(32, 32) for _ in range(images)]} if images else {}
    with pytest.raises(ValueError, match=f"Found {placeholders} image tokens but got {images} images"):
        processor(mm.IMAGE_PLACEHOLDER * placeholders, processor.info.parse_mm_data(data))


def test_production_processor_admits_only_one_image_per_request(release):
    config, _, tokenizer = release
    processor = processor_for(config, tokenizer)
    assert processor.info.allowed_mm_limits == {"image": 1}
    with pytest.raises(VLLMValidationError, match="1 image"):
        processor(
            mm.IMAGE_PLACEHOLDER * 2,
            processor.info.parse_mm_data({"image": [make_image(32, 32), make_image(32, 32)]}),
        )


@pytest.mark.parametrize("prompt", ["只有文本，没有图片。", f"literal {mm.IMAGE_PLACEHOLDER} text"])
def test_text_only_preserves_real_token_ids_and_empty_multimodal_fields(release, prompt):
    config, _, tokenizer = release
    processor = processor_for(config, tokenizer)
    output = processor(prompt, processor.info.parse_mm_data({}))
    assert output["prompt_token_ids"] == tokenizer.encode(prompt)
    assert not output["mm_placeholders"] and not output["mm_kwargs"]
    assert not mm.DeepseekV41VLProcessor(config)(images=[])


def test_image_rejected_when_config_has_no_vision_tower(release):
    original, _, tokenizer = release
    config = copy.deepcopy(original)
    config.vision_n_layers = 0
    processor = processor_for(config, tokenizer)
    with pytest.raises(ValueError, match="no vision tower"):
        processor(mm.IMAGE_PLACEHOLDER, processor.info.parse_mm_data({"image": [make_image(32, 32)]}))


def test_uncapped_release_dummy_exercises_full_token_and_patch_budget(release):
    config, _, tokenizer = release
    processor = processor_for(config, tokenizer)
    size = processor.info.get_image_size_with_most_features()
    output = mm.DeepseekV41VLProcessor(config)(images=[Image.new("RGB", (size.width, size.height))])
    assert output["types"].numel() == config.vision_max_n_token == 1024
    assert output["patches"].shape[0] == 9189
    assert processor.info.get_mm_max_tokens_per_item(16384, {"image": 1}) == {"image": 1024}
    assert processor.dummy_inputs.get_dummy_text({"image": 2}) == mm.IMAGE_PLACEHOLDER * 2


def test_bad_tokenizer_image_id_is_rejected():
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 129265)
    with pytest.raises(ValueError, match="expected 129264"):
        mm.validate_image_sentinel_ids(tokenizer)


def test_v41_image_mask_does_not_treat_v4_role_ids_as_image_spans():
    ids = torch.tensor([129263, 129264, 129265, 129266, 129267, 129268])
    assert mm.image_sentinel_mask(ids).tolist() == [False, True, False, False, False, False]
