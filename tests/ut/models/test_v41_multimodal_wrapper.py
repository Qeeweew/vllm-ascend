# SPDX-License-Identifier: Apache-2.0
"""CPU boundaries for the Ascend V4.1 multimodal composition."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.inputs import PlaceholderRange

from vllm_ascend.models.deepseek_v4 import model as models
from vllm_ascend.ops import mm_encoder_attention
from vllm_ascend.patch.worker import patch_deepseek_v41_mm as processor


def test_production_architecture_exposes_multimodal_wrapper():
    from vllm import ModelRegistry
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    registered = ModelRegistry.models["DeepseekV41ForCausalLM"]
    assert registered.load_model_cls() is models.AscendDeepseekV41ForConditionalGeneration
    info = registered.inspect_model_cls()
    assert info.supports_multimodal
    assert models.AscendDeepseekV41ForConditionalGeneration._processor_factory is not None


class LanguageModel(torch.nn.Module):
    def __init__(self, *, vllm_config, prefix):
        super().__init__()
        self.width = vllm_config.model_config.hf_config.hidden_size
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.prefix = prefix
        self.calls = []
        self.load_calls = 0

    def embed_input_ids(self, ids):
        return ids[:, None].expand(-1, self.width).to(torch.bfloat16).clone()

    def forward(self, ids, positions, intermediate, embeddings, **kwargs):
        self.calls.append((ids, positions, intermediate, embeddings, kwargs))
        return self.embed_input_ids(ids) if embeddings is None else embeddings

    def compute_logits(self, hidden):
        return hidden + 1

    def create_engram_runtime(self):
        return self.calls

    def get_expert_mapping(self):
        return [("anchor", "source", 0, "w1")]

    def load_weights(self, weights):
        self.load_calls += 1
        loaded = set()
        for name, value in weights:
            assert name == "anchor"
            self.anchor.data.copy_(value)
            loaded.add(name)
        return loaded


def config(image_limit=1):
    hf = SimpleNamespace(
        hidden_size=8,
        vision_dim=16,
        vision_n_heads=2,
        vision_n_layers=1,
        vision_inter_dim=24,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        vision_max_n_token=1024,
        vision_min_pixels=0,
        vision_max_wh_ratio=None,
        image_token_id=129264,
    )
    mm = MultiModalConfig(limit_per_prompt={"image": image_limit})
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, dtype=torch.bfloat16, get_multimodal_config=lambda: mm),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False, compile_mm_encoder=False),
        scheduler_config=SimpleNamespace(disable_chunked_mm_input=True),
    )


@pytest.fixture
def make_wrapper(monkeypatch):
    monkeypatch.setattr(models, "AscendDeepseekV41ForCausalLM", LanguageModel)
    monkeypatch.setattr(
        mm_encoder_attention, "AscendMMEncoderAttention", lambda heads, dim: models.AscendV41VisionSDPA()
    )

    def make(cfg=None):
        wrapper = models.AscendDeepseekV41ForConditionalGeneration(vllm_config=cfg or config(), prefix="outer")
        if wrapper.image_limit:
            with torch.no_grad():
                wrapper.image_start.fill_(10)
                wrapper.image_newline.fill_(20)
                wrapper.image_end.fill_(30)
        return wrapper

    return make


def inputs(grids=((4, 5),)):
    aligned = [((height + 2) // 3, (width + 2) // 3) for height, width in grids]
    return dict(
        patches=torch.randn(sum(height * width for height, width in grids), 3, 2, 2).bfloat16(),
        vit_grid=torch.tensor(grids),
        llm_grid=torch.tensor(aligned),
        types=torch.cat([processor.image_token_types(*grid) for grid in aligned]),
    )


def test_eager_encoder_delimiters_reading_order_and_batched_images(make_wrapper):
    wrapper = make_wrapper()
    data = inputs(((4, 5), (1, 1)))
    with torch.inference_mode():
        spans = wrapper.embed_multimodal(**data)
        expected = wrapper.aligner(wrapper.vision(data["patches"][:20], 4, 5), 4, 5)
    assert [span.shape for span in spans] == [(8, 8), (4, 8)]
    torch.testing.assert_close(spans[0][[1, 2, 4, 5]], expected, rtol=0, atol=0)
    torch.testing.assert_close(spans[0][[0, 3, 6, 7], 0], torch.tensor([10, 20, 20, 30]).bfloat16())
    assert wrapper._language_model_names == ["language_model"]
    assert set(wrapper._tower_model_names) == {"vision", "aligner"}
    assert wrapper.language_model.prefix == "outer.language_model"
    assert wrapper.get_language_model() is wrapper.language_model
    assert wrapper.requires_raw_input_tokens and not wrapper.supports_encoder_tp_data
    assert wrapper.get_placeholder_str("image", 0) == processor.IMAGE_PLACEHOLDER
    with pytest.raises(ValueError, match="Unsupported"):
        wrapper.get_placeholder_str("audio", 0)


def test_consumes_existing_processor_batchfeature_protocol(make_wrapper):
    wrapper = make_wrapper()
    encoded = processor.DeepseekV41VLProcessor(wrapper.config)(images=[Image.new("RGB", (7, 9), (33, 91, 147))])
    spans = wrapper.embed_multimodal(**encoded)
    assert encoded["vit_grid"].tolist() == [[5, 4]]
    assert encoded["llm_grid"].tolist() == [[2, 2]]
    assert len(spans) == 1 and spans[0].shape == (8, 8)


def test_merge_full_span_then_forward_preserves_ids_embeddings_and_typed_mask(make_wrapper):
    wrapper = make_wrapper()
    spans = wrapper.embed_multimodal(**inputs(((1, 1),)))
    ids = torch.tensor([7, 129264, 129264, 129264, 129264, 9])
    mask = torch.tensor([False, True, True, True, True, False])
    merged = wrapper.embed_input_ids(ids, spans, is_multimodal=mask)
    torch.testing.assert_close(merged[1:5], spans[0], rtol=0, atol=0)
    assert merged.shape == (6, 8)  # HC expansion belongs to the language child.
    assert merged[0, 0] == 7 and merged[-1, 0] == 9
    positions = torch.arange(6)
    result = wrapper(ids, positions, inputs_embeds=merged, image_token_mask=mask, engram_token_mask=~mask)
    call = wrapper.language_model.calls[-1]
    assert call[0] is ids and call[1] is positions and call[3] is merged
    assert call[4]["image_token_mask"] is mask and call[4]["engram_token_mask"].equal(~mask)
    assert result is merged
    assert wrapper.create_engram_runtime() is wrapper.language_model.calls
    assert wrapper.get_expert_mapping() == wrapper.language_model.get_expert_mapping()
    torch.testing.assert_close(wrapper.compute_logits(merged), merged + 1)
    with pytest.raises(ValueError, match="raw token IDs"):
        wrapper(None, positions, inputs_embeds=merged)


@pytest.mark.parametrize("failure", ["missing", "dtype", "shape", "device"])
def test_forward_requires_explicit_typed_image_mask(make_wrapper, failure):
    wrapper = make_wrapper()
    ids = torch.tensor([129264, 7])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    if failure == "missing":
        mask = None
    elif failure == "dtype":
        mask = mask.long()
    elif failure == "shape":
        mask = mask[:1]
    else:
        mask = mask.to("meta")
    with pytest.raises(ValueError, match="explicit bool image_token_mask"):
        wrapper(ids, torch.arange(2), image_token_mask=mask)
    assert not wrapper.language_model.calls


@pytest.mark.parametrize("image_limit", [0, 1])
def test_text_forward_keeps_literal_image_id_and_independent_history_validity(make_wrapper, image_limit):
    wrapper = make_wrapper(config(image_limit))
    ids = torch.tensor([129264, 7])
    # A historical n-gram can be invalid at a text position. That position
    # must not become a vision routing position by negating Engram keep.
    image_mask = torch.zeros(2, dtype=torch.bool)
    keep_mask = torch.tensor([False, True])
    wrapper(ids, torch.arange(2), image_token_mask=image_mask, engram_token_mask=keep_mask)
    typed = wrapper.language_model.calls[-1][4]
    assert typed["image_token_mask"] is image_mask
    assert typed["engram_token_mask"] is keep_mask


@pytest.mark.parametrize("failure", ["missing_mask", "omit_delimiter", "wrong_raw_id", "mask_dtype"])
def test_embedding_merge_rejects_incomplete_or_wrong_masks(make_wrapper, failure):
    wrapper = make_wrapper()
    spans = wrapper.embed_multimodal(**inputs(((1, 1),)))
    ids = torch.full((4,), 129264)
    mask = torch.ones(4, dtype=torch.bool)
    if failure == "missing_mask":
        mask = None
    elif failure == "omit_delimiter":
        mask[0] = False
    elif failure == "wrong_raw_id":
        ids[0] = 129265
    else:
        mask = mask.long()
    with pytest.raises(ValueError):
        wrapper.embed_input_ids(ids, spans, is_multimodal=mask)


@pytest.mark.parametrize("failure", ["grid", "roles", "extra_roles", "patches", "budget", "missing_field"])
def test_encoder_rejects_inconsistent_processor_contract_before_forward(make_wrapper, failure):
    wrapper = make_wrapper()
    data = inputs()
    if failure == "grid":
        data["llm_grid"][0, 0] += 1
    elif failure == "roles":
        data["types"][0] = 1
    elif failure == "extra_roles":
        data["types"] = torch.cat((data["types"], torch.tensor([3])))
    elif failure == "patches":
        data["patches"] = data["patches"][:-1]
    elif failure == "budget":
        wrapper.config.vision_max_n_token = 7
    else:
        data.pop("types")
    with pytest.raises(ValueError):
        wrapper.embed_multimodal(**data)


def request(ids, offset=1):
    feature = SimpleNamespace(
        modality="image",
        mm_position=PlaceholderRange(offset=offset, length=4),
        data={"types": SimpleNamespace(data=processor.image_token_types(1, 1))},
    )
    return SimpleNamespace(prompt_token_ids=ids, mm_features=[feature])


def test_prompt_mask_uses_owned_ranges_including_delimiters_and_keeps_literal_id(make_wrapper):
    wrapper = make_wrapper()
    value = request([129264, 129264, 129264, 129264, 129264, 8])
    keep = wrapper.engram_prompt_mask(value)
    assert keep.device.type == "cpu" and keep.dtype == torch.bool
    assert keep.tolist() == [True, False, False, False, False, True]
    value.mm_features[0].data = None  # Processor cache hit still has a complete range.
    assert wrapper.engram_prompt_mask(value).equal(keep)
    value.prompt_token_ids[2] = 129265
    with pytest.raises(ValueError, match="raw token"):
        wrapper.engram_prompt_mask(value)


def test_one_image_limit_applies_per_request_not_encoder_batch(make_wrapper):
    wrapper = make_wrapper()
    value = request([129264] * 8, offset=0)
    value.mm_features.append(request([129264] * 8, offset=4).mm_features[0])
    with pytest.raises(ValueError, match="image limit"):
        wrapper.engram_prompt_mask(value)


def checkpoint(wrapper):
    return [(name, value.detach().bfloat16().clone()) for name, value in wrapper.named_parameters()]


def test_streaming_weight_dispatch_once_and_real_parameter_names(make_wrapper):
    wrapper = make_wrapper()
    values = checkpoint(wrapper)
    values.reverse()  # Interleaved ordering must not require sorting/materialization.
    yielded = []

    def stream():
        for name, tensor in values:
            yielded.append(name)
            yield name, tensor

    loaded = wrapper.load_weights(stream())
    assert loaded == set(dict(wrapper.named_parameters()))
    assert yielded == [name for name, _ in values]
    assert wrapper.language_model.load_calls == 1


def test_full_checkpoint_all_266_mm_names_shapes_and_dispatch_without_allocation(make_wrapper):
    root = Path("/mnt/models/DeepSeek-V4.1-Flash")
    if not (root / "config.json").is_file():
        pytest.skip("released V4.1 checkpoint is unavailable")
    document = json.loads((root / "config.json").read_text())
    vision = document["vision_config"]
    cfg = config()
    hf = cfg.model_config.hf_config
    for name, key in (
        ("vision_dim", "hidden_size"),
        ("vision_n_heads", "num_attention_heads"),
        ("vision_n_layers", "num_hidden_layers"),
        ("vision_inter_dim", "intermediate_size"),
        ("vision_patch_size", "patch_size"),
        ("vision_rope_theta", "rope_theta"),
        ("vision_downsample_ratio", "downsample_ratio"),
    ):
        setattr(hf, name, vision[key])
    hf.hidden_size = document["text_config"]["hidden_size"]
    with torch.device("meta"):
        wrapper = make_wrapper(cfg)
    parameters = dict(wrapper.named_parameters())
    names = set(parameters) - {"language_model.anchor"}
    assert len(names) == 266
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    mm_names = {
        name
        for name in index
        if name.startswith(("vision.", "aligner.")) or name in {"image_start", "image_end", "image_newline"}
    }
    assert mm_names == names
    headers = {}

    def stream():
        yield "anchor", torch.ones(1, dtype=torch.bfloat16, device="meta")
        for name in sorted(names):
            shard = index[name]
            if shard not in headers:
                with (root / shard).open("rb") as reader:
                    size = struct.unpack("<Q", reader.read(8))[0]
                    headers[shard] = json.loads(reader.read(size))
            descriptor = headers[shard][name]
            assert descriptor["shape"] == list(parameters[name].shape)
            assert descriptor["dtype"] == "BF16"
            yield name, torch.empty(descriptor["shape"], dtype=torch.bfloat16, device="meta")

    assert wrapper.load_weights(stream()) == set(parameters)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "shape", "dtype", "unexpected"])
def test_vision_weight_dispatch_rejects_incomplete_or_invalid_checkpoint(make_wrapper, failure):
    wrapper = make_wrapper()
    values = checkpoint(wrapper)
    index = next(index for index, (name, _) in enumerate(values) if name == "image_start")
    if failure == "missing":
        values.pop(index)
    elif failure == "duplicate":
        values.append(values[index])
    elif failure == "shape":
        values[index] = ("image_start", torch.zeros(1, dtype=torch.bfloat16))
    elif failure == "dtype":
        values[index] = ("image_start", values[index][1].float())
    else:
        values.append(("vision.unexpected", torch.zeros(1, dtype=torch.bfloat16)))
    with pytest.raises(ValueError):
        wrapper.load_weights(iter(values))


def test_image_limit_zero_has_no_vision_allocations_and_keeps_text_path(make_wrapper):
    wrapper = make_wrapper(config(image_limit=0))
    assert wrapper.vision is wrapper.aligner is wrapper.image_start is None
    assert set(dict(wrapper.named_parameters())) == {"language_model.anchor"}
    loaded = wrapper.load_weights(
        iter([("vision.unused.weight", torch.ones(1)), ("image_start", torch.ones(8)), ("anchor", torch.ones(1))])
    )
    assert loaded == {"language_model.anchor"}
    ids = torch.tensor([7, 129264])
    torch.testing.assert_close(wrapper.embed_input_ids(ids), wrapper.language_model.embed_input_ids(ids))
    assert wrapper.embed_multimodal() == ()
    assert wrapper.engram_prompt_mask(SimpleNamespace(prompt_token_ids=ids.tolist(), mm_features=[])).all()
    with pytest.raises(ValueError, match="disabled"):
        wrapper.embed_multimodal(**inputs())


@pytest.mark.parametrize(
    "failure",
    [
        "image_limit",
        "pp",
        "spec",
        "encoder_dp",
        "encoder_only",
        "dtype",
        "encoder_graph",
        "encoder_compile",
        "chunked_image",
        "external_embeddings",
    ],
)
def test_initial_wrapper_scope_is_explicit(make_wrapper, failure):
    cfg = config(2 if failure == "image_limit" else 1)
    if failure == "pp":
        cfg.parallel_config.pipeline_parallel_size = 2
    elif failure == "spec":
        cfg.speculative_config = object()
    elif failure == "encoder_dp":
        cfg.model_config.get_multimodal_config().mm_encoder_tp_mode = "data"
    elif failure == "encoder_only":
        cfg.model_config.get_multimodal_config().mm_encoder_only = True
    elif failure == "dtype":
        cfg.model_config.dtype = torch.float16
    elif failure == "encoder_graph":
        cfg.compilation_config.cudagraph_mm_encoder = True
    elif failure == "encoder_compile":
        cfg.compilation_config.compile_mm_encoder = True
    elif failure == "chunked_image":
        cfg.scheduler_config.disable_chunked_mm_input = False
    elif failure == "external_embeddings":
        cfg.model_config.get_multimodal_config().enable_mm_embeds = True
    with pytest.raises(ValueError):
        make_wrapper(cfg)


def test_text_only_wrapper_does_not_require_image_scheduler_or_encoder_options(make_wrapper):
    cfg = config(0)
    cfg.compilation_config = None
    cfg.scheduler_config = None
    assert make_wrapper(cfg).vision is None


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6, 7, 8])
def test_text_only_wrapper_admits_configurable_dspark(make_wrapper, tokens):
    cfg = config(0)
    cfg.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=tokens)
    assert make_wrapper(cfg).vision is None


@pytest.mark.parametrize(("method", "tokens"), [("mtp", 5), ("dspark", 0), ("dspark", 9)])
def test_text_only_wrapper_rejects_unimplemented_speculation(make_wrapper, method, tokens):
    cfg = config(0)
    cfg.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=tokens)
    with pytest.raises(ValueError, match="text-only DSpark"):
        make_wrapper(cfg)


@pytest.mark.parametrize("language_only", [False, True])
def test_default_image_limit_matches_processor_single_image_support(make_wrapper, language_only):
    cfg = config()
    mm = MultiModalConfig(language_model_only=language_only)
    cfg.model_config.get_multimodal_config = lambda: mm
    wrapper = make_wrapper(cfg)
    assert wrapper.image_limit == (0 if language_only else 1)
    assert (wrapper.vision is None) == language_only
    info = processor.DeepseekV41VLProcessingInfo.__new__(processor.DeepseekV41VLProcessingInfo)
    assert info.get_supported_mm_limits() == {"image": 1}
