# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint format and resumability tests, independent of NPU kernels."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONVERTER_PATH = Path(__file__).resolve().parents[3] / "examples" / "quantization" / "convert_deepseek_v41.py"
SPEC = importlib.util.spec_from_file_location("convert_deepseek_v41", CONVERTER_PATH)
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def unpack_checkpoint(packed):
    return torch.stack([((packed >> (4 * i)) & 15) - 8 for i in range(8)], dim=-1).flatten(-2).to(torch.int8)


def test_all_fp4_codes_have_correct_low_high_order():
    # Deliberately put distinct adjacent codes in each byte.
    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[::2] | (codes[1::2] << 4)).view(torch.int8).reshape(1, 16)
    actual = converter.dequantize_mxfp4(packed, torch.tensor([[2.0]]))
    expected = torch.tensor(converter.FP4_VALUES * 2).reshape(1, 32).mul(2).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_signed_scale_dominance_ties_and_zero():
    w = torch.zeros(4, 32, dtype=torch.bfloat16)
    w[0, :2] = torch.tensor([6.0, -4.0])
    w[1, :2] = torch.tensor([-6.0, 4.0])
    w[2, :2] = torch.tensor([-6.0, 6.0])
    q, s = converter.signed_scale_rtn(w)
    assert s[0, 0] == -0.75 and s[1, 0] == 0.75
    assert s[2, 0] == torch.tensor(-6 / 7).bfloat16()
    assert q[0, 0] == -8 and q[1, 0] == -8
    assert q[2, 0] == 7 and q[2, 1] == -7
    assert torch.equal(q[3], torch.zeros(32, dtype=torch.int8))
    assert torch.isfinite(s).all() and (s != 0).all()
    assert torch.equal(unpack_checkpoint(converter.pack_checkpoint_int4(q)), q)


def test_checkpoint_packing_is_offset_binary_not_twos_complement():
    q = torch.arange(-8, 8, dtype=torch.int8).repeat(2).reshape(1, 32)
    packed = converter.pack_checkpoint_int4(q)
    assert (packed[0, 0].item() & 0xFFFFFFFF) == 0x76543210
    assert torch.equal(unpack_checkpoint(packed), q)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_rtn_rejects_nonfinite_weights(bad):
    w = torch.zeros(1, 32)
    w[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        converter.signed_scale_rtn(w)


def make_source(path, cross_shard=True):
    path.mkdir()
    expert = "layers.0.ffn.experts.0.w1"
    dense = "layers.0.attn.wkv"
    embedding = "layers.1.engram.embed"
    weights = {
        expert + ".weight": torch.arange(32 * 16, dtype=torch.int16).remainder(256).to(torch.int8).reshape(32, 16),
        dense + ".weight": torch.ones(64, 64).to(torch.float8_e4m3fn),
        embedding + ".weight": torch.ones(5, 64).to(torch.float8_e4m3fn),
        "layers.0.attn.attn_sink": torch.tensor([1.25, -float("inf")]),
        "norm.weight": torch.ones(32, dtype=torch.bfloat16),
    }
    scales = {
        expert + ".scale": torch.full((32, 1), 2.0).to(torch.float8_e8m0fnu),
        dense + ".scale": torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e8m0fnu),
        embedding + ".scale": torch.tensor([[1.0, 2.0]] * 5).to(torch.float8_e8m0fnu),
    }
    shards = {"model-00001.safetensors": weights, "model-00002.safetensors": scales}
    if not cross_shard:
        shards = {"model-00001.safetensors": weights | scales}
    weight_map = {}
    for filename, data in shards.items():
        save_file(data, path / filename)
        weight_map.update(dict.fromkeys(data, filename))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
                "text_config": {"quantization_config": {"quant_method": "fp8"}, "hidden_size": 32},
            }
        )
    )
    (path / "tokenizer_config.json").write_text("{}")
    return weights, scales


def test_cross_shard_conversion_matches_tensor_reference_and_preserves_f32(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    weights, scales = make_source(source)
    manifest = converter.convert(source, output, rows=16)
    assert manifest["complete"]
    with safe_open(output / "model-00001.safetensors", framework="pt") as reader:
        q, s = converter.signed_scale_rtn(
            converter.dequantize_mxfp4(
                weights["layers.0.ffn.experts.0.w1.weight"], scales["layers.0.ffn.experts.0.w1.scale"]
            )
        )
        assert torch.equal(unpack_checkpoint(reader.get_tensor("layers.0.ffn.experts.0.w1.weight_packed")), q)
        assert torch.equal(reader.get_tensor("layers.0.ffn.experts.0.w1.weight_scale"), s)
        assert reader.get_tensor("layers.0.ffn.experts.0.w1.weight_shape").tolist() == [32, 32]
        dense = reader.get_tensor("layers.0.attn.wkv.weight")
        assert dense.dtype == torch.bfloat16
        assert dense[0, 0] == 1 and dense[31, 33] == 2 and dense[32, 31] == 4 and dense[-1, -1] == 8
        embedding = reader.get_tensor("layers.1.engram.embed.weight")
        assert torch.equal(embedding[:, :32], torch.ones(5, 32, dtype=torch.bfloat16))
        assert (embedding[:, 32:] == 2).all()
        assert reader.get_tensor("layers.0.attn.attn_sink").dtype == torch.float32
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"
    assert "quantization_config" not in config["text_config"]
    assert config["ascend_weight_format"]["signed_scale"]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert not any(key.endswith(".scale") for key in index["weight_map"])
    # A scale-only output shard is a valid empty safetensors file.
    with safe_open(output / "model-00002.safetensors", framework="pt") as reader:
        assert not list(reader.keys())


def test_resume_does_not_publish_partial_model_and_detects_output_corruption(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    first = converter.convert(source, output, max_shards=1)
    assert not first["complete"] and not (output / "config.json").exists()
    stat = (output / "model-00001.safetensors").stat()
    assert converter.convert(source, output)["complete"]
    assert (output / "model-00001.safetensors").stat().st_mtime_ns == stat.st_mtime_ns
    with (output / "model-00001.safetensors").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xaa")
    with pytest.raises(ValueError, match="checksum mismatch"):
        converter.convert(source, output)


def test_missing_scale_shard_is_not_converted_or_published(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    (source / "model-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        converter.convert(source, output)
    result = converter.convert(source, output, allow_incomplete=True)
    assert not result["shards"] and not result["complete"]
    assert not (output / "model.safetensors.index.json").exists()


def test_truncated_source_and_changed_config_fail_closed(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source, cross_shard=False)
    converter.convert(source, output)
    (source / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        converter.convert(source, output)
    with (source / "model-00001.safetensors").open("r+b") as stream:
        stream.truncate(100)
    with pytest.raises(ValueError, match="header"):
        converter.inventory(source)
