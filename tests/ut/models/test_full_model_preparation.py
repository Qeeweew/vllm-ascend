# SPDX-License-Identifier: Apache-2.0
"""CPU admission and real-format tiny row oracles; never launch workers or NPU."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks/deepseek_v41"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(BENCHMARKS), *sys.path]):
        spec.loader.exec_module(module)
    return module


preflight = load_script("preflight_full_model")
audit = load_script("full_engram_audit")
factory = load_script("validate_real_engram_factory")


def test_tp8_capacity_excludes_real_host_vision_and_draft_and_preserves_replication():
    assert preflight.resident_class("layers.1.engram.embed.weight") == ("host_engram", 0)
    assert preflight.resident_class("mtp.0.ffn.experts.0.w1.weight_packed")[1] == 0
    assert preflight.resident_class("vision.blocks.0.attn.wo.weight")[1] == 0
    assert preflight.resident_class("layers.0.ffn.experts.0.w1.weight_packed") == ("moe_packed", 8)
    assert preflight.resident_class("layers.0.ffn.experts.0.w1.weight_scale") == ("moe_scales", 8)
    assert preflight.resident_class("layers.0.attn.wq_b.weight")[1] == 8
    assert preflight.resident_class("layers.2.attn.indexer.wq_b.weight")[1] == 1
    assert preflight.resident_class("layers.1.engram.wkv.weight")[1] == 1
    elements = 40 * 384 * 3 * 2304 * 5120
    assert (elements // 2 + elements // 32 * 2) // 8 == 38220595200


def test_independent_factory_does_not_inherit_full_model_hbm_floor_or_bypass_conversion():
    state = {
        "blockers": ["Rank 7 free HBM below estimated static weights + load/KV/reserve"],
        "host_pinned_bytes": 393227699200,
        "host_admission_bytes": 393227699200 + 256 * 1024**3,
        "hbm_memory": {str(rank): {"free_bytes": 20 * 1024**3} for rank in range(8)},
    }
    assert factory.factory_admission(state)["ready"]
    state["blockers"].append("Converted manifest is not complete with 48 matching output headers")
    assert not factory.factory_admission(state)["ready"]
    state["blockers"].pop()
    state["hbm_memory"]["7"]["free_bytes"] = 1024**3
    assert not factory.factory_admission(state)["ready"]


def tiny_real_rows(tmp_path, monkeypatch):
    source, converted = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    converted.mkdir()
    name, scale_name = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    raw = (torch.arange(12 * 256).reshape(12, 256) % 17 - 8).to(torch.float8_e4m3fn)
    scales = torch.arange(96).reshape(12, 8).float() / 256 + 1
    expected = (raw.float() * scales.repeat_interleave(32, dim=1)).bfloat16()
    save_file({name: raw, scale_name: scales}, str(source / "source.safetensors"))
    save_file({name: expected}, str(converted / "converted.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "source.safetensors", scale_name: "source.safetensors"}})
    )
    (converted / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "converted.safetensors"}}))

    class Weight:
        shape, dtype = expected.shape, expected.dtype

        def is_pinned(self):
            return True

        def numel(self):
            return expected.numel()

        def element_size(self):
            return expected.element_size()

        def __getitem__(self, index):
            return expected[index]

    class Mapping(bytearray):
        registered = True

    owner = SimpleNamespace(_mapping=Mapping(expected.numel() * 2), numa_node=6)
    shard = SimpleNamespace(
        weight=Weight(), _pinned_owner=owner, head_indices=(0, 1, 2), head_ranges=((0, 4), (4, 8), (8, 12))
    )
    runtime = SimpleNamespace(
        history=SimpleNamespace(hasher=SimpleNamespace(layout=SimpleNamespace(layer_ids=(1,)))),
        offload=SimpleNamespace(shards=[shard]),
    )
    records = {"reads": [{"name": name, "start": begin, "stop": end} for begin, end in shard.head_ranges]}
    monkeypatch.setitem(
        sys.modules, "probe_engram_full_capacity", SimpleNamespace(sample_placement=lambda *args: {"sampled": True})
    )
    return runtime, source, converted, records, expected


def test_source_conversion_loaded_rows_all_use_group32_and_exact_local_boundaries(tmp_path, monkeypatch):
    runtime, source, converted, records, _ = tiny_real_rows(tmp_path, monkeypatch)
    result, samples = audit.inspect_tables(runtime, source, converted, records)
    assert result[0]["logical_reads_exact"] and len(result[0]["samples"]) == 9
    assert [item["global_row"] for item in result[0]["samples"]] == [0, 2, 3, 4, 6, 7, 8, 10, 11]
    assert len(samples[0]) == 3


def test_wrong_local_row_or_repeated_loader_ranges_cannot_pass(tmp_path, monkeypatch):
    runtime, source, converted, records, expected = tiny_real_rows(tmp_path, monkeypatch)
    original = list(records["reads"])
    records["reads"][1] = records["reads"][0]
    with pytest.raises(AssertionError, match="exactly partition"):
        audit.inspect_tables(runtime, source, converted, records)
    records["reads"] = original
    expected[0, 0] += 1
    with pytest.raises(AssertionError, match="oracle mismatch"):
        audit.inspect_tables(runtime, source, converted, records)


@pytest.mark.parametrize(
    "corrupt", ["status", "graph", "synthetic_weights", "native_decode", "source", "converted", "prompts"]
)
def test_full_model_reference_rejects_failed_or_incompatible_runs(corrupt):
    from copy import deepcopy

    driver = load_script("validate_full_model_tp8")
    baseline = {
        "status": "passed",
        "graph": False,
        "synthetic_weights": False,
        "native_decode": False,
        "prompts": [[101]],
        "preflight": {"source_fingerprints": {"shard": "identity"}, "converted": "/real/converted"},
    }
    current = deepcopy(baseline)
    current["graph"] = True
    driver.validate_reference(baseline, current, [[101]])
    invalid = deepcopy(baseline)
    if corrupt == "source":
        invalid["preflight"]["source_fingerprints"] = {"shard": "changed"}
    elif corrupt == "converted":
        invalid["preflight"]["converted"] = "/other/converted"
    else:
        invalid[corrupt] = "failed" if corrupt == "status" else True
    with pytest.raises(ValueError, match="passed eager full-model"):
        driver.validate_reference(invalid, current, [[101]])


def test_graph_run_requires_reference_before_preflight(tmp_path, monkeypatch, capsys):
    driver = load_script("validate_full_model_tp8")
    monkeypatch.setattr(driver, "build_preflight", lambda *_: pytest.fail("Preflight must not start"))
    monkeypatch.setattr(sys, "argv", ["driver", "--run", "--graph", "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    assert "requires --reference" in capsys.readouterr().err


@pytest.mark.parametrize("run", [False, True])
def test_graph_preparation_allows_no_reference_but_actual_run_checks_reference_before_admission(
    tmp_path, monkeypatch, capsys, run
):
    driver = load_script("validate_full_model_tp8")
    state = {"status": "not_ready", "source_fingerprints": {"shard": "identity"}, "converted": "/real/converted"}
    monkeypatch.setattr(driver, "build_preflight", lambda *_: state)
    output = tmp_path / "result.json"
    arguments = ["driver", "--graph", "--output", str(output)]
    if run:
        reference = tmp_path / "eager.json"
        reference.write_text(json.dumps({"status": "failed"}))
        arguments += ["--run", "--reference", str(reference)]
    monkeypatch.setattr(sys, "argv", arguments)
    if run:
        with pytest.raises(SystemExit) as error:
            driver.main()
        assert error.value.code == 2
        assert "Invalid eager reference" in capsys.readouterr().err
        assert not output.exists()
    else:
        assert driver.main() == 0
        assert json.loads(output.read_text())["status"] == "prepared_only"


def test_matching_eager_reference_reaches_capacity_admission_without_npu(tmp_path, monkeypatch):
    driver = load_script("validate_full_model_tp8")
    state = {"status": "not_ready", "source_fingerprints": {"shard": "identity"}, "converted": "/real/converted"}
    monkeypatch.setattr(driver, "build_preflight", lambda *_: state)
    reference = tmp_path / "eager.json"
    reference.write_text(
        json.dumps(
            {
                "status": "passed",
                "graph": False,
                "synthetic_weights": False,
                "native_decode": False,
                "prompts": [list(range(100, 140)), [0, 129264, 101], list(range(100, 229)), list(range(100, 484))],
                "preflight": state,
            }
        )
    )
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys, "argv", ["driver", "--run", "--graph", "--reference", str(reference), "--output", str(output)]
    )
    assert driver.main() == 1
    assert json.loads(output.read_text())["status"] == "blocked_before_launch"
