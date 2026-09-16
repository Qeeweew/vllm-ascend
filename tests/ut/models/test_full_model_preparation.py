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


def test_dspark_capacity_includes_sharded_experts_and_replicated_markov_weights():
    assert preflight.resident_class("mtp.0.ffn.experts.0.w1.weight_packed", include_dspark=True) == (
        "draft_moe_packed",
        8,
    )
    assert preflight.resident_class("mtp.0.attn.wq_b.weight", include_dspark=True) == ("draft_tp_attention", 8)
    assert preflight.resident_class("mtp.2.markov_head.head.weight", include_dspark=True)[1] == 1
    assert preflight.resident_class("mtp.0.main_proj.weight", include_dspark=True)[1] == 1
    assert preflight.resident_class("embed.weight", include_dspark=True) == ("tp_embedding_head", 8)


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


def test_factory_cannot_publish_pass_until_every_owner_release_and_worker_exit():
    report = {"status": "resident_checks_passed_cleanup_pending", "events": []}
    assert factory.final_status(report) == "failed_cleanup"
    for rank in range(8):
        report["events"].append(
            {
                "event": "released",
                "rank": rank,
                "success": True,
                "cleanup_errors": [],
                "registration_events": [{"event": "unregistered"}, {"event": "unregistered"}],
            }
        )
        assert factory.final_status(report) == "failed_cleanup"
        report["events"].append({"event": "worker_exit", "pid": 100 + rank, "exitcode": 0})
        assert factory.final_status(report) == ("passed" if rank == 7 else "failed_cleanup")
    report["events"][-1]["exitcode"] = 1
    assert factory.final_status(report) == "failed_cleanup"
    report["events"][-1]["exitcode"] = 0
    report["events"][-2]["registration_events"].pop()
    assert factory.final_status(report) == "failed_cleanup"
    report["status"] = "failed"
    assert factory.final_status(report) == "failed"


@pytest.fixture
def dispatch_audit(monkeypatch):
    """Exercise observer logic with no NPU graph or operator execution."""
    import vllm.forward_context as forward_context

    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", SimpleNamespace(NPUWorker=object))
    worker_module = load_script("full_model_worker")
    capturing = [False]
    context = SimpleNamespace(
        batch_descriptor=forward_context.BatchDescriptor(num_tokens=4, num_reqs=4),
        cudagraph_runtime_mode="FULL",
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: capturing[0])
    failure = RuntimeError("original submission failed")
    sentinels = {}
    for name in ("v41_rope", "v41_main_cache_store", "v41_index_cache_store", "v41_moe_router", "npu_w4a16_moe"):
        sentinels[name] = object()

        def native(value, *, keyword=None, _sentinel=sentinels[name]):
            if value == "fail":
                raise failure
            return value, keyword, _sentinel

        monkeypatch.setattr(torch.ops._C_ascend, name, native, raising=False)

    class GraphWrapper:
        def __init__(self, runnable):
            self.runnable = runnable
            self.runtime_mode = "FULL"
            self.concrete_aclgraph_entries = {}
            self.fail_replay = False

        def __call__(self, *args, **kwargs):
            if context.cudagraph_runtime_mode != self.runtime_mode:
                return self.runnable(*args, **kwargs)
            entry = self.concrete_aclgraph_entries.setdefault(context.batch_descriptor, SimpleNamespace(aclgraph=None))
            if entry.aclgraph is not None:
                if self.fail_replay:
                    raise failure
                return entry.output
            capturing[0] = True
            try:
                output = self.runnable(*args, **kwargs)
            finally:
                capturing[0] = False
            entry.aclgraph, entry.output = object(), output
            return output

    monkeypatch.setitem(sys.modules, "vllm_ascend.compilation.acl_graph", SimpleNamespace(ACLGraphWrapper=GraphWrapper))
    worker = worker_module.V41FullModelWorker()
    worker.model_runner = SimpleNamespace(drafter=None)
    worker._observe_native_dispatch()

    def snapshot():
        return {
            "operator_dispatch": dict(worker.full_operator_dispatch),
            "graph_dispatch": worker._inspect_graph_dispatch(),
        }

    return SimpleNamespace(
        worker=worker,
        wrapper=GraphWrapper,
        context=context,
        failure=failure,
        sentinels=sentinels,
        snapshot=snapshot,
        smoke=load_script("check_full_text_tp8"),
    )


def test_graph_family_audit_identifies_actual_draft_wrappers(dispatch_audit):
    audit = dispatch_audit
    context = audit.wrapper(lambda value: value)
    query = audit.wrapper(lambda value: value)
    target = audit.wrapper(lambda value: value)
    audit.worker.model_runner.drafter = SimpleNamespace(_v41_graph=SimpleNamespace(context=context, query=query))
    for wrapper in (context, query, target):
        wrapper(1)
    audit.worker._start_request_dispatch_audit()
    context(2)
    target(2)
    records = {row["family"]: row for row in audit.worker._inspect_graph_dispatch()}
    assert records["draft_context"]["request_replays"] == 1
    assert records["target"]["request_replays"] == 1
    assert records["draft_query"]["request_replays"] == 0


def test_draft_native_dispatch_cannot_stand_in_for_target(dispatch_audit):
    audit = dispatch_audit
    query = audit.wrapper(lambda value: torch.ops._C_ascend.v41_rope(value))
    audit.worker.model_runner.drafter = SimpleNamespace(_v41_graph=SimpleNamespace(context=object(), query=query))
    query(1)
    audit.worker._start_request_dispatch_audit()
    query(2)
    with pytest.raises(AssertionError, match="no request replay"):
        audit.smoke.check_native_dispatch(audit.snapshot(), enabled=True, name="v41_rope", graph=True)


def test_dispatch_observer_preserves_native_results_and_exceptions(dispatch_audit):
    audit = dispatch_audit
    for name, sentinel in audit.sentinels.items():
        native = getattr(torch.ops._C_ascend, name)
        assert native(123, keyword=456) == (123, 456, sentinel)
        label = "w4_native" if name == "npu_w4a16_moe" else name
        assert audit.worker.full_operator_dispatch[f"{label}_eager"] == 1
        with pytest.raises(RuntimeError) as error:
            native("fail")
        assert error.value is audit.failure
        assert audit.worker.full_operator_dispatch[f"{label}_eager"] == 1
    assert not audit.worker._inspect_graph_dispatch()


def test_dispatch_audit_requires_same_wrapper_entry_and_graph(dispatch_audit):
    audit = dispatch_audit
    native = audit.wrapper(lambda: torch.ops._C_ascend.npu_w4a16_moe(123))
    unrelated = audit.wrapper(lambda: "fallback")
    result = native()
    assert unrelated() == "fallback"
    # Warmup replay must not count as a request, but capture provenance survives.
    assert native() is result
    audit.worker._start_request_dispatch_audit()
    assert all(row["request_replays"] == 0 for row in audit.worker._inspect_graph_dispatch())
    assert unrelated() == "fallback"
    check = audit.smoke.check_native_dispatch
    with pytest.raises(AssertionError, match="no request replay"):
        check(audit.snapshot(), enabled=True, name="w4_native", graph=True)
    # Both wrappers have the SAME descriptor, including T. Only this replay counts.
    assert native() is result
    check(audit.snapshot(), enabled=True, name="w4_native", graph=True)
    rows = audit.worker._inspect_graph_dispatch()
    assert len({row["wrapper_id"] for row in rows}) == 2
    assert rows[0]["descriptor"] == rows[1]["descriptor"]
    assert rows[0]["captured_native_ops"] == {"w4_native": 1}
    assert rows[1]["captured_native_ops"] == {}
    json.dumps(audit.snapshot())
    # Replacing the graph within the same entry must not inherit old evidence.
    audit.worker._start_request_dispatch_audit()
    entry = native.concrete_aclgraph_entries[audit.context.batch_descriptor]
    entry.aclgraph = object()
    assert native() is result
    with pytest.raises(AssertionError, match="no request replay"):
        check(audit.snapshot(), enabled=True, name="w4_native", graph=True)


def test_dispatch_audit_separates_descriptors_and_failed_calls(dispatch_audit):
    audit = dispatch_audit
    graph = audit.wrapper(lambda value: torch.ops._C_ascend.v41_rope(value))
    with pytest.raises(RuntimeError) as error:
        graph("fail")
    assert error.value is audit.failure
    assert audit.worker._full_capture_stack == []
    assert not audit.worker._inspect_graph_dispatch()
    first = graph(123)
    first_descriptor = audit.context.batch_descriptor
    audit.context.batch_descriptor = type(first_descriptor)(num_tokens=4, num_reqs=1)
    # Same token bucket, different batch descriptor and concrete entry.
    graph.runnable = lambda value: value
    assert graph(456) == 456
    audit.worker._start_request_dispatch_audit()
    assert graph(789) == 456
    with pytest.raises(AssertionError, match="no request replay"):
        audit.smoke.check_native_dispatch(audit.snapshot(), enabled=True, name="v41_rope", graph=True)
    audit.context.batch_descriptor = first_descriptor
    graph.fail_replay = True
    before = audit.worker.full_graph_replays
    with pytest.raises(RuntimeError) as error:
        graph(123)
    assert error.value is audit.failure
    assert audit.worker.full_graph_replays == before
    assert all(row["request_replays"] == 0 for row in audit.worker._inspect_graph_dispatch()[:1])
    graph.fail_replay = False
    assert graph(123) is first
    audit.smoke.check_native_dispatch(audit.snapshot(), enabled=True, name="v41_rope", graph=True)


def test_dispatch_audit_discards_partial_capture_and_preserves_eager_path(dispatch_audit):
    audit = dispatch_audit

    def partial_capture():
        torch.ops._C_ascend.v41_moe_router(123)
        raise audit.failure

    graph = audit.wrapper(partial_capture)
    with pytest.raises(RuntimeError) as error:
        graph()
    assert error.value is audit.failure
    assert audit.worker.full_operator_dispatch["v41_moe_router_capture"] == 1
    assert not audit.worker._inspect_graph_dispatch()
    # Successful eager submission is reported without inventing a graph.
    audit.context.cudagraph_runtime_mode = "NONE"
    graph.runnable = lambda: torch.ops._C_ascend.v41_moe_router(456)
    assert graph() == (456, None, audit.sentinels["v41_moe_router"])
    audit.smoke.check_native_dispatch(audit.snapshot(), enabled=True, name="v41_moe_router", graph=False)
    with pytest.raises(AssertionError, match="no request replay"):
        audit.smoke.check_native_dispatch(audit.snapshot(), enabled=True, name="v41_moe_router", graph=True)
    with pytest.raises(AssertionError):
        audit.smoke.check_native_dispatch(audit.snapshot(), enabled=False, name="v41_moe_router", graph=False)
