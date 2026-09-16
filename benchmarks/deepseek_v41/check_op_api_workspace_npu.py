# SPDX-License-Identifier: Apache-2.0
"""Native scratch-lifetime acceptance; run each task-queue mode in a fresh process.

This is a functional stress test, not a latency benchmark or the DSpark proposer
acceptance gate. No NPU is initialized by --help. See OP_API_WORKSPACE_TEST.md.
"""

import argparse
import ctypes as ct
import hashlib
import importlib
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu


def runtime_mode():
    library = Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
    query = ct.CDLL(str(library))._Z23OpApiGetTaskQueueEnablev
    query.argtypes = []
    query.restype = ct.c_uint32
    return query()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class NativeScatterProbe:
    """Measure native scratch and release its executor through a real launch.

    The stress loop below uses the production Torch binding, not this ctypes
    launch. All descriptors and scratch here remain owned through synchronize.
    """

    def __init__(self, path):
        self.library = ct.CDLL(str(path))
        self.create = self.library.aclCreateTensor
        self.create.restype = ct.c_void_p
        self.create.argtypes = [
            ct.POINTER(ct.c_int64),
            ct.c_uint64,
            ct.c_int,
            ct.POINTER(ct.c_int64),
            ct.c_int64,
            ct.c_int,
            ct.POINTER(ct.c_int64),
            ct.c_uint64,
            ct.c_void_p,
        ]
        self.destroy = self.library.aclDestroyTensor
        self.destroy.argtypes = [ct.c_void_p]
        self.create_array = self.library.aclCreateIntArray
        self.create_array.argtypes = [ct.POINTER(ct.c_int64), ct.c_uint64]
        self.create_array.restype = ct.c_void_p
        self.destroy_array = self.library.aclDestroyIntArray
        self.destroy_array.argtypes = [ct.c_void_p]
        self.query = self.library.aclnnScatterNdUpdateSkGetWorkspaceSize
        self.query.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_void_p)]
        self.query.restype = ct.c_int
        self.launch = self.library.aclnnScatterNdUpdateSk
        self.launch.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
        self.launch.restype = ct.c_int

    def descriptor(self, tensor):
        # aclDataType: INT64=9, BF16=27; aclFormat: ND=2.
        dtype = {torch.int64: 9, torch.bfloat16: 27}[tensor.dtype]
        shape = (ct.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (ct.c_int64 * tensor.ndim)(*tensor.stride())
        storage = (ct.c_int64 * 1)(tensor.untyped_storage().nbytes() // tensor.element_size())
        result = self.create(
            shape,
            tensor.ndim,
            dtype,
            stride,
            tensor.storage_offset(),
            2,
            storage,
            1,
            tensor.untyped_storage().data_ptr(),
        )
        assert result, "aclCreateTensor failed"
        return result

    def measure(self, cache, indices, values):
        descriptors = [self.descriptor(tensor) for tensor in (cache, indices, values)]
        stride = (ct.c_int64 * cache.ndim)(*cache.stride())
        strides = self.create_array(stride, cache.ndim)
        assert strides, "aclCreateIntArray failed"
        workspace_size, executor = ct.c_uint64(), ct.c_void_p()
        try:
            status = self.query(*descriptors, strides, ct.byref(workspace_size), ct.byref(executor))
            assert status == 0, f"GetWorkspaceSize returned {status}"
            assert workspace_size.value > 0, "Test must exercise actual nonzero native scratch"
            scratch = torch.empty(workspace_size.value, dtype=torch.uint8, device=cache.device)
            torch.npu.synchronize()
            status = self.launch(
                scratch.data_ptr(), workspace_size.value, executor, torch.npu.current_stream().npu_stream
            )
            assert status == 0, f"Native scatter launch returned {status}"
            torch.npu.synchronize()
            return workspace_size.value
        finally:
            for descriptor in descriptors:
                self.destroy(descriptor)
            self.destroy_array(strides)


def make_builder(cache, device):
    # Import after custom OPP bootstrap, as in the actual operator runtime.
    from vllm_ascend.attention.dsa_v41 import AscendV41CacheMetadataBuilder
    from vllm_ascend.core.kv_cache_interface import AscendV41SWACacheSpec

    spec = AscendV41SWACacheSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        head_size_v=0,
        sliding_window=128,
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256, max_num_seqs=1),
        model_config=SimpleNamespace(max_model_len=256, hf_config=SimpleNamespace(num_attention_heads=64)),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        compilation_config=SimpleNamespace(static_forward_context={"cache": SimpleNamespace(kv_cache=cache)}),
    )
    builder = AscendV41CacheMetadataBuilder(spec, ["cache"], config, device)
    builder.enable_dspark_device_metadata(5)
    return builder


def make_case(prefix, device):
    length = prefix + 5
    table = torch.arange(7, -1, -1, dtype=torch.int32)[None]
    positions = torch.arange(length, dtype=torch.int64)
    slots = table[0, positions // 32].long() * 32 + positions % 32
    indices = torch.stack((slots // 32, slots % 32), dim=-1)
    values = ((positions[:, None] * 7 + torch.arange(512)[None]) % 31 - 15).float().div_(64).bfloat16()
    expected = torch.zeros((8, 32, 1, 512), dtype=torch.bfloat16)
    expected.view(256, 512)[slots] = values
    start = max(prefix - 128, 0)
    # Q=0 and sink=0: each visible key and the sink have logit zero.
    attention = values[start:].float().sum(0) / (length - start + 1)
    common = SimpleNamespace(
        positions=positions[-5:].to(device),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([length], dtype=torch.int32, device=device),
        block_table_tensor=table.to(device),
        slot_mapping=torch.empty(5, dtype=torch.int32, device=device),
        num_reqs=1,
        num_actual_tokens=5,
        max_query_len=5,
        max_seq_len=length,
        causal=False,
    )
    return {
        "prefix": prefix,
        "length": length,
        "span": length - start,
        "common": common,
        "indices": indices.to(device),
        "values": values[:, None, :].to(device),
        "expected_cache": expected,
        "expected_attention": attention.expand(5, 8, 512).contiguous(),
    }


def stress(args, report):
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    from vllm_ascend.attention.dsa_v41 import make_v41_attention_metadata
    from vllm_ascend.ops.dsa_v41 import AscendDSAV41Ops

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    device = torch.device("npu", 0)
    library = Path(__file__).resolve().parents[2] / (
        "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
    )
    report.update(
        {
            "extension": str(Path(extension.__file__).resolve()),
            "extension_sha256": sha256(Path(extension.__file__)),
            "opapi_library": str(library.resolve()),
            "opapi_sha256": sha256(library),
            "torch_npu_version": torch_npu.__version__,
            "torch_npu_commit": torch_npu.version.git_version,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "actual_task_queue_mode": runtime_mode(),
            "acl_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            "iterations": args.iterations,
            "burst": args.burst,
            "graph_validated": False,
            "real_proposer_validated": False,
            "performance_measurement": False,
        }
    )
    assert report["actual_task_queue_mode"] == args.queue_mode, "Actual queue mode differs (blocking override?)"
    caches = [torch.zeros((8, 32, 1, 512), dtype=torch.bfloat16, device=device) for _ in range(3)]
    builder = make_builder(caches[0], device)
    cases = [make_case(prefix, device) for prefix in (9, 33, 129)]
    probe = NativeScatterProbe(library)
    report["workspace_bytes"] = {}
    for case in cases:
        for part, rows in (("context", slice(None, case["prefix"])), ("query", slice(case["prefix"], None))):
            size = probe.measure(caches[0], case["indices"][rows], case["values"][rows])
            report["workspace_bytes"][f"{case['prefix']}_{part}"] = size
    report["status"] = "running"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    query = torch.zeros((5, 8, 512), dtype=torch.bfloat16, device=device)
    sinks = torch.zeros(8, dtype=torch.float32, device=device)
    attention = AscendDSAV41Ops(0)
    retained = []
    checked = 0
    maximum_attention_error = 0.0
    # Keep consecutive calls asynchronous. Every snapshot is checked after a
    # burst, so a later valid cache update cannot hide an earlier corruption.
    for step in range(args.iterations):
        case = cases[step % len(cases)]
        cache_snapshots = []
        for cache in caches:
            cache.zero_()
            for part, rows in (("context", slice(None, case["prefix"])), ("query", slice(case["prefix"], None))):
                torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, case["indices"][rows], case["values"][rows])
                size = report["workspace_bytes"][f"{case['prefix']}_{part}"]
                pressure = torch.empty(size, dtype=torch.uint8, device=device)
                pressure.fill_(165)
                del pressure
            cache_snapshots.append(cache.clone())
        metadata = builder.build_for_drafting(case["common"], draft_index=1)
        outputs = [attention.forward(query, cache, sinks, make_v41_attention_metadata(metadata))[0] for cache in caches]
        retained.append((case, cache_snapshots, metadata.draft_swa_lengths.clone(), outputs))
        if len(retained) < args.burst and step + 1 < args.iterations:
            continue
        torch.npu.synchronize()
        for saved_case, snapshots, spans, outputs in retained:
            for snapshot in snapshots:
                assert torch.equal(snapshot.cpu(), saved_case["expected_cache"]), "Scatter cache differs bitwise"
            assert spans.cpu().tolist() == [[saved_case["span"]]] * 5
            for output in outputs:
                actual = output.cpu().float()
                expected = saved_case["expected_attention"]
                torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
                maximum_attention_error = max(maximum_attention_error, (actual - expected).abs().max().item())
            checked += 1
        retained.clear()
        allocated = torch.npu.memory_allocated()
        report.setdefault("allocated_bytes_after_bursts", []).append(allocated)
        report.update({"checked_iterations": checked, "maximum_attention_abs_error": maximum_attention_error})
        # Completed queue slots may retain callback copies. They must not keep
        # native scratch alive after submission and the burst synchronization.
        assert allocated < 256 * 1024**2, "Completed command handlers retain native scratch"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report.update(
        {
            "status": "passed",
            "cache_bitwise_checks": checked * 3,
            "attention_checks": checked * 3,
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-mode", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--burst", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or not 1 <= args.burst <= 24:
        parser.error("iterations must be positive and burst must be in [1,24]")
    if args.output.exists():
        parser.error("choose a new output path")
    if os.environ.get("TASK_QUEUE_ENABLE") != str(args.queue_mode):
        parser.error("set TASK_QUEUE_ENABLE before launching this fresh Python process")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "requested_task_queue_mode": args.queue_mode}
    try:
        stress(args, report)
    except Exception:
        report.update({"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
