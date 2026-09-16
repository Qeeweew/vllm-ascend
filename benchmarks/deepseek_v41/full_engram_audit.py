# SPDX-License-Identifier: Apache-2.0
"""Bounded real-table audit shared by full-model and host-factory diagnostics."""

import hashlib
import json
import resource
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


def process_memory():
    result = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    for filename in ("status", "smaps_rollup", "io"):
        path = Path("/proc/self") / filename
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) > 1 and parts[0].rstrip(":") in {
                "VmRSS",
                "RssAnon",
                "VmLck",
                "VmPin",
                "Rss",
                "Pss",
                "Anonymous",
                "read_bytes",
                "write_bytes",
            }:
                result[f"{filename}.{parts[0].rstrip(':')}"] = int(parts[1]) * (1024 if parts[-1] == "kB" else 1)
    result["major_faults"] = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    return result


def digest(tensor):
    import torch

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def audit_loading(callback=None):
    """Observe actual slice requests and checked register/unregister calls."""
    import vllm_ascend.ops.engram_offload as offload
    import vllm_ascend.ops.engram_pinned_host as pinned

    records = {"before": process_memory(), "reads": [], "registration_events": []}
    original_open = offload.safe_open

    class Slice:
        def __init__(self, underlying, name):
            self.underlying, self.name = underlying, name

        def get_shape(self):
            return self.underlying.get_shape()

        def get_dtype(self):
            return self.underlying.get_dtype()

        def __getitem__(self, selection):
            if not isinstance(selection, slice) or selection.step not in (None, 1):
                raise AssertionError("Unexpected noncontiguous full-table read")
            records["reads"].append({"name": self.name, "start": selection.start, "stop": selection.stop})
            if callback is not None and len(records["reads"]) % 128 == 0:
                callback(
                    {
                        "event": "factory_copy_progress",
                        "slices": len(records["reads"]),
                        "table": self.name,
                        "source_row": selection.start,
                    }
                )
            return self.underlying[selection]

    class Reader:
        def __init__(self, *args, **kwargs):
            self.context = original_open(*args, **kwargs)

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_slice(self, name):
            return Slice(self.reader.get_slice(name), name)

    class TimedAPI(pinned._HostMemoryAPI):
        def register(self, pointer, size):
            start = time.monotonic()
            super().register(pointer, size)
            records["registration_events"].append(
                {"event": "registered", "bytes": size, "seconds": time.monotonic() - start, "memory": process_memory()}
            )

        def unregister(self, pointer):
            start = time.monotonic()
            super().unregister(pointer)
            records["registration_events"].append({"event": "unregistered", "seconds": time.monotonic() - start})

    start = time.monotonic()
    with patch.object(offload, "safe_open", Reader), patch.object(pinned, "_HostMemoryAPI", TimedAPI):
        yield records
    records["after"] = process_memory()
    records["load_seconds"] = time.monotonic() - start


def inspect_tables(runtime, source, converted, load_records):
    """18 rows/rank: source FP8 oracle == converted BF16 == loaded local row."""
    import torch
    from probe_engram_full_capacity import sample_placement
    from safetensors import safe_open

    source, converted = Path(source), Path(converted)
    source_index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    output_index = json.loads((converted / "model.safetensors.index.json").read_text())["weight_map"]
    reports, samples = [], []
    for layer_id, shard in zip(runtime.history.hasher.layout.layer_ids, runtime.offload.shards, strict=True):
        name = f"layers.{layer_id}.engram.embed.weight"
        scale_name = name.removesuffix("weight") + "scale"
        owner = shard._pinned_owner
        if owner is None or not owner._mapping.registered or not shard.weight.is_pinned():
            raise AssertionError("Real table is not a live registered NUMA owner")
        expected_reads = [
            {"name": name, "start": row, "stop": min(row + 65536, end)}
            for begin, end in shard.head_ranges
            for row in range(begin, end, 65536)
        ]
        reads = [entry for entry in load_records["reads"] if entry["name"] == name]
        if reads != expected_reads:
            raise AssertionError("Actual loader slices do not exactly partition owned head ranges")
        report = {
            "layer": layer_id,
            "heads": list(shard.head_indices),
            "ranges": list(shard.head_ranges),
            "shape": list(shard.weight.shape),
            "dtype": str(shard.weight.dtype),
            "pinned": True,
            "numa_node": owner.numa_node,
            "registered_bytes": len(owner._mapping),
            "payload_bytes": shard.weight.numel() * shard.weight.element_size(),
            "placement": sample_placement(shard.weight, owner.numa_node),
            "logical_reads_exact": True,
            "logical_slice_count": len(reads),
            "maximum_slice_bytes": max(entry["stop"] - entry["start"] for entry in reads) * 512,
            "samples": [],
        }
        layer_samples = []
        cursor = 0
        with ExitStack() as stack:
            source_reader = stack.enter_context(safe_open(source / source_index[name], framework="pt", device="cpu"))
            scale_reader = stack.enter_context(
                safe_open(source / source_index[scale_name], framework="pt", device="cpu")
            )
            output_reader = stack.enter_context(safe_open(converted / output_index[name], framework="pt", device="cpu"))
            raw, scales, output = (
                source_reader.get_slice(name),
                scale_reader.get_slice(scale_name),
                output_reader.get_slice(name),
            )
            if raw.get_dtype() != "F8_E4M3" or scales.get_shape() != [raw.get_shape()[0], 8]:
                raise AssertionError("Engram source scale layout must be one row by 32 columns")
            for head, (begin, end) in zip(shard.head_indices, shard.head_ranges, strict=True):
                head_samples = []
                for row in (begin, begin + (end - begin) // 2, end - 1):
                    original = raw[row : row + 1].float()
                    scale = scales[row : row + 1].float().repeat_interleave(32, dim=1)
                    oracle = (original * scale).bfloat16().squeeze(0)
                    converted_row = output[row : row + 1].squeeze(0)
                    local = cursor + row - begin
                    loaded = shard.weight[local].clone()
                    if not torch.equal(oracle, converted_row) or not torch.equal(oracle, loaded):
                        raise AssertionError(f"Real row oracle mismatch at layer={layer_id}, head={head}, row={row}")
                    report["samples"].append(
                        {
                            "head": head,
                            "global_row": row,
                            "local_row": local,
                            "exact": True,
                            "bf16_sha256": digest(loaded),
                        }
                    )
                    head_samples.append((row, loaded))
                layer_samples.append(head_samples)
                cursor += end - begin
        reports.append(report)
        samples.append(layer_samples)
    return reports, samples


def replay_sample_rows(runtime, samples, *, steps=20):
    """Real-table DMA and changed-input replay, including DEAD/padded rows."""
    import torch

    from vllm_ascend.ops.engram_offload import DEAD_HASH_ID

    manager = runtime.offload
    if manager.max_tokens != 4:
        raise ValueError("The independent host-factory replay probe requires a four-token staging bucket")
    graph = torch.npu.NPUGraph()
    pointers = [row.data_ptr() for row in manager.device_rows]
    with torch.npu.graph(graph):
        outputs = tuple(row.clone() for row in manager.device_rows)
    for step in range(steps):
        hashes, expected = [], []
        for shard, layer_samples in zip(manager.shards, samples, strict=True):
            ids = torch.full((3, 24), DEAD_HASH_ID, dtype=torch.int64, device="cpu")
            want = torch.zeros((4, 3, 256), dtype=torch.bfloat16, device="cpu")
            for local_head, (head, head_samples) in enumerate(zip(shard.head_indices, layer_samples, strict=True)):
                for token in range(3):
                    row, value = head_samples[(step + token) % 3]
                    if token == 2 and step % 2:
                        continue
                    ids[token, head] = row
                    want[token, local_head] = value
            hashes.append(ids)
            expected.append(want)
        manager.prepare(hashes, 4)
        manager.wait_ready()
        graph.replay()
        actual = tuple(output.clone() for output in outputs)
        manager.mark_consumed()
        for output, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(output.cpu(), want, rtol=0, atol=0)
        assert [row.data_ptr() for row in manager.device_rows] == pointers
    return {
        "graph_replays": steps,
        "exact": True,
        "stable_pointers": pointers,
        "device_allocated_bytes": torch.npu.memory_allocated(),
        "device_reserved_bytes": torch.npu.memory_reserved(),
    }
