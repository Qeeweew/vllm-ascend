# SPDX-License-Identifier: Apache-2.0
"""Read-only header/capacity admission for the real 40-layer checkpoint; CPU only."""

import argparse
import importlib.util
import json
import math
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import regex as re

GIB = 1024**3
NUMA_NODES = (6, 7, 4, 5, 0, 1, 2, 3)


def converter_module():
    path = Path(__file__).resolve().parents[2] / "examples/quantization/convert_deepseek_v41.py"
    spec = importlib.util.spec_from_file_location("v41_full_preflight_converter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(size))
    return {name: value for name, value in header.items() if name != "__metadata__"}


def resident_class(name):
    """Current TP8/EP1 text model ownership, before device layout padding."""
    if name.startswith(("vision.", "aligner.", "mtp.")) or name in {"image_start", "image_end", "image_newline"}:
        return "excluded_vision_or_draft", 0
    if ".engram.embed." in name:
        return "host_engram", 0
    if name.endswith("weight_shape"):
        return "checkpoint_shape_metadata", 0
    if ".ffn.experts." in name:
        return ("moe_packed" if name.endswith("weight_packed") else "moe_scales"), 8
    if name in {"embed.weight", "head.weight"}:
        return "tp_embedding_head", 8
    if ".ffn.shared_experts." in name:
        return "tp_shared_expert", 8
    if any(
        name.endswith(suffix)
        for suffix in (
            ".attn.wq_b.weight",
            ".attn.wo_a.weight",
            ".attn.wo_b.weight",
            ".attn.attn_sink",
        )
    ):
        return "tp_attention", 8
    return "replicated_dense_norm_hc_engram", 1


def host_memory():
    values = {
        line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
    }
    nodes = {}
    for node in NUMA_NODES:
        data = {
            parts[2].rstrip(":"): int(parts[3]) * 1024
            for line in Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines()
            if len(parts := line.split()) >= 4
        }
        nodes[str(node)] = data["MemFree"] + data["FilePages"] - data["Shmem"] + data["SReclaimable"]
    cgroup = {}
    for key in ("memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / key
        if path.exists():
            cgroup[key] = path.read_text().strip()
    if not cgroup:
        legacy = Path("/sys/fs/cgroup/memory")
        for key, filename in (("memory.max", "memory.limit_in_bytes"), ("memory.current", "memory.usage_in_bytes")):
            path = legacy / filename
            if path.exists():
                cgroup[key] = path.read_text().strip()
        if cgroup:
            cgroup["version"] = 1
            stat = legacy / "memory.stat"
            if stat.exists():
                for line in stat.read_text().splitlines():
                    if line.startswith("hierarchical_memory_limit "):
                        cgroup["memory.max"] = str(min(int(cgroup["memory.max"]), int(line.split()[1])))
    else:
        cgroup["version"] = 2
    return {"MemAvailable": values["MemAvailable"], "node_available_estimate": nodes, "cgroup": cgroup}


def hbm_memory():
    """npu-smi is read-only; never creates a torch/NPU context."""
    result = {}
    command = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
    rank = None
    for line in command.stdout.splitlines():
        model = re.match(r"\|\s*(\d+)\s+910\w*\s", line)
        if model:
            rank = model[1]
        memory = re.search(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", line)
        if rank is not None and re.search(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:", line) and memory:
            used, total = map(int, memory.groups())
            result[rank] = {
                "returncode": command.returncode,
                "raw": line,
                "total_bytes": total * 1024**2,
                "used_bytes": used * 1024**2,
                "free_bytes": (total - used) * 1024**2,
            }
            rank = None
    return result


def build_preflight(source, converted, *, query_hbm=True, reserve_gib=8):
    converter = converter_module()
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    if text["num_hidden_layers"] != 40 or text["n_routed_experts"] != 384:
        raise ValueError("Full-model admission requires all 40 layers and 384 experts")
    if text["engram_layer_ids"] != [1, 14] or text["engram_num_embeddings"] != [384006168, 384016682]:
        raise ValueError("Synthetic or modified Engram tables are forbidden")
    tensors, fingerprints, missing = converter.inventory(source)
    if missing:
        raise ValueError(f"Source headers incomplete: {missing}")
    expected = converter.output_specs(list(tensors), tensors)
    manifest_path = converted / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    converted_headers = {}
    for filename in manifest.get("shards", {}):
        path = converted / filename
        if path.is_file():
            converted_headers[filename] = read_header(path)
    ready, discrepancies = [], []
    for filename, entry in manifest.get("shards", {}).items():
        path = converted / filename
        planned = {name: info for name, info in expected.items() if info.file == filename}
        header = converted_headers.get(filename, {})
        if not path.is_file() or path.stat().st_size != entry["size"] or set(header) != set(planned):
            discrepancies.append(filename + ": missing/size/names mismatch")
            continue
        if any(
            header[name]["dtype"] != info.dtype or tuple(header[name]["shape"]) != info.shape
            for name, info in planned.items()
        ):
            discrepancies.append(filename + ": dtype/shape mismatch")
            continue
        ready.append(filename)
    categories = defaultdict(int)
    examples = defaultdict(list)
    per_rank = defaultdict(int)
    router_fp32 = 0
    for name, info in expected.items():
        category, divisor = resident_class(name)
        categories[category] += info.nbytes
        if len(examples[category]) < 3:
            examples[category].append(name)
        if divisor:
            if info.nbytes % divisor:
                raise ValueError(f"Unexpected fractional TP ownership: {name}")
            per_rank[category] += info.nbytes // divisor
        if name.endswith(".ffn.gate.weight") and name.startswith("layers."):
            # BF16 checkpoint/parameter stays live beside its precast FP32 copy.
            router_fp32 += math.prod(info.shape) * 4
    per_rank["router_additional_fp32_copy"] = router_fp32
    per_rank["runtime_fused_weight_shape_metadata"] = 40 * 384 * 2 * 2 * 4
    static = sum(per_rank.values())
    from vllm_ascend.ops.engram_hash import HostEngramLayout

    layout = HostEngramLayout.from_config(SimpleNamespace(**text))
    owners = []
    for rank in range(8):
        for layer, layer_id in enumerate(layout.layer_ids):
            heads, ranges = layout.head_shard(layer, rank, 8)
            rows = sum(end - start for start, end in ranges)
            owners.append(
                {
                    "rank": rank,
                    "node": NUMA_NODES[rank],
                    "layer": layer_id,
                    "heads": heads,
                    "ranges": ranges,
                    "bytes": rows * 256 * 2,
                }
            )
    host = host_memory()
    pinned = sum(owner["bytes"] for owner in owners)
    host_floor = pinned + 256 * GIB
    hbm = hbm_memory() if query_hbm else {}
    # One layer's w13 output allocation plus expanded single-expert temporaries;
    # conservatively reserve 1 GiB during bounded repacking, plus fixed 256 MiB KV.
    hbm_floor = static + GIB + 256 * 1024**2 + reserve_gib * GIB
    reasons = []
    if not manifest.get("complete") or len(ready) != 48 or discrepancies:
        reasons.append("Converted manifest is not complete with 48 matching output headers")
    if not (converted / "config.json").is_file() or not (converted / "model.safetensors.index.json").is_file():
        reasons.append("Final converted config/index are not published")
    else:
        published = json.loads((converted / "config.json").read_text())
        fields = (
            "num_hidden_layers",
            "n_routed_experts",
            "hidden_size",
            "moe_intermediate_size",
            "engram_layer_ids",
            "engram_num_embeddings",
            "engram_vocab_size",
            "engram_n_heads",
        )
        if any(published.get("text_config", {}).get(key) != text[key] for key in fields):
            reasons.append("Published config does not preserve the complete target/Engram shapes")
        target_format = converter.converted_config(config)
        if any(published.get(key) != target_format[key] for key in ("quantization_config", "ascend_weight_format")):
            reasons.append("Published checkpoint quantization/packing differs from signed-scale INT4 group32 RTN")
        published_index = json.loads((converted / "model.safetensors.index.json").read_text())["weight_map"]
        if published_index != {name: info.file for name, info in expected.items()}:
            reasons.append("Published converted index differs from the complete expected tensor map")
    for key, filename in (
        ("source_config_sha256", "config.json"),
        ("source_index_sha256", "model.safetensors.index.json"),
    ):
        if manifest.get(key) != converter.sha256_file(source / filename):
            reasons.append(f"Conversion manifest {key} does not match current source metadata")
    if host["MemAvailable"] < host_floor:
        reasons.append("Host MemAvailable is below pinned payload plus 256 GiB reserve")
    for rank in range(8):
        required = sum(owner["bytes"] for owner in owners if owner["rank"] == rank) + 32 * GIB
        if host["node_available_estimate"][str(NUMA_NODES[rank])] < required:
            reasons.append(f"Rank {rank} NUMA node lacks its pinned payload plus 32 GiB reserve")
        if str(rank) not in hbm or "free_bytes" not in hbm[str(rank)]:
            reasons.append(f"Rank {rank} current HBM availability has not been verified")
        elif hbm[str(rank)]["free_bytes"] < hbm_floor:
            reasons.append(f"Rank {rank} free HBM below estimated static weights + load/KV/reserve")
    limit = host["cgroup"].get("memory.max", "max")
    if limit != "max" and int(limit) - int(host["cgroup"].get("memory.current", 0)) < host_floor:
        reasons.append("Cgroup remaining memory below host pinned payload plus reserve")
    import torch

    initialized = hasattr(torch, "npu") and torch.npu.is_initialized()
    if initialized:
        raise RuntimeError("CPU preflight unexpectedly initialized NPU")
    return {
        "status": "ready_for_scheduled_launch" if not reasons else "not_ready",
        "npu_initialized": False,
        "source": str(source),
        "converted": str(converted),
        "source_headers": len(fingerprints),
        "converted_headers_verified": len(ready),
        "conversion_manifest_complete": manifest.get("complete", False),
        "discrepancies": discrepancies,
        "source_fingerprints": fingerprints,
        "header_category_bytes": dict(categories),
        "category_examples": dict(examples),
        "estimated_static_device_bytes_per_rank": static,
        "per_rank_device_categories": dict(per_rank),
        "hbm_admission_bytes_per_rank": hbm_floor,
        "hbm_workspace_reserve_gib": reserve_gib,
        "hbm_estimate_limits": (
            "Header ownership plus known FP32 router copies; excludes unmeasured allocator/NZ/workspace/graph peaks. "
            "Reserved headroom is a policy floor, not a measured peak."
        ),
        "host_pinned_bytes": pinned,
        "owners": owners,
        "host_admission_bytes": host_floor,
        "converted_checkpoint_payload_bytes": sum(info.nbytes for info in expected.values()),
        "host_file_cache_caveat": (
            "Real loading also faults converted file pages; budget up to the full checkpoint payload as reclaimable "
            "cache, beyond private pinned storage."
        ),
        "host_memory": host,
        "hbm_memory": hbm,
        "blockers": reasons,
        "payload_checksum_validation": (
            "Not repeated: this preflight reads headers/stat only. Preserve completed converter/recovery checksums "
            "and verify immutable outputs centrally before final acceptance."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--converted", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-hbm-query", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new report path")
    report = build_preflight(args.source, args.converted, query_hbm=not args.skip_hbm_query)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "estimated_static_device_bytes_per_rank", "host_pinned_bytes", "blockers")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
