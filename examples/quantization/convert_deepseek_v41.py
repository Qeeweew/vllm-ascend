# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream DeepSeek V4.1 MXFP4/FP8 checkpoints into signed-scale W4A16/BF16.

The output keeps DeepSeek's native tensor names. Expert ``weight`` becomes
``weight_packed`` plus ``weight_scale`` and ``weight_shape``. Packed checkpoint
nibbles encode q+8, matching Ascend's compressed-tensors loader; this is NOT the
two's-complement layout consumed by the device kernel after repacking.
"""

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

FORMAT_VERSION = 1
GROUP_SIZE = 32
PACK_FACTOR = 8
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}
EXPERT_WEIGHT = re.compile(r"(?:^|\.)experts\.\d+\.w[123]\.weight$")


@dataclass(frozen=True)
class TensorInfo:
    file: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def inventory(source: Path) -> tuple[dict[str, TensorInfo], dict, list[str]]:
    """Validate all available headers against the authoritative tensor index."""
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors: dict[str, TensorInfo] = {}
    fingerprints = {}
    missing = []
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError(f"Shard must be a plain filename: {filename}")
        path = source / filename
        if not path.is_file():
            missing.append(filename)
            continue
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > path.stat().st_size - 8 or header_size > 100_000_000:
                raise ValueError(f"Invalid safetensors header length: {path}")
            raw_header = stream.read(header_size)
            header = json.loads(raw_header)
        expected_names = {key for key, shard in weight_map.items() if shard == filename}
        actual_names = set(header) - {"__metadata__"}
        if actual_names != expected_names:
            raise ValueError(f"Tensor index/header mismatch in {filename}")
        cursor = 0
        for name in sorted(actual_names, key=lambda key: header[key]["data_offsets"]):
            entry = header[name]
            shape = tuple(entry["shape"])
            if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise ValueError(f"Invalid shape for {name}: {shape}")
            info = TensorInfo(filename, entry["dtype"], shape)
            start, end = entry["data_offsets"]
            if start != cursor or end - start != info.nbytes:
                raise ValueError(f"Invalid/overlapping tensor offsets for {name}")
            cursor = end
            tensors[name] = info
        stat = path.stat()
        if stat.st_size != 8 + header_size + cursor:
            raise ValueError(f"Incomplete or oversized shard: {path}")
        fingerprints[filename] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        }
    return tensors, fingerprints, missing


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 low/high nibbles and per-row UE8M0 groups to BF16."""
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError("MXFP4 weight must be int8 [N,K/2]")
    if scales.shape != (packed.shape[0], packed.shape[1] * 2 // GROUP_SIZE):
        raise ValueError("MXFP4 scale shape must be [N,K/32]")
    if packed.shape[1] * 2 % GROUP_SIZE:
        raise ValueError("MXFP4 K must be divisible by 32")
    codes = packed.view(torch.uint8)
    table = torch.tensor(FP4_VALUES, dtype=torch.float32)
    values = torch.stack((table[(codes & 15).long()], table[(codes >> 4).long()]), dim=-1).flatten(-2)
    values = values.unflatten(-1, (-1, GROUP_SIZE)) * scales.float().unsqueeze(-1)
    result = values.flatten(-2).bfloat16()
    if not torch.isfinite(result).all():
        raise ValueError("MXFP4 dequantization produced a non-finite BF16 weight")
    return result


def signed_scale_rtn(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed q and BF16 scales using the stored scale for RTN."""
    if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
        raise ValueError("RTN expects [N,K] with K divisible by 32")
    values = weight.bfloat16().float().unflatten(-1, (-1, GROUP_SIZE))
    if not torch.isfinite(values).all():
        raise ValueError("RTN input must be finite")
    negative = -values.amin(dim=-1).clamp(max=0)
    positive = values.amax(dim=-1).clamp(min=0)
    scale = torch.where(negative > positive, negative / 8, -positive / torch.where(negative == positive, 7, 8))
    scale = torch.where(scale == 0, torch.finfo(torch.float32).eps, scale).bfloat16()
    if not torch.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("Signed scale is not representable as a nonzero BF16")
    q = (values / scale.float().unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8).flatten(-2)
    return q, scale


def pack_checkpoint_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack q+8 along K; native NPU two's-complement packing is separate."""
    if q.ndim != 2 or q.shape[1] % PACK_FACTOR or q.dtype != torch.int8:
        raise ValueError("Packing expects int8 [N,K] with K divisible by 8")
    if ((q < -8) | (q > 7)).any():
        raise ValueError("INT4 codes must be in [-8,7]")
    codes = (q.int() + 8).unflatten(-1, (-1, PACK_FACTOR))
    result = torch.zeros(codes.shape[:-1], dtype=torch.int32)
    for nibble in range(PACK_FACTOR):
        result |= codes[..., nibble] << (4 * nibble)
    return result


def operation(name: str, info: TensorInfo) -> str:
    if name.endswith(".weight") and info.dtype == "I8" and EXPERT_WEIGHT.search(name):
        return "mxfp4"
    if name.endswith(".weight") and info.dtype == "F8_E4M3":
        return "fp8"
    return "copy"


def output_specs(names: list[str], tensors: dict[str, TensorInfo]) -> dict[str, TensorInfo]:
    consumed_scales = {
        name.removesuffix(".weight") + ".scale" for name, info in tensors.items() if operation(name, info) != "copy"
    }
    result = {}
    for name in names:
        info = tensors[name]
        if name in consumed_scales:
            continue
        action = operation(name, info)
        if action == "mxfp4":
            n, packed_k = info.shape
            k = packed_k * 2
            if k % GROUP_SIZE:
                raise ValueError(f"Invalid MXFP4 K: {name}")
            stem = name.removesuffix("weight")
            result[stem + "weight_packed"] = TensorInfo(info.file, "I32", (n, k // PACK_FACTOR))
            result[stem + "weight_scale"] = TensorInfo(info.file, "BF16", (n, k // GROUP_SIZE))
            result[stem + "weight_shape"] = TensorInfo(info.file, "I32", (2,))
        elif action == "fp8":
            result[name] = TensorInfo(info.file, "BF16", info.shape)
        else:
            if info.dtype.startswith("F8_"):
                raise ValueError(f"Unpaired or unsupported FP8 tensor: {name}")
            result[name] = info
    return result


class TensorWriter:
    """Bounded-memory safetensors writer supporting disjoint row writes."""

    def __init__(self, path: Path, specs: dict[str, TensorInfo]):
        self.specs = specs
        self.written = dict.fromkeys(specs, 0)
        header: dict = {"__metadata__": {"format": "pt", "ascend_v41_format": str(FORMAT_VERSION)}}
        cursor = 0
        self.offsets = {}
        for name, info in specs.items():
            self.offsets[name] = cursor
            header[name] = {"dtype": info.dtype, "shape": info.shape, "data_offsets": [cursor, cursor + info.nbytes]}
            cursor += info.nbytes
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        self.data_start = 8 + len(raw)
        self.stream = path.open("w+b")
        self.stream.write(struct.pack("<Q", len(raw)) + raw)
        self.stream.truncate(self.data_start + cursor)

    def write(self, name: str, tensor: torch.Tensor) -> None:
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        size = raw.nbytes
        if self.written[name] + size > self.specs[name].nbytes:
            raise ValueError(f"Too many bytes for {name}")
        self.stream.seek(self.data_start + self.offsets[name] + self.written[name])
        self.stream.write(memoryview(raw))
        self.written[name] += size

    def finish(self) -> None:
        for name, size in self.written.items():
            if size != self.specs[name].nbytes:
                raise ValueError(f"Incomplete output tensor: {name}")
        self.stream.flush()
        os.fsync(self.stream.fileno())


def dequantize_fp8_rows(weight: torch.Tensor, scale: torch.Tensor, block_rows: int, block_cols: int) -> torch.Tensor:
    expanded = scale.float().repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)
    result = (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).bfloat16()
    if not torch.isfinite(result).all():
        raise ValueError("FP8 dequantization produced a non-finite BF16 weight")
    return result


def convert_shard(source: Path, destination: Path, filename: str, tensors: dict[str, TensorInfo], rows: int) -> dict:
    """Convert a shard in row blocks; paired scales may reside in another file."""
    names = [name for name, info in tensors.items() if info.file == filename]
    specs = output_specs(names, tensors)
    temporary = destination / (filename + ".partial")
    writer = TensorWriter(temporary, specs)
    try:
        with contextlib.ExitStack() as stack:
            readers = {}

            def reader(shard: str):
                if shard not in readers:
                    readers[shard] = stack.enter_context(safe_open(source / shard, framework="pt", device="cpu"))
                return readers[shard]

            for name in names:
                info = tensors[name]
                action = operation(name, info)
                if action == "copy" and name not in specs:
                    continue
                weight_slice = reader(info.file).get_slice(name)
                scale_slice = None
                block_rows = block_cols = 1
                if action != "copy":
                    scale_name = name.removesuffix(".weight") + ".scale"
                    if scale_name not in tensors:
                        raise ValueError(f"Missing paired scale: {scale_name}")
                    scale_info = tensors[scale_name]
                    scale_slice = reader(scale_info.file).get_slice(scale_name)
                    if action == "fp8":
                        if len(info.shape) != 2 or len(scale_info.shape) != 2:
                            raise ValueError(f"Expected matrix FP8 weights/scales: {name}")
                        block_rows = 1 if ".engram.embed." in name else GROUP_SIZE
                        block_cols = GROUP_SIZE
                        expected = (math.ceil(info.shape[0] / block_rows), math.ceil(info.shape[1] / block_cols))
                        if scale_info.shape != expected:
                            raise ValueError(f"Invalid scale shape for {name}: {scale_info.shape}, expected {expected}")
                step = max(block_rows, rows // block_rows * block_rows)
                if not info.shape:
                    writer.write(name, reader(info.file).get_tensor(name))
                    continue
                for start in range(0, info.shape[0], step):
                    end = min(start + step, info.shape[0])
                    weight = weight_slice[start:end]
                    if action == "mxfp4":
                        assert scale_slice is not None
                        q, scales = signed_scale_rtn(dequantize_mxfp4(weight, scale_slice[start:end]))
                        stem = name.removesuffix("weight")
                        writer.write(stem + "weight_packed", pack_checkpoint_int4(q))
                        writer.write(stem + "weight_scale", scales)
                    elif action == "fp8":
                        assert scale_slice is not None
                        scale = scale_slice[start // block_rows : math.ceil(end / block_rows)]
                        writer.write(name, dequantize_fp8_rows(weight, scale, block_rows, block_cols))
                    else:
                        writer.write(name, weight)
                if action == "mxfp4":
                    writer.write(
                        name.removesuffix("weight") + "weight_shape",
                        torch.tensor([info.shape[0], info.shape[1] * 2], dtype=torch.int32),
                    )
        writer.finish()
    finally:
        writer.stream.close()
    checksum = sha256_file(temporary)
    temporary.replace(destination / filename)
    return {
        "sha256": checksum,
        "size": (destination / filename).stat().st_size,
        "tensors": {name: {"dtype": info.dtype, "shape": info.shape} for name, info in specs.items()},
    }


def converted_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    quant = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "moe": {
                "targets": [r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": GROUP_SIZE,
                    "dynamic": False,
                },
                "input_activations": None,
            }
        },
        "ignore": [],
    }
    result["quantization_config"] = quant
    if "text_config" in result:
        result["text_config"].pop("quantization_config", None)
        result["text_config"].pop("expert_dtype", None)
    result["ascend_weight_format"] = {
        "version": FORMAT_VERSION,
        "signed_scale": True,
        "scale_dtype": "bfloat16",
        "group_size": GROUP_SIZE,
        "group_axis": "K",
        "checkpoint_packing": "offset_binary_q_plus_8",
        "rounding": "nearest_even_using_stored_scale",
        "zero_group_scale": float(torch.finfo(torch.float32).eps),
        "engram_dtype": "bfloat16",
        "source_format": "deepseek_v41_mxfp4_fp8",
    }
    return result


def convert(
    source: Path,
    destination: Path,
    rows: int = 256,
    allow_incomplete: bool = False,
    max_shards: int | None = None,
    verify_only: bool = False,
) -> dict:
    """Resume verified shards; publish config/index only for a complete model."""
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        raise ValueError("Output must be separate from the source model directory")
    if rows <= 0 or (max_shards is not None and max_shards <= 0):
        raise ValueError("rows/max_shards must be positive")
    tensors, fingerprints, missing = inventory(source)
    if missing and not allow_incomplete:
        raise ValueError(f"Checkpoint is incomplete: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".conversion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = destination / "conversion_manifest.json"
        if not manifest_path.exists() and any(path.name != ".conversion.lock" for path in destination.iterdir()):
            raise ValueError("Output directory contains files without a conversion manifest")
        config = json.loads((source / "config.json").read_text())
        identity = {
            "format_version": FORMAT_VERSION,
            "source_config_sha256": sha256_file(source / "config.json"),
            "source_index_sha256": sha256_file(source / "model.safetensors.index.json"),
            "weight_format": converted_config(config)["ascend_weight_format"],
        }
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {**identity, "shards": {}}
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("Conversion parameters/source identity differ from the existing manifest")
        atomic_json(manifest_path, manifest)
        converted = 0
        for filename, fingerprint in fingerprints.items():
            previous = manifest["shards"].get(filename)
            if previous:
                if previous["source"] != fingerprint:
                    raise ValueError(f"Source shard changed since conversion: {filename}")
                output = destination / filename
                if not output.is_file() or output.stat().st_size != previous["size"]:
                    raise ValueError(f"Missing/truncated converted shard: {filename}")
                if sha256_file(output) != previous["sha256"]:
                    raise ValueError(f"Converted shard checksum mismatch: {filename}")
                continue
            if verify_only or (max_shards is not None and converted >= max_shards):
                continue
            names = [name for name, info in tensors.items() if info.file == filename]
            dependencies = {
                name.removesuffix(".weight") + ".scale" for name in names if operation(name, tensors[name]) != "copy"
            }
            if dependencies - tensors.keys():
                if allow_incomplete:
                    continue
                raise ValueError(f"Missing scales for {filename}: {dependencies - tensors.keys()}")
            print(f"Converting {filename}", flush=True)
            entry = convert_shard(source, destination, filename, tensors, rows)
            for shard in {filename} | {tensors[name].file for name in dependencies}:
                stat = (source / shard).stat()
                before = fingerprints[shard]
                if (stat.st_size, stat.st_mtime_ns) != (before["size"], before["mtime_ns"]):
                    raise ValueError(f"Source shard changed during conversion: {shard}")
            entry["source"] = fingerprint
            # Cross-shard scale dependencies must remain unchanged on resume too.
            entry["scale_sources"] = {tensors[name].file: fingerprints[tensors[name].file] for name in dependencies}
            manifest["shards"][filename] = entry
            converted += 1
            atomic_json(manifest_path, manifest)
        for entry in manifest["shards"].values():
            for filename, fingerprint in entry.get("scale_sources", {}).items():
                if fingerprints.get(filename) != fingerprint:
                    raise ValueError(f"Paired scale shard changed or disappeared: {filename}")
        expected_shards = set(json.loads((source / "model.safetensors.index.json").read_text())["weight_map"].values())
        complete = not missing and set(manifest["shards"]) == expected_shards
        manifest["complete"] = complete
        manifest["missing_source_shards"] = missing
        atomic_json(manifest_path, manifest)
        if complete and not verify_only:
            weight_map = {}
            total_size = 0
            for filename, entry in manifest["shards"].items():
                for name, spec in entry["tensors"].items():
                    if name in weight_map:
                        raise ValueError(f"Duplicate output tensor: {name}")
                    weight_map[name] = filename
                    total_size += math.prod(spec["shape"]) * DTYPE_BYTES[spec["dtype"]]
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "preprocessor_config.json",
            ):
                if (source / filename).is_file():
                    shutil.copy2(source / filename, destination / filename)
            if (source / "encoding").is_dir():
                shutil.copytree(source / "encoding", destination / "encoding", dirs_exist_ok=True)
            atomic_json(destination / "config.json", converted_config(config))
            atomic_json(
                destination / "model.safetensors.index.json",
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            )
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-block", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = convert(
        args.source, args.output, args.rows_per_block, args.allow_incomplete, args.max_shards, args.verify_only
    )
    print(
        json.dumps(
            {
                "complete": manifest["complete"],
                "converted_shards": len(manifest["shards"]),
                "missing_source_shards": manifest["missing_source_shards"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
