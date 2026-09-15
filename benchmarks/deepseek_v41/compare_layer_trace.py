# SPDX-License-Identifier: Apache-2.0
"""Compare bounded CPU activation traces from the eager worker diagnostic."""

import argparse
import json
from pathlib import Path

import torch


def tensors(value, name):
    if isinstance(value, dict) and "tensor" in value:
        yield name, value["tensor"]
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{name}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f"{name}.{key}")


def difference(first, second):
    if first.shape != second.shape or first.dtype != second.dtype:
        return {
            "equal": False,
            "shape_a": list(first.shape),
            "shape_b": list(second.shape),
            "dtype_a": str(first.dtype),
            "dtype_b": str(second.dtype),
        }
    changed = first != second
    delta = first.double() - second.double()
    indices = changed.nonzero()
    result = {
        "equal": torch.equal(first, second),
        "numel": first.numel(),
        "changed": int(changed.sum()),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "nrmse": float(delta.norm() / first.double().norm().clamp_min(1e-30)),
        "finite_a": bool(torch.isfinite(first).all()),
        "finite_b": bool(torch.isfinite(second).all()),
    }
    if indices.numel():
        index = tuple(indices[0].tolist())
        result.update(first_difference_index=list(index), value_a=float(first[index]), value_b=float(second[index]))
    return result


def compare(directory, first_id, second_id):
    first, second = [
        torch.load(directory / f"forward_{sequence:05d}.pt", map_location="cpu", weights_only=True)
        for sequence in (first_id, second_id)
    ]
    inputs = {}
    for name in ("input_ids", "positions", "cu_seqlens_q", "seqused_kv"):
        if first[name]["truncated"] or second[name]["truncated"]:
            raise ValueError(f"Cannot certify the same full request from truncated {name}")
        inputs[name] = difference(first[name]["tensor"], second[name]["tensor"])
    if not all(value["equal"] for value in inputs.values()):
        raise ValueError(f"Trace requests/boundaries differ: {inputs}")
    stages = []
    if list(first["stages"]) != list(second["stages"]):
        raise ValueError("Trace stage order differs")
    for name, value in first["stages"].items():
        lhs, rhs = dict(tensors(value, name)), dict(tensors(second["stages"][name], name))
        if lhs.keys() != rhs.keys():
            raise ValueError(f"Trace tensor structure differs at {name}")
        for key, tensor in lhs.items():
            stages.append({"stage": key, **difference(tensor, rhs[key])})
    logits = [directory / f"forward_{sequence:05d}_logits_input.pt" for sequence in (first_id, second_id)]
    if all(path.exists() for path in logits):
        a, b = [torch.load(path, map_location="cpu", weights_only=True)["hidden_states"]["tensor"] for path in logits]
        stages.append({"stage": "compute_logits.input", **difference(a, b)})
    metadata = []
    for section in ("attention_cache", "compressor_state"):
        lhs = dict(tensors(first.get(section, {}), section))
        rhs = dict(tensors(second.get(section, {}), section))
        for name, tensor in lhs.items():
            metadata.append({"name": name, **difference(tensor, rhs[name])})
    result = {
        "directory": str(directory),
        "first": first_id,
        "second": second_id,
        "tp_rank": first["tp_rank"],
        "req_ids_a": first["req_ids"],
        "req_ids_b": second["req_ids"],
        "inputs": inputs,
        "first_difference": next((stage for stage in stages if not stage["equal"]), None),
        "stages": stages,
        "metadata": metadata,
        "scope": "CPU snapshots from one rank; HCCL attribution requires all ranks' local inputs",
    }
    return result


def compare_all_ranks(directory, first_id, second_id):
    rank_results = []
    local_a, local_b, reduced_a, reduced_b = [], [], [], []
    for rank in range(8):
        rank_dir = directory / f"rank{rank}"
        rank_results.append(compare(rank_dir, first_id, second_id))
        for sequence, local, reduced in ((first_id, local_a, reduced_a), (second_id, local_b, reduced_b)):
            payload = torch.load(rank_dir / f"forward_{sequence:05d}.pt", map_location="cpu", weights_only=True)
            local.append(payload["stages"]["layer0.attention.wo_b.local_matmul"]["tensor"])
            reduced.append(payload["stages"]["layer0.attention.wo_b.output"]["tensor"])
    sum_a = torch.stack(local_a).float().sum(0)
    sum_b = torch.stack(local_b).float().sum(0)
    return {
        "directory": str(directory),
        "first": first_id,
        "second": second_id,
        "ranks": rank_results,
        "all_rank_local_inputs_identical": all(torch.equal(a, b) for a, b in zip(local_a, local_b)),
        "first_output_identical_across_ranks": all(torch.equal(reduced_a[0], value) for value in reduced_a),
        "second_output_identical_across_ranks": all(torch.equal(reduced_b[0], value) for value in reduced_b),
        "fp32_sum_across_forwards": difference(sum_a, sum_b),
        "hccl_output_across_forwards": difference(reduced_a[0], reduced_b[0]),
        "first_hccl_vs_fp32_sum": difference(reduced_a[0].float(), sum_a),
        "second_hccl_vs_fp32_sum": difference(reduced_b[0].float(), sum_b),
        "first_hccl_vs_once_rounded_fp32_sum": difference(reduced_a[0], sum_a.to(reduced_a[0].dtype)),
        "second_hccl_vs_once_rounded_fp32_sum": difference(reduced_b[0], sum_b.to(reduced_b[0].dtype)),
        "scope": "Eight TP-rank CPU snapshots immediately before and after first-layer wo_b all-reduce",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--second", type=int, default=11)
    parser.add_argument("--all-ranks", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    compare_fn = compare_all_ranks if args.all_ranks else compare
    result = compare_fn(args.directory, args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.all_ranks:
        summary = {key: value for key, value in result.items() if key != "ranks"}
        summary["first_differences"] = [rank["first_difference"] for rank in result["ranks"]]
    else:
        summary = {
            "first_difference": result["first_difference"],
            "different_stages": [row["stage"] for row in result["stages"] if not row["equal"]],
            "different_metadata": [row["name"] for row in result["metadata"] if not row["equal"]],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
