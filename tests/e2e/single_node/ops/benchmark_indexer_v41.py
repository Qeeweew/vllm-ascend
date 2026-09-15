# SPDX-License-Identifier: Apache-2.0
"""CSA native selector latency and synthetic MXFP4/INT8 ranking audit.

Run directly using the same interpreter as the installed vllm-ascend.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from test_indexer_v41 import build_metadata, check_outputs, device_case, make_case, select

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Ops  # noqa: E402


def dense_baseline(case, mode, candidates):
    # Decode batch: each row owns its own contiguous cache region. This is a
    # fully batched baseline; no Python token loop, CPU copy, or CPU topk.
    batch = case["q"].shape[0]
    keys = case["k"].reshape(batch, -1, 128).float()
    scales = case["ks"].reshape(batch, -1).float()
    query = case["q"].float()
    weights = (case["w"] * case["qs"]).float()
    positions = torch.arange(keys.shape[1], device="npu")
    valid = positions[None] < case["sk"][:, None]
    block_ids = positions // 8
    if mode == "consumer":
        allowed = torch.zeros((batch, keys.shape[1] // 8 + 1), dtype=torch.int32, device="npu")
        safe = torch.where(candidates[:, 0] >= 0, candidates[:, 0], keys.shape[1] // 8).long()
        allowed.scatter_(1, safe, 1)
        valid = valid & allowed[:, block_ids].bool()

    def run():
        qk = (torch.bmm(query, keys.transpose(1, 2)) / 1024).relu().half().float()
        score = (qk * weights[:, :, None]).sum(1) * scales
        score = score.masked_fill(~valid, -torch.inf)
        indices = score.topk(512, sorted=False).indices.float().sort(-1).values
        indices = torch.where(indices < case["sk"][:, None], indices, -1).int()
        blocks = None
        if mode == "source":
            block_scores = score.reshape(batch, -1, 8).amax(-1)
            last = (case["sk"] - 1) // 8
            block_scores.scatter_(1, last[:, None].long(), torch.inf)
            vals, ids = block_scores.topk(min(2048, block_scores.shape[1]))
            blocks = torch.where(vals > -torch.inf, ids, -1).int()
        return indices, blocks

    return run


def gathered_baseline(case, candidates):
    """Experimental existing-CANN path: gather only the selected 2048x8 K.

    BF16 exactly represents INT8 integers. BF16 bmm with true FP32 output
    therefore reproduces INT32 QK exactly for D128 without INT8 GEMM API
    constraints. This is a diagnostic, not a production dispatch policy.
    """
    batch = case["q"].shape[0]
    query = case["q"].bfloat16()
    weights = (case["w"] * case["qs"]).float()
    keys = case["k"].reshape(-1, 128)
    scales = case["ks"].reshape(-1)
    offsets = torch.arange(8, device="npu")

    def run():
        block_ids = candidates[:, 0]
        safe = block_ids.clamp_min(0)
        physical = case["bt"].gather(1, (safe // 4).long())
        physical_positions = (physical * 32 + (safe % 4) * 8)[:, :, None] + offsets
        positions = (safe[:, :, None] * 8 + offsets).reshape(batch, -1)
        valid = (block_ids >= 0)[:, :, None].expand(-1, -1, 8).reshape(batch, -1)
        valid = valid & (positions < case["sk"][:, None])
        selected_keys = keys.index_select(0, physical_positions.reshape(-1).long()).reshape(batch, -1, 128)
        selected_scales = scales.index_select(0, physical_positions.reshape(-1).long()).reshape(batch, -1)
        qk = torch.bmm(query, selected_keys.bfloat16().transpose(1, 2), out_dtype=torch.float32)
        score = ((qk / 1024).relu().half().float() * weights[:, :, None]).sum(1) * selected_scales.float()
        score = score.masked_fill(~valid, -torch.inf)
        values, ids = score.topk(512, sorted=False)
        selected = positions.gather(1, ids)
        selected = torch.where(values > -torch.inf, selected, 2**24)
        selected = selected.float().sort(-1).values
        return torch.where(selected < 2**24, selected, -1).int(), None

    return run


def measure(run, unroll, samples=12):
    result = []
    for _ in range(samples):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        result.append(start.elapsed_time(end) * 1000 / unroll)
    return result


def summarize(rounds):
    medians = [statistics.median(r) for r in rounds]
    data = sorted(sum(rounds, []))
    median = statistics.median(data)
    return dict(
        median_us=median,
        p95_us=data[int(0.95 * (len(data) - 1))],
        spread=(max(medians) - min(medians)) / median,
        samples_us=rounds,
    )


def mxfp4_fakequant(value):
    groups = value.float().reshape(*value.shape[:-1], -1, 32)
    scale = 2.0 ** torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126) / 6))
    normalized = (groups / scale).abs().clamp_max(6)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])
    distance = (normalized[..., None] - levels).abs()
    # E2M1 RNE: on ties prefer even encoding index, as IEEE mantissa rounding.
    minimum = distance.amin(-1, keepdim=True)
    tied = distance == minimum
    codes = torch.arange(8)
    priority = codes + (codes % 2) * 16
    choice = torch.where(tied, priority, 100).argmin(-1)
    return (levels[choice] * groups.sign() * scale).reshape_as(value).bfloat16().float()


def numerical_audit():
    torch.manual_seed(41)
    q = torch.randn(8, 32, 128).bfloat16()
    k = torch.randn(32771, 128).bfloat16()
    weights = (torch.randn(8, 32).bfloat16() / 64).float()
    qi, qs = AscendIndexerV41Ops.quantize(q.npu())
    ki, ks = AscendIndexerV41Ops.quantize(k.npu())
    qi, qs, ki, ks = [v.cpu() for v in (qi, qs, ki, ks)]
    bf16 = ((q.float() @ k.float().T).relu() * weights[:, :, None]).sum(1)
    fp4 = ((mxfp4_fakequant(q) @ mxfp4_fakequant(k).T).relu() * weights[:, :, None]).sum(1)
    native = (
        ((qi.float() @ ki.float().T / 1024).relu().half().float() * (weights.half() * qs).float()[:, :, None]).sum(1)
        * ks.float()[None]
        * 1024
    )
    result = {}
    for name, score in (("mxfp4", fp4), ("int8_native", native)):
        result[name] = dict(
            nrmse_vs_bf16=float((score - bf16).square().mean().sqrt() / bf16.square().mean().sqrt()),
            recall512_vs_bf16=statistics.mean(
                [
                    len(set(a.tolist()) & set(b.tolist())) / 512
                    for a, b in zip(score.topk(512).indices, bf16.topk(512).indices)
                ]
            ),
        )
    result["int8_vs_mxfp4"] = dict(
        nrmse=float((native - fp4).square().mean().sqrt() / fp4.square().mean().sqrt()),
        recall512=statistics.mean(
            [
                len(set(a.tolist()) & set(b.tolist())) / 512
                for a, b in zip(native.topk(512).indices, fp4.topk(512).indices)
            ]
        ),
    )
    result["scope"] = "Seed41 synthetic Gaussian BF16 activations; not checkpoint activations or model quality."
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--lengths", type=int, nargs="+", default=[4097, 32771, 131075])
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    torch.npu.matmul.allow_hf32 = False
    report = dict(
        cases=[],
        numerical=numerical_audit(),
        gate="native median <= dense baseline, native p95 <= baseline*1.05; round spread <= 3%",
    )
    for batch in args.batches:
        for length in args.lengths:
            case = make_case(1, [1] * batch, [length] * batch)
            device = device_case(case)
            source = AscendIndexerV41Ops(1, "source")
            metadata = build_metadata(source, device, max_q=1, max_k=length)
            _, candidates = select(source, device, metadata)
            for mode in ("off", "source", "consumer"):
                ops = AscendIndexerV41Ops(1, mode)
                candidate_arg = candidates if mode == "consumer" else None

                def native(ops=ops, candidate_arg=candidate_arg, device=device, metadata=metadata):
                    return select(ops, device, metadata, candidate_arg)

                idx, blk = native()
                check_outputs(case, 1, mode, idx.cpu(), blk.cpu(), candidates.cpu() if mode == "consumer" else None)
                baseline = dense_baseline(device, mode, candidates)
                original = dict(native=native, dense=baseline)
                if mode == "consumer":
                    gathered = gathered_baseline(device, candidates)
                    gathered_idx, _ = gathered()
                    check_outputs(case, 1, mode, gathered_idx.cpu()[:, None], None, candidates.cpu())
                    original["candidate_gather"] = gathered
                graphs, outputs, runs, samples = {}, {}, {}, {}
                for name, run in original.items():
                    for _ in range(3):
                        run()
                    graph = torch.npu.NPUGraph()
                    # Different unroll counts avoid retaining dense score slabs
                    # proportional to native's much smaller workspace.
                    unroll = 64 if name == "native" else 4
                    with torch.npu.graph(graph):
                        for _ in range(unroll):
                            output = run()
                    graphs[name], outputs[name], runs[name] = graph, output, graph.replay
                    samples[name] = []
                for round_id in range(3):
                    for name in list(original) if round_id % 2 == 0 else list(reversed(original)):
                        samples[name].append(measure(runs[name], 64 if name == "native" else 4))
                stats = {name: summarize(values) for name, values in samples.items()}
                n, d = stats["native"], stats["dense"]
                record = dict(
                    batch=batch,
                    length=length,
                    mode=mode,
                    **stats,
                    speedup=d["median_us"] / n["median_us"],
                    latency_gate=n["median_us"] <= d["median_us"] and n["p95_us"] <= d["p95_us"] * 1.05,
                    noise_gate=max(n["spread"], d["spread"]) <= 0.03,
                )
                report["cases"].append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(batch, length, mode, round(n["median_us"], 3), round(record["speedup"], 2), flush=True)
                # External tensor-owning closures stay alive until graphs die.
                del graphs, runs, outputs, original, baseline
                torch.npu.synchronize()
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
