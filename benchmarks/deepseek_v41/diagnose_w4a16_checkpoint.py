# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint W4 numerical diagnosis; no timing or acceptance gate changes.

The selected-expert bank is remapped densely without changing any selected
weight. This isolates arithmetic, not E384 dispatch or TP communication.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def error(actual, reference):
    actual, reference = actual.float(), reference.float()
    diff = actual - reference
    return dict(
        nrmse=float(diff.norm() / reference.norm().clamp_min(1e-20)),
        max_abs=float(diff.abs().max()),
        rms=float(actual.square().mean().sqrt()),
        max_scaled=float(diff.abs().max() / reference.abs().max().clamp_min(1e-20)),
    )


def repeat_summary(samples, reference):
    baseline = samples[0]
    comparisons = [error(sample, baseline) for sample in samples]
    changes = [int((sample != baseline).sum()) for sample in samples]
    return {
        "samples": len(samples),
        "changed_samples": sum(count > 0 for count in changes),
        "max_changed_elements": max(changes),
        "max_abs_from_first": max(row["max_abs"] for row in comparisons),
        "max_nrmse_from_first": max(row["nrmse"] for row in comparisons),
        "max_nrmse_vs_fp32_contract": max(error(sample, reference)["nrmse"] for sample in samples),
        "all_finite": all(bool(torch.isfinite(sample).all()) for sample in samples),
    }


def unpack(packed):
    return torch.stack([(packed >> (4 * i)) & 15 for i in range(8)], -1).flatten(-2).sub(8).to(torch.int8)


def repack(q):
    q = q.t().contiguous().int()
    packed = torch.zeros(q.shape[0], q.shape[1] // 8, dtype=torch.int32)
    for i in range(8):
        packed |= (q[:, i::8] & 15) << (4 * i)
    return packed


def effective(q, scales, round_weight=False):
    value = q.float() * scales.float().repeat_interleave(32, -1)
    return value.bfloat16().float() if round_weight else value


def reference(x, experts, ids, routing, *, round_gate=False, round_down=False, round_router=False, round_weight=False):
    output = torch.zeros_like(x, dtype=torch.float32)
    clipping = [0, 0, 0]
    for expert, (q13, s13, q2, s2) in enumerate(experts):
        tokens, routes = (ids == expert).nonzero(as_tuple=True)
        if not tokens.numel():
            continue
        h13 = F.linear(x[tokens].float(), effective(q13, s13, round_weight))
        if round_gate:
            h13 = h13.bfloat16().float()
        gate, up = h13.chunk(2, -1)
        clipping[0] += int((gate > 10).sum())
        clipping[1] += int((up.abs() > 10).sum())
        clipping[2] += gate.numel()
        act = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16()
        down = F.linear(act.float(), effective(q2, s2, round_weight))
        if round_down:
            down = down.bfloat16().float()
        weights = routing[tokens, routes]
        if round_router:
            weights = weights.bfloat16().float()
        output.index_add_(0, tokens, down * weights[:, None])
    return output.bfloat16(), clipping


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--input", type=Path, help="Actual MoE torch dict: x, ids, routing, optional layer/tp_rank")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--native-repeats", type=int, default=1, help="Isolate same-input eager and graph repeatability"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.native_repeats < 1:
        parser.error("--native-repeats must be positive")
    torch.set_num_threads(8)
    index_path = args.checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        # Conversion writes the complete index last; finished shards are
        # already usable for a bounded layer diagnosis while it continues.
        index = {}
        for shard in sorted(args.checkpoint.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                index.update({name: shard.name for name in reader.keys()})  # noqa: SIM118

    def read(name, rows=None, columns=None):
        rows = slice(None) if rows is None else rows
        columns = slice(None) if columns is None else columns
        with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as reader:
            return reader.get_slice(name)[rows, columns].clone()

    prefix = f"layers.{args.layer}.ffn"
    if args.input:
        capture = torch.load(args.input, map_location="cpu", weights_only=True)
        assert capture.get("layer", args.layer) == args.layer
        assert capture.get("tp_rank", args.tp_rank) == args.tp_rank
        x, source_ids, routing = (capture[key].cpu() for key in ("x", "ids", "routing"))
    else:
        generator = torch.Generator().manual_seed(41091)
        raw = torch.randn(4, 5120, generator=generator)
        norm_name = f"layers.{args.layer}.ffn_norm.weight"
        with safe_open(args.checkpoint / index[norm_name], framework="pt", device="cpu") as reader:
            norm = reader.get_tensor(norm_name)
        x = (raw * raw.square().mean(-1, keepdim=True).rsqrt() * norm.float()).bfloat16()
        gate = read(f"{prefix}.gate.weight")
        with safe_open(args.checkpoint / index[f"{prefix}.gate.bias"], framework="pt", device="cpu") as reader:
            bias = reader.get_tensor(f"{prefix}.gate.bias").float()
        score = F.softplus(F.linear(x, gate).float()).sqrt()
        source_ids = (score + bias).topk(6, dim=-1).indices
        routing = score.gather(1, source_ids)
        routing = routing / routing.sum(-1, keepdim=True) * 1.5
    x, source_ids, routing = x.bfloat16(), source_ids.long(), routing.float()
    selected, inverse = torch.unique(source_ids, sorted=True, return_inverse=True)
    ids = inverse.reshape_as(source_ids).int()
    experts, packed13, packed2, scales13, scales2 = [], [], [], [], []
    start, end = args.tp_rank * 288, (args.tp_rank + 1) * 288
    for expert in selected.tolist():
        stem = f"{prefix}.experts.{expert}"
        q1 = unpack(read(f"{stem}.w1.weight_packed", slice(start, end)))
        q3 = unpack(read(f"{stem}.w3.weight_packed", slice(start, end)))
        q13 = torch.cat((q1, q3))
        s13 = torch.cat(
            (read(f"{stem}.w1.weight_scale", slice(start, end)), read(f"{stem}.w3.weight_scale", slice(start, end)))
        )
        q2 = unpack(read(f"{stem}.w2.weight_packed", columns=slice(start // 8, end // 8)))
        s2 = read(f"{stem}.w2.weight_scale", columns=slice(start // 32, end // 32))
        experts.append((q13, s13, q2, s2))
        packed13.append(repack(q13))
        packed2.append(repack(q2))
        scales13.append(s13.t().contiguous())
        scales2.append(s2.t().contiguous())
    references, clipping_counts = {}, {}
    variants = {
        "fp32_contract": {},
        "bf16_gmm1": dict(round_gate=True),
        "bf16_gmm1_gmm2": dict(round_gate=True, round_down=True),
        "bf16_gmm1_gmm2_router": dict(round_gate=True, round_down=True, round_router=True),
        "bf16_dequant_and_boundaries": dict(round_gate=True, round_down=True, round_router=True, round_weight=True),
    }
    for name, flags in variants.items():
        references[name], clipping_counts[name] = reference(x, experts, ids, routing, **flags)
    result = dict(
        checkpoint=str(args.checkpoint),
        layer=args.layer,
        tp_rank=args.tp_rank,
        input_kind="captured_decode" if args.input else "synthetic_normalized_hidden_real_router",
        input_file=str(args.input) if args.input else None,
        source_ids=source_ids.tolist(),
        routing=routing.tolist(),
        input_rms=float(x.float().square().mean().sqrt()),
        experts_loaded=selected.numel(),
        input_bytes=x.numel() * x.element_size(),
        clipping_gate_up_total=clipping_counts,
        cpu_variants={name: error(value, references["fp32_contract"]) for name, value in references.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.cpu_only:
        import torch_npu
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        dx, di, dr = (value.to(device) for value in (x, ids, routing))
        w13, s13, w2, s2 = (torch.stack(value).to(device) for value in (packed13, scales13, packed2, scales2))
        native = torch.ops._C_ascend.npu_w4a16_moe(dx, w13, s13, w2, s2, di, dr, 10.0).cpu()
        repeat_samples = {}
        if args.native_repeats > 1:

            def call_native():
                return torch.ops._C_ascend.npu_w4a16_moe(dx, w13, s13, w2, s2, di, dr, 10.0)

            # Each CPU copy synchronizes and owns storage before the next call.
            # This deliberately measures numerical stability, never latency.
            repeat_samples["eager"] = [call_native().cpu().clone() for _ in range(args.native_repeats)]
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_output = call_native()
            repeat_samples["graph"] = []
            for _ in range(args.native_repeats):
                graph.replay()
                repeat_samples["graph"].append(graph_output.cpu().clone())
            result["native_repeatability"] = {
                mode: repeat_summary(samples, references["fp32_contract"]) for mode, samples in repeat_samples.items()
            }
        routed, reverse, counts, _ = torch_npu.npu_moe_init_routing_v2(
            dx,
            di,
            active_num=di.numel(),
            expert_num=selected.numel(),
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, selected.numel()],
            quant_mode=-1,
        )
        counts = counts.long()

        def gmm(value, weight, scales):
            return torch_npu.npu_grouped_matmul(
                x=[value],
                weight=[weight],
                antiquant_scale=[scales],
                group_list=counts,
                group_list_type=1,
                group_type=0,
                split_item=2,
                output_dtype=torch.bfloat16,
            )[0]

        h13 = gmm(routed, w13, s13)
        h13_before_clip = h13.cpu()
        gate, up = h13.chunk(2, -1)
        gate.clamp_(max=10)
        up.clamp_(-10, 10)
        act = torch_npu.npu_swiglu(h13)
        h2 = gmm(act, w2, s2)
        cann = torch_npu.npu_moe_token_unpermute(h2, reverse.abs(), probs=dr.bfloat16()).cpu()
        result["native_vs_cann"] = error(native, cann)
        result["native_vs_cpu"] = {name: error(native, value) for name, value in references.items()}
        result["cann_vs_cpu"] = {name: error(cann, value) for name, value in references.items()}
        if args.input and "output" in capture:
            result["rerun_vs_captured_cann"] = error(cann, capture["output"])
            result["native_vs_captured_cann"] = error(native, capture["output"])
        # Isolate down-GMM arithmetic using exactly the same CANN activation.
        cursor = 0
        exact_down, rounded_weight_down, exact_gate, rounded_weight_gate = [], [], [], []
        for expert, rows in enumerate(counts.cpu().tolist()):
            if rows:
                part = act[cursor : cursor + rows].cpu().float()
                routed_part = routed[cursor : cursor + rows].cpu().float()
                q13, scale13 = experts[expert][:2]
                exact_gate.append(F.linear(routed_part, effective(q13, scale13)).bfloat16())
                rounded_weight_gate.append(F.linear(routed_part, effective(q13, scale13, True)).bfloat16())
                q2, scale = experts[expert][2:]
                exact_down.append(F.linear(part, effective(q2, scale)).bfloat16())
                rounded_weight_down.append(F.linear(part, effective(q2, scale, True)).bfloat16())
                cursor += rows
        result["cann_down_same_activation"] = {
            "exact_dequant": error(h2.cpu(), torch.cat(exact_down)),
            "bf16_dequant": error(h2.cpu(), torch.cat(rounded_weight_down)),
        }
        result["cann_gate_same_input"] = {
            "exact_dequant": error(h13_before_clip, torch.cat(exact_gate)),
            "bf16_dequant": error(h13_before_clip, torch.cat(rounded_weight_gate)),
        }
        result["npu_peak_allocated_bytes"] = torch.npu.max_memory_allocated(device)
        torch.save(
            dict(
                x=x,
                ids=source_ids,
                routing=routing,
                native=native,
                cann=cann,
                references=references,
                native_repeats=repeat_samples,
            ),
            args.output.with_suffix(".pt"),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
