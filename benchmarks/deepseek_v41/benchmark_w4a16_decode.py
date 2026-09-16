# SPDX-License-Identifier: Apache-2.0
"""Decode-only comparison with CANN GMM including route gather/reduction.

Run from the repository root. The default fixture uses all 384 experts with
balanced routing; --routing hot selects the same six experts for every token.
Use --routing uniform --route-steps 64 --graph to sample six distinct experts
independently per token and cycle prebuilt routes between graph replays.
Route updates are excluded from timing; both complete MoE paths include routing
or direct expert selection, both projections, activation, and output reduction.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import torch_npu


def make_routes(batch, experts, top_k, routing, steps):
    generator = torch.Generator().manual_seed(4101)
    routes = []
    for step in range(steps):
        if routing == "uniform":
            # Independent uniform samples without replacement within each token.
            ids = torch.rand(batch, experts, generator=generator).argsort(dim=-1)[:, :top_k]
        elif routing == "hot":
            ids = (torch.arange(top_k).expand(batch, -1) + step * top_k) % experts
        else:
            ids = (torch.arange(batch * top_k).reshape(batch, top_k) + step * top_k) % experts
        routes.append(ids.to(torch.int32).contiguous())
    return routes


def measure(fn, iterations, eviction=None, prepare=None):
    # Sustain work long enough for the device frequency to settle.
    for step in range(1000):
        if prepare is not None:
            prepare(step)
        fn()
    torch.npu.synchronize()
    samples, rounds = [], []
    for round_index in range(5):
        round_samples = []
        for step in range(iterations):
            if prepare is not None:
                prepare(round_index * iterations + step)
            if eviction is not None:
                eviction.add_(1)
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            round_samples.append(start.elapsed_time(end) * 1000)
        samples.extend(round_samples)
        rounds.append({"median_us": statistics.median(round_samples), "raw_us": round_samples})
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[int(len(ordered) * 0.95)],
        "raw_us": samples,
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--routing", choices=("spread", "hot", "uniform"), default="spread")
    parser.add_argument(
        "--route-steps", type=int, default=1, help="Prebuilt routes cycled outside timing; 1 fixes routes"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold", action="store_true", help="Touch 512 MiB before each timed call (outside timing)")
    parser.add_argument("--graph", action="store_true", help="Measure graph replay for each complete MoE path")
    options = parser.parse_args()
    if options.route_steps < 1 or min(options.batches) < 1 or options.experts < 6:
        parser.error("route steps and batches must be positive, with at least six experts")
    torch.npu.set_device(options.device)
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "w4a16_reference", root / "tests/e2e/single_node/ops/test_w4a16_moe_kernel.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # Construct the full expert bank once; spread routing touches up to 384
    # different experts. Hot routing selects the same top-six set for every token.
    bank, reference_weights = reference.make_case(max(options.batches), experts=options.experts)
    bank = list(bank)
    host_routes = make_routes(
        max(options.batches), options.experts, reference.TOP_K, options.routing, options.route_steps
    )
    bank[5].copy_(host_routes[0])
    q13, q2 = reference_weights
    expected = reference.moe_reference(bank[0], q13, bank[2], q2, bank[4], bank[5], bank[6], 10.0)
    expected_changed = (
        reference.moe_reference(bank[0], q13, bank[2], q2, bank[4], host_routes[-1], bank[6], 10.0)
        if options.route_steps > 1
        else expected
    )
    del reference_weights, q13, q2
    device_bank = tuple(t.npu() for t in bank)
    device_routes = [ids.npu() for ids in host_routes]
    # 910B3 platform_config declares 192 MiB L2. This read/write sweep exceeds
    # twice that capacity and is completed before the timing start event.
    eviction = torch.zeros(256 * 1024 * 1024, dtype=torch.float16, device="npu") if options.cold else None
    results = []
    for batch in options.batches:
        x, w13, s13, w2, s2, ids, routing = (
            device_bank[0][:batch],
            device_bank[1],
            device_bank[2],
            device_bank[3],
            device_bank[4],
            device_bank[5][:batch],
            device_bank[6][:batch],
        )
        ids.copy_(device_routes[0][:batch])

        def native(x=x, w13=w13, s13=s13, w2=w2, s2=s2, ids=ids, routing=routing):
            return torch.ops._C_ascend.npu_w4a16_moe(x, w13, s13, w2, s2, ids, routing, 10.0)

        def cann(x=x, w13=w13, s13=s13, w2=w2, s2=s2, ids=ids, routing=routing):
            # Match production AllGather routing, SwiGLU and combine operators.
            routed, reverse, counts, _ = torch_npu.npu_moe_init_routing_v2(
                x,
                ids,
                active_num=ids.numel(),
                expert_num=w13.shape[0],
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, w13.shape[0]],
                quant_mode=-1,
            )
            counts = counts.to(torch.int64)
            h13 = torch_npu.npu_grouped_matmul(
                x=[routed],
                weight=[w13],
                antiquant_scale=[s13],
                group_list=counts,
                group_list_type=1,
                group_type=0,
                split_item=2,
                output_dtype=x.dtype,
            )[0]
            gate, up = h13.chunk(2, -1)
            gate.clamp_(max=10)
            up.clamp_(-10, 10)
            act = torch_npu.npu_swiglu(h13)
            h2 = torch_npu.npu_grouped_matmul(
                x=[act],
                weight=[w2],
                antiquant_scale=[s2],
                group_list=counts,
                group_list_type=1,
                group_type=0,
                split_item=2,
                output_dtype=x.dtype,
            )[0]
            return torch_npu.npu_moe_token_unpermute(
                permuted_tokens=h2, sorted_indices=reverse.abs(), probs=routing.to(h2.dtype)
            )

        native_checked, cann_checked = native(), cann()
        reference.assert_accurate(native_checked, expected[:batch])
        reference.assert_accurate(cann_checked, expected[:batch])
        native_error = (native_checked.cpu().float() - expected[:batch].float()).norm()
        cann_error = (cann_checked.cpu().float() - expected[:batch].float()).norm()
        reference_norm = expected[:batch].float().norm().clamp_min(1e-12)
        numerical = {
            "native_nrmse": (native_error / reference_norm).item(),
            "cann_nrmse": (cann_error / reference_norm).item(),
        }

        if options.graph:
            native_graph, cann_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
            for _ in range(3):
                native()
                cann()
            torch.npu.synchronize()
            with torch.npu.graph(native_graph):
                native_output = native()
            with torch.npu.graph(cann_graph):
                cann_output = cann()
            # Keep both outputs alive with their captured memory pools.
            assert native_output.shape == cann_output.shape
            native, cann = native_graph.replay, cann_graph.replay

        ids.copy_(device_routes[-1][:batch])
        if options.graph:
            native()
            cann()
            changed_native, changed_cann = native_output, cann_output
        else:
            changed_native, changed_cann = native(), cann()
        reference.assert_accurate(changed_native, expected_changed[:batch])
        reference.assert_accurate(changed_cann, expected_changed[:batch])

        def prepare(step, ids=ids, batch=batch):
            # Same-stream D2D update precedes the timing event and graph replay.
            # Also copy for fixed routes so update overhead is matched.
            ids.copy_(device_routes[step % len(device_routes)][:batch])

        record = {
            "baseline": "production_cann_allgather",
            "batch": batch,
            "experts": options.experts,
            "routing": options.routing,
            "route_steps": options.route_steps,
            "mean_active_experts": statistics.mean(route[:batch].unique().numel() for route in host_routes),
            "cold": options.cold,
            **numerical,
            "mode": "graph" if options.graph else "eager",
            "native": measure(native, options.iterations, eviction, prepare),
            "cann": measure(cann, options.iterations, eviction, prepare),
        }
        record["speedup"] = record["cann"]["median_us"] / record["native"]["median_us"]
        results.append(record)
        summary = {k: v for k, v in record.items() if k not in ("native", "cann")}
        for path in ("native", "cann"):
            for metric in ("median_us", "p95_us"):
                summary[f"{path}_{metric}"] = record[path][metric]
        print(json.dumps(summary), flush=True)
        options.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
