# SPDX-License-Identifier: Apache-2.0
"""B1 candidate experiment: frozen per-shape gates plus stage timings."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from benchmark_indexer_v41 import dense_baseline, measure, summarize
from indexer_v41_candidate_reference import assert_candidate_selection, candidate_reference
from test_indexer_v41 import build_metadata, device_case, make_case, select

from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Ops
from vllm_ascend.ops.indexer_v41_candidate import CandidateIndexerB1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4097, 32771, 131075])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stages", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    torch.npu.matmul.allow_hf32 = False
    frozen_path = Path(__file__).parents[4] / "benchmarks/deepseek_v41/indexer_v41/candidate_frozen_baseline.json"
    frozen = json.loads(frozen_path.read_text())
    report = dict(
        cases=[],
        frozen_baseline=str(frozen_path),
        gate=frozen["gate"],
        product_default_changed=False,
        environment=dict(
            device=torch.npu.get_device_name(args.device),
            torch=str(torch.__version__),
            torch_npu=str(torch_npu.__version__),
            rounds=3,
            event_samples_per_round=12,
            graph_unroll=64,
            dense_graph_unroll=4,
        ),
    )
    for length in args.lengths:
        case = make_case(1, [1], [length])
        device = device_case(case)
        source = AscendIndexerV41Ops(1, "source")
        metadata = build_metadata(source, device, max_q=1, max_k=length)
        _, candidates = select(source, device, metadata)
        native = AscendIndexerV41Ops(1, "consumer")
        candidate = CandidateIndexerB1(length, device["q"].device)

        def run_candidate(candidate=candidate, device=device, metadata=metadata, candidates=candidates):
            return candidate(device["q"], device["k"], device["w"], device["qs"], device["ks"], metadata, candidates)

        def run_native(native=native, device=device, metadata=metadata, candidates=candidates):
            return select(native, device, metadata, candidates)

        dense = dense_baseline(device, "consumer", candidates)
        reference = candidate_reference(case, candidates.cpu())
        original = dict(candidate=run_candidate, native=run_native, dense=dense)
        for run in original.values():
            indices, _ = run()
            assert_candidate_selection(indices.cpu(), reference)
        if args.stages:
            sorted_blocks = candidates.flatten().clamp(-1, candidate.max_blocks).float().sort(descending=True).values
            query = device["q"].bfloat16()

            def prepare(candidate=candidate, candidates=candidates, device=device):
                return (
                    candidates.flatten().clamp(-1, candidate.max_blocks).float().sort(descending=True).values,
                    device["q"].bfloat16(),
                )

            def gather(candidate=candidate, device=device, sorted_blocks=sorted_blocks, metadata=metadata):
                torch.ops._C_ascend.indexer_v41_candidate_gather(
                    device["k"],
                    device["ks"],
                    sorted_blocks,
                    metadata.block_table,
                    metadata.seqused_k,
                    metadata.cu_seqlens_q,
                    candidate.key,
                    candidate.scale,
                    candidate.positions,
                )

            def bmm(query=query, candidate=candidate):
                return torch.bmm(query, candidate.key.transpose(1, 2), out_dtype=torch.float32, out=candidate.qk)

            def score(candidate=candidate, device=device):
                torch.ops._C_ascend.indexer_v41_candidate_score(
                    candidate.qk, device["w"], device["qs"], candidate.scale, candidate.positions, candidate.scores
                )

            original.update(
                prepare=prepare, gather=gather, bmm=bmm, score=score, topk_remap_sort=candidate.select_scores
            )
        graphs, outputs, samples = {}, {}, {}
        for name, run in original.items():
            for _ in range(3):
                run()
            graph = torch.npu.NPUGraph()
            unroll = 4 if name == "dense" else 64
            with torch.npu.graph(graph):
                for _ in range(unroll):
                    output = run()
            graphs[name], outputs[name], samples[name] = graph, output, []
        for round_id in range(3):
            names = list(original) if round_id % 2 == 0 else list(reversed(original))
            for name in names:
                samples[name].append(measure(graphs[name].replay, 4 if name == "dense" else 64))
        statistics = {name: summarize(values) for name, values in samples.items()}
        c, d = statistics["candidate"], statistics["dense"]
        baseline = next(row for row in frozen["cases"] if row["batch"] == 1 and row["length"] == length)["dense"]
        record = dict(
            batch=1,
            length=length,
            gathered_positions=candidate.count,
            **statistics,
            live_latency_gate=c["median_us"] <= d["median_us"] and c["p95_us"] <= d["p95_us"] * 1.05,
            frozen_latency_gate=(c["median_us"] <= baseline["median_us"] and c["p95_us"] <= baseline["p95_us"] * 1.05),
            noise_gate=max(value["spread"] for value in statistics.values()) <= 0.03,
        )
        report["cases"].append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if not isinstance(v, dict)}), flush=True)
        # Keep all closures/tensor owners alive until their graphs are destroyed.
        del graphs, outputs, original, candidate, native, source, dense
        torch.npu.synchronize()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
