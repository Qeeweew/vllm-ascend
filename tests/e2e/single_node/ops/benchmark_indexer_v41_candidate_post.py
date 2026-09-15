# SPDX-License-Identifier: Apache-2.0
"""Separate validity comparison, masking, remapping, and final sorting costs."""

import argparse
import json
from pathlib import Path

import torch
from benchmark_indexer_v41 import measure, summarize
from test_indexer_v41 import build_metadata, device_case, make_case, select

from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Ops
from vllm_ascend.ops.indexer_v41_candidate import CandidateIndexerB1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    case = device_case(make_case(1, [1], [32771]))
    source = AscendIndexerV41Ops(1, "source")
    metadata = build_metadata(source, case, max_q=1, max_k=32771)
    _, candidates = select(source, case, metadata)
    selector = CandidateIndexerB1(32771, "npu:0")
    selector(case["q"], case["k"], case["w"], case["qs"], case["ks"], metadata, candidates)
    values, indices = selector.scores.topk(512, sorted=False)
    positions = selector.positions
    positions_float = positions.float()
    mapped = positions[None, :].gather(1, indices)
    mapped_float = mapped.float()
    valid = values > -torch.inf
    negative_infinity = torch.full_like(values, -torch.inf)
    masked = torch.where(values > -torch.inf, mapped, 2**24)
    ordered = masked.float().sort(-1).values

    def legacy_int32_pipeline():
        values, indices = selector.scores.topk(512, sorted=False)
        original = selector.positions[None, :].gather(1, indices)
        ordered = torch.where(values > -torch.inf, original, 2**24).float().sort(-1).values
        return torch.where(ordered < 2**24, ordered, -1).int()

    def fp32_score_mask_pipeline():
        values, indices = selector.scores.topk(512, sorted=False)
        original = selector.positions[None, :].gather(1, indices)
        ordered = torch.where(values > -torch.inf, original.float(), float(2**24)).sort(-1).values
        return torch.where(ordered < 2**24, ordered, -1).int()

    assert torch.equal(legacy_int32_pipeline(), selector.select_scores())
    runs = {
        "topk": lambda: selector.scores.topk(512, sorted=False),
        "gather_i32": lambda: positions[None, :].gather(1, indices),
        "index_i32": lambda: positions[indices],
        "index_select_i32": lambda: torch.index_select(positions, 0, indices.flatten()),
        "gather_f32": lambda: positions_float[None, :].gather(1, indices),
        "cast_gather_f32": lambda: positions.float()[None, :].gather(1, indices),
        "compare_negative_inf_scalar": lambda: values > -torch.inf,
        "compare_negative_inf_tensor": lambda: values > negative_infinity,
        "compare_finite_scalar_diagnostic": lambda: values > torch.finfo(torch.float32).min,
        "compare_position_validity": lambda: mapped >= 0,
        "where_i32_precomputed_condition": lambda: torch.where(valid, mapped, 2**24),
        "where_f32_precomputed_condition": lambda: torch.where(valid, mapped_float, float(2**24)),
        "mask_i32": lambda: torch.where(values > -torch.inf, mapped, 2**24),
        "cast_mask_f32": lambda: torch.where(values > -torch.inf, mapped.float(), float(2**24)),
        "float_sort": lambda: masked.float().sort(-1).values,
        "restore_int": lambda: torch.where(ordered < 2**24, ordered, -1).int(),
        "legacy_int32_pipeline": legacy_int32_pipeline,
        "fp32_score_mask_pipeline": fp32_score_mask_pipeline,
        "position_mask_pipeline": selector.select_scores,
    }
    graphs, outputs, samples = {}, {}, {}
    for name, run in runs.items():
        for _ in range(3):
            run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(64):
                output = run()
        graphs[name], outputs[name], samples[name] = graph, output, []
    for round_id in range(3):
        names = list(runs) if round_id % 2 == 0 else list(reversed(runs))
        for name in names:
            samples[name].append(measure(graphs[name].replay, 64))
    statistics = {name: summarize(sample) for name, sample in samples.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in row.items() if k != "samples_us"} for name, row in statistics.items()}))


if __name__ == "__main__":
    main()
