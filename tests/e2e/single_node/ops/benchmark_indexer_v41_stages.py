# SPDX-License-Identifier: Apache-2.0
"""Reproduce the 910B INT32 versus exact FP32 index-ordering bottleneck."""

import argparse
import json
from pathlib import Path

import torch
from benchmark_indexer_v41 import AscendIndexerV41Ops, measure, summarize
from test_indexer_v41 import build_metadata, device_case, make_case, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    report = []
    for batch in (1, 8, 32):
        case = device_case(make_case(1, [1] * batch, [32771] * batch))
        ops = AscendIndexerV41Ops(1, "source")
        meta = build_metadata(ops, case)

        def raw(case=case, meta=meta):
            return torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
                case["q"],
                case["k"],
                case["w"],
                case["qs"],
                case["ks"],
                512,
                2,
                cu_seqlens_q=meta.cu_seqlens_q,
                seqused_k=meta.seqused_k,
                block_table=meta.block_table,
                metadata=meta.qli_metadata,
                candidate_mode=1,
            )

        indices, _, _ = raw()
        functions = {
            "raw": raw,
            "wrapper": lambda ops=ops, case=case, meta=meta: select(ops, case, meta),
            "int32_sort": lambda indices=indices: indices.sort(-1).values,
            "float32_sort": lambda indices=indices: indices.float().sort(-1).values.int(),
            "int32_topk": lambda indices=indices: indices.topk(512, dim=-1, largest=False).values,
        }
        for name, run in functions.items():
            for _ in range(3):
                run()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(64):
                    output = run()
            samples = [measure(graph.replay, 64) for _ in range(3)]
            record = dict(batch=batch, stage=name, **summarize(samples))
            report.append(record)
            print(batch, name, record["median_us"], flush=True)
            del graph, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
