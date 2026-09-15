import argparse
import json
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    torch.npu.set_device(options.device)
    torch.set_num_threads(8)
    torch.manual_seed(41)
    results = []
    for k, n in [(5120, 576), (288, 5120)]:
        e = 6
        q = torch.randint(-8, 8, (e, k, n), dtype=torch.int32)
        scale = (torch.rand(e, k // 32, n) * 0.015 + 0.001).bfloat16()
        scale[..., ::2].neg_()
        q[:, 0, 0] = -8
        qn = q.npu()
        packed = torch_npu.npu_convert_weight_to_int4pack(qn.flatten(0, 1)).view(e, k, n // 8)
        sn = scale.npu()
        zn = torch.zeros_like(sn)
        dense = (q.float() * scale.repeat_interleave(32, 1).float()).bfloat16()
        for m in [1, 16, 128]:
            x = (torch.randn(e * m, k) * 0.2).bfloat16()
            xn = x.npu()
            counts = torch.full((e,), m, dtype=torch.int64, device="npu")

            def run(xn=xn, packed=packed, sn=sn, zn=zn, counts=counts):
                return torch_npu.npu_grouped_matmul(
                    x=[xn],
                    weight=[packed],
                    antiquant_scale=[sn],
                    antiquant_offset=[zn],
                    group_list=counts,
                    split_item=2,
                    group_list_type=1,
                    group_type=0,
                    output_dtype=torch.bfloat16,
                )[0]

            try:
                y = run()
                torch.npu.synchronize()
                ref = torch.cat([x[i * m : (i + 1) * m].float() @ dense[i].float() for i in range(e)])
                diff = y.cpu().float() - ref
                measures = []
                for _ in range(5):
                    run()
                for _ in range(5):
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        run()
                    end.record()
                    end.synchronize()
                    measures.append(start.elapsed_time(end) * 1000 / 20)
                rec = dict(
                    k=k,
                    n=n,
                    experts=e,
                    rows_per_expert=m,
                    nrmse=(diff.norm() / ref.norm()).item(),
                    max_abs=diff.abs().max().item(),
                    us=measures,
                )
            except Exception as ex:
                rec = dict(k=k, n=n, rows_per_expert=m, error=str(ex))
            print(json.dumps(rec), flush=True)
            results.append(rec)
    options.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
