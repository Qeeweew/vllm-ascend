# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import json
import resource
import time

import torch
import torch_npu  # noqa: F401

torch.set_num_threads(8)
torch.npu.set_device(0)
results = []
for gib in (1, 8, 46):
    start = time.perf_counter()
    try:
        weight = torch.empty(gib * 1024**3 // 2, dtype=torch.bfloat16, pin_memory=True)
        allocated = time.perf_counter()
        weight.zero_()
        touched = time.perf_counter()
        table = weight.view(-1, 256)
        indices = torch.tensor([0, table.shape[0] // 2, table.shape[0] - 1], dtype=torch.int64)
        for i, index in enumerate(indices):
            table[index].fill_(i + 1)
        staging = torch.empty((3, 256), dtype=torch.bfloat16, pin_memory=True)
        torch.index_select(table, 0, indices, out=staging)
        rows = staging.npu(non_blocking=True)
        torch.npu.synchronize()
        results.append(
            {
                "GiB": gib,
                "pinned": weight.is_pinned(),
                "allocate_s": allocated - start,
                "touch_s": touched - allocated,
                "gather_h2d_correct": bool((rows.cpu() == torch.tensor([1, 2, 3])[:, None]).all()),
                "maxrss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        del weight, rows, table, staging
        gc.collect()
    except Exception as error:
        results.append({"GiB": gib, "error": str(error)})
        print(json.dumps(results[-1]), flush=True)
        raise
    print(json.dumps(results[-1]), flush=True)
