# SPDX-License-Identifier: Apache-2.0
"""Small host-only NUMA placement/migration probe; never allocates device tensors.

Launch in a fresh process under numactl before torch/driver initialization.
The pinned allocator may initialize an NPU context; coordinate its device with
other users. move_pages only operates on pages inside these small buffers.
"""

import argparse
import ctypes
import errno
import json
import os
from collections import Counter
from pathlib import Path


def query_or_move(pointer, size, target_node=None):
    page_size = os.sysconf("SC_PAGE_SIZE")
    first = (pointer + page_size - 1) // page_size * page_size
    addresses = list(range(first, pointer + size, page_size))
    pages = (ctypes.c_void_p * len(addresses))(*addresses)
    status = (ctypes.c_int * len(addresses))(*([-999] * len(addresses)))
    nodes = None if target_node is None else (ctypes.c_int * len(addresses))(*([target_node] * len(addresses)))
    numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
    numa.move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    numa.move_pages.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = numa.move_pages(0, len(addresses), pages, nodes, status, 0 if nodes is None else 2)
    error = ctypes.get_errno() if result < 0 else 0
    counts = Counter(int(value) for value in status)
    return {
        "return": result,
        "errno": error,
        "error": os.strerror(error) if error else None,
        "pages": len(addresses),
        "page_size": page_size,
        "status_counts": {str(k): v for k, v in counts.items()},
        "status_errors": {str(k): errno.errorcode.get(-k, "unknown") for k in counts if k < 0},
    }


def mapping(pointer):
    for line in Path("/proc/self/maps").read_text().splitlines():
        first, last = [int(value, 16) for value in line.split()[0].split("-")]
        if first <= pointer < last:
            numa = next(
                (
                    row
                    for row in Path("/proc/self/numa_maps").read_text().splitlines()
                    if int(row.split()[0], 16) == first
                ),
                None,
            )
            return {"maps": line, "numa_maps": numa}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--mib", type=int, default=4)
    parser.add_argument("--target-node", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.mib <= 16:
        parser.error("Keep this diagnostic bounded to 1–16 MiB per buffer")

    # Worker-only imports: launch mempolicy must precede allocator initialization.
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    result = {
        "pid": os.getpid(),
        "device": args.device,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "kernel": os.uname().release,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "bytes_per_buffer": args.mib * 1024**2,
        "target_node": args.target_node,
        "buffers": [],
    }
    buffers = []
    for pinned in (False, True):
        tensor = torch.empty(result["bytes_per_buffer"], dtype=torch.uint8, pin_memory=pinned)
        tensor.fill_(37)
        buffers.append(tensor)
        pointer = tensor.data_ptr()
        record = {
            "requested_pinned": pinned,
            "is_pinned": tensor.is_pinned(),
            "pointer": pointer,
            "before": query_or_move(pointer, tensor.numel()),
            "mapping_before": mapping(pointer),
        }
        record["migration"] = query_or_move(pointer, tensor.numel(), args.target_node)
        record["after"] = query_or_move(pointer, tensor.numel())
        record["mapping_after"] = mapping(pointer)
        record["contents_preserved"] = bool(torch.all(tensor == 37))
        result["buffers"].append(record)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
