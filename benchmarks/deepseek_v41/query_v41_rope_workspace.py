# SPDX-License-Identifier: Apache-2.0
"""Query exact generated ACLNN scratch, then launch with retained ownership."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def measure(library, name, tensors, attrs, attr_types):
    create = library.aclCreateTensor
    create.argtypes = [
        ct.POINTER(ct.c_int64),
        ct.c_uint64,
        ct.c_int,
        ct.POINTER(ct.c_int64),
        ct.c_int64,
        ct.c_int,
        ct.POINTER(ct.c_int64),
        ct.c_uint64,
        ct.c_void_p,
    ]
    create.restype = ct.c_void_p
    destroy = library.aclDestroyTensor
    destroy.argtypes = [ct.c_void_p]
    dtypes = {torch.float32: 0, torch.float16: 1, torch.int8: 2, torch.int64: 9, torch.bfloat16: 27}
    descriptors = []
    for tensor in tensors:
        shape = (ct.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (ct.c_int64 * tensor.ndim)(*tensor.stride())
        storage = (ct.c_int64 * 1)(tensor.untyped_storage().nbytes() // tensor.element_size())
        descriptor = create(
            shape,
            tensor.ndim,
            dtypes[tensor.dtype],
            stride,
            tensor.storage_offset(),
            2,
            storage,
            1,
            tensor.untyped_storage().data_ptr(),
        )
        assert descriptor
        descriptors.append(descriptor)
    query = getattr(library, name + "GetWorkspaceSize")
    query.argtypes = [ct.c_void_p] * len(tensors) + attr_types + [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_void_p)]
    query.restype = ct.c_int
    launch = getattr(library, name)
    launch.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
    launch.restype = ct.c_int
    workspace, executor = ct.c_uint64(), ct.c_void_p()
    try:
        status = query(*descriptors, *attrs, ct.byref(workspace), ct.byref(executor))
        assert status == 0, f"{name} phase1 status={status}"
        scratch = torch.empty(workspace.value, dtype=torch.uint8, device=tensors[0].device)
        torch.npu.synchronize()
        status = launch(scratch.data_ptr(), workspace.value, executor, torch.npu.current_stream().npu_stream)
        assert status == 0, f"{name} phase2 status={status}"
        torch.npu.synchronize()
        return workspace.value
    finally:
        for descriptor in descriptors:
            destroy(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opapi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    torch.npu.set_device(0)
    library = ct.CDLL(str(args.opapi.resolve()))
    positions = torch.tensor([1], dtype=torch.int64, device="npu")
    slots = torch.tensor([0], dtype=torch.int64, device="npu")
    cos = torch.ones((4, 32), dtype=torch.float32, device="npu")
    sin = torch.zeros_like(cos)
    x = torch.ones((1, 512), dtype=torch.bfloat16, device="npu")
    key = torch.ones((1, 128), dtype=torch.bfloat16, device="npu")
    output = torch.empty_like(x)
    main_stride = 32 * 512 + 7
    main_raw = torch.zeros(2 * main_stride + 16, dtype=torch.bfloat16, device="npu")
    cache = main_raw.as_strided((2, 32, 1, 512), (main_stride, 512, 512, 1), 5)
    index_stride = 32 * 130 + 18
    index_raw = torch.zeros(2 * index_stride + 16, dtype=torch.uint8, device="npu")
    key_cache = index_raw.view(torch.int8).as_strided((2, 32, 1, 128), (index_stride, 128, 128, 1), 6)
    scales = index_raw.view(torch.float16).as_strided((2, 32, 1), (index_stride // 2, 1, 1), (6 + 32 * 128) // 2)
    calls = (
        ("aclnnV41Rope", (x, positions, cos, sin, output), (False,), [ct.c_bool]),
        ("aclnnV41MainCacheStore", (x, positions, slots, cos, sin, cache), (1, cache.stride(0)), [ct.c_int64] * 2),
        (
            "aclnnV41IndexCacheStore",
            (key, positions, slots, cos, sin, key_cache, scales),
            (1, key_cache.stride(0), scales.stride(0)),
            [ct.c_int64] * 3,
        ),
    )
    report = {
        "opapi": str(args.opapi.resolve()),
        "opapi_sha256": hashlib.sha256(args.opapi.read_bytes()).hexdigest(),
        "workspace_bytes": {},
        "cache_layout": "gapped main pages and shared packed index pages; nonzero storage offsets",
    }
    for name, tensors, attrs, types in calls:
        report["workspace_bytes"][name] = measure(library, name, tensors, attrs, types)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
