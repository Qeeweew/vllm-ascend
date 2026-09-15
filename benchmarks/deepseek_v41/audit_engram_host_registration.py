# SPDX-License-Identifier: Apache-2.0
"""Probe small NUMA-placed host registration and explicit host-NUMA allocation.

No device tensors, DMA, graph, or kernels. Run under numactl in a fresh process;
all registrations and physical handles belong only to this diagnostic process.
"""

import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path

from audit_engram_numa import mapping, query_or_move


class Location(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_int)]


class PointerAttributes(ctypes.Structure):
    _fields_ = [("location", Location), ("page_size", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 4)]


class PhysicalProperties(ctypes.Structure):
    _fields_ = [
        ("handle_type", ctypes.c_int),
        ("allocation_type", ctypes.c_int),
        ("memory_attr", ctypes.c_int),
        ("location", Location),
        ("reserved", ctypes.c_uint64),
    ]


def bind(library, name, arguments):
    function = getattr(library, name, None)
    if function is not None:
        function.argtypes = arguments
        function.restype = ctypes.c_int
    return function


def pointer_attributes(function, pointer):
    attributes = PointerAttributes()
    status = function(pointer, ctypes.byref(attributes))
    return {
        "return": status,
        "location_type": attributes.location.type,
        "location_id": attributes.location.id,
        "page_size": attributes.page_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--allocation-node", type=int, default=0)
    parser.add_argument("--target-node", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    acl = ctypes.CDLL("libascendcl.so")
    register_v2 = bind(acl, "aclrtHostRegisterV2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32])
    register = bind(
        acl, "aclrtHostRegister", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    )
    unregister = bind(acl, "aclrtHostUnregister", [ctypes.c_void_p])
    attributes = bind(acl, "aclrtPointerGetAttributes", [ctypes.c_void_p, ctypes.POINTER(PointerAttributes)])
    result = {
        "torch_npu": torch_npu.__version__,
        "device": args.device,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "allocation_node": args.allocation_node,
        "registration": [],
        "physical": [],
    }
    size = 4 * 1024**2
    for mode, flag in (("v2_pinned", 0x10000000), ("v2_mapped", 2), ("legacy_mapped", 0)):
        storage = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        # Apply a per-VMA policy after driver init, before first touch; this
        # works without changing other worker threads or global affinity.
        pointer = ctypes.addressof(ctypes.c_char.from_buffer(storage))
        numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
        numa.mbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        numa.mbind.restype = ctypes.c_long
        node_mask = ctypes.c_ulong(1 << args.allocation_node)
        mbind_status = numa.mbind(pointer, size, 2, ctypes.byref(node_mask), ctypes.sizeof(node_mask) * 8, 0)
        if mbind_status != 0:
            raise OSError(ctypes.get_errno(), "Per-VMA mbind failed")
        tensor = torch.frombuffer(storage, dtype=torch.uint8)
        tensor.fill_(39)
        pointer = tensor.data_ptr()
        row = {
            "mode": mode,
            "bytes": size,
            "before": query_or_move(pointer, size),
            "mapping_before": mapping(pointer),
            "is_pinned_before": tensor.is_pinned(),
            "mbind_return": mbind_status,
        }
        device_pointer = ctypes.c_void_p()
        status = (
            register(pointer, size, flag, ctypes.byref(device_pointer))
            if mode == "legacy_mapped"
            else register_v2(pointer, size, flag)
        )
        row["register_return"] = status
        try:
            row["attributes"] = pointer_attributes(attributes, pointer)
            row["is_pinned_after"] = tensor.is_pinned()
            row["query_after_register"] = query_or_move(pointer, size)
            row["mapping_after_register"] = mapping(pointer)
            if status == 0:
                row["move_while_registered"] = query_or_move(pointer, size, args.target_node)
                row["query_after_move"] = query_or_move(pointer, size)
            row["contents_preserved"] = bool(torch.all(tensor == 39))
        finally:
            if status == 0:
                row["unregister_return"] = unregister(pointer)
                if row["unregister_return"] != 0:
                    result["registration"].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    raise RuntimeError("Registration cleanup failed; retaining mapped storage until process exit")
                row["is_pinned_after_unregister"] = tensor.is_pinned()
                row["move_after_unregister"] = query_or_move(pointer, size, args.target_node)
                row["query_after_unregister_move"] = query_or_move(pointer, size)
            del tensor
            storage.close()
        result["registration"].append(row)

    granularity = bind(
        acl,
        "aclrtMemGetAllocationGranularity",
        [ctypes.POINTER(PhysicalProperties), ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)],
    )
    allocate = bind(
        acl,
        "aclrtMallocPhysical",
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.POINTER(PhysicalProperties), ctypes.c_uint64],
    )
    free = bind(acl, "aclrtFreePhysical", [ctypes.c_void_p])
    get_properties = bind(
        acl, "aclrtMemGetAllocationPropertiesFromHandle", [ctypes.c_void_p, ctypes.POINTER(PhysicalProperties)]
    )
    for node in (0, 2):
        # PINNED allocation, DDR normal pages, explicit HOST_NUMA location.
        properties = PhysicalProperties(0, 0, 3, Location(node, 4), 0)
        unit = ctypes.c_size_t()
        row = {
            "node": node,
            "granularity_return": granularity(ctypes.byref(properties), 0, ctypes.byref(unit)),
            "granularity": unit.value,
        }
        if row["granularity_return"] == 0 and 0 < unit.value <= 4 * 1024**2:
            handle = ctypes.c_void_p()
            row["allocate_return"] = allocate(ctypes.byref(handle), unit.value, ctypes.byref(properties), 0)
            if row["allocate_return"] == 0:
                try:
                    actual = PhysicalProperties()
                    row["get_properties_return"] = get_properties(handle, ctypes.byref(actual))
                    row["actual_location_type"] = actual.location.type if row["get_properties_return"] == 0 else None
                    row["actual_location_id"] = actual.location.id if row["get_properties_return"] == 0 else None
                finally:
                    row["free_return"] = free(handle)
                    if row["free_return"] != 0:
                        raise RuntimeError("Physical handle cleanup failed")
        result["physical"].append(row)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
