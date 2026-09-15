# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit NUMA placement for Engram host tensors; opt-in and outside graphs.

A private anonymous mapping gets a per-range NUMA policy before first touch,
then CANN MAPPED registration. Existing torch pinned allocators are unchanged.
Owners must record DMA completion events and explicitly close after use.
"""

import ctypes
import math
import mmap
import os
import warnings
from collections.abc import Callable, Sequence

import torch
import torch_npu  # noqa: F401

_MPOL_BIND = 2
_ACL_HOST_REG_MAPPED = 0x2


class _HostMemoryAPI:
    def __init__(self):
        self.numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
        self.numa.mbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        self.numa.mbind.restype = ctypes.c_long
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        self.acl.aclrtHostRegisterV2.restype = ctypes.c_int
        self.acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.acl.aclrtHostUnregister.restype = ctypes.c_int

    def bind(self, pointer: int, size: int, node: int) -> None:
        word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        # Supply a full mask word, including for node zero: maxnode=1 is
        # rejected by this host kernel's NUMA syscall mask convention.
        words = node // word_bits + 1
        mask = (ctypes.c_ulong * words)()
        mask[node // word_bits] = 1 << (node % word_bits)
        status = self.numa.mbind(pointer, size, _MPOL_BIND, mask, words * word_bits, 0)
        if status != 0:
            error = ctypes.get_errno()
            raise OSError(error, f"Engram mbind to NUMA node {node} failed: {os.strerror(error)}")

    def register(self, pointer: int, size: int) -> None:
        status = self.acl.aclrtHostRegisterV2(pointer, size, _ACL_HOST_REG_MAPPED)
        if status != 0:
            raise RuntimeError(f"Engram MAPPED host registration failed with CANN status {status}")

    def unregister(self, pointer: int) -> None:
        status = self.acl.aclrtHostUnregister(pointer)
        if status != 0:
            raise RuntimeError(f"Engram host unregister failed with CANN status {status}")


class _RegisteredMapping(mmap.mmap):
    """The tensor's frombuffer storage retains this object, including for views."""

    def __new__(cls, size, api, device):
        return super().__new__(cls, -1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)

    def __init__(self, size, api, device):
        self.api = api
        self.device = device
        self.pointer = ctypes.addressof(ctypes.c_char.from_buffer(self))
        self.registered = False
        self.events = []

    def release_registration(self):
        if not self.registered:
            return
        # Do not clear events or release storage when either operation fails;
        # the explicit owner can retry close after handling the error.
        for event in self.events:
            event.synchronize()
        with torch.npu.device(self.device):
            self.api.unregister(self.pointer)
        self.registered = False
        self.events.clear()

    def __del__(self):
        # Explicit close is required for checked cleanup. This fallback also
        # keeps registrations alive if a tensor outlives an unclosed owner.
        try:
            self.release_registration()
        except Exception as error:
            warnings.warn(f"Engram registered host cleanup failed: {error}", ResourceWarning, stacklevel=2)


class EngramPinnedHostTensor:
    """Own a NUMA-placed CPU BF16 tensor and its CANN registration.

    This optional helper does not select nodes or replace the default allocator.
    The caller supplies the worker's intended NUMA node and initialized NPU.
    Optional ``initialize`` must populate every element before registration;
    otherwise the tensor is zero-filled to establish first-touch placement.
    Register each DMA completion event with ``record_event`` after submission.
    ``close`` waits those events, unregisters, and drops the owner's references.
    Tensor aliases remain valid CPU storage afterward, but are no longer pinned
    and must not be used for new DMA. All methods are outside graph capture.
    """

    def __init__(
        self,
        shape: Sequence[int],
        *,
        numa_node: int,
        device: torch.device,
        initialize: Callable[[torch.Tensor], None] | None = None,
    ):
        shape = tuple(shape)
        device = torch.device(device)
        if not shape or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape):
            raise ValueError("Engram host tensor dimensions must be positive integers")
        if not isinstance(numa_node, int) or numa_node < 0:
            raise ValueError("Engram host NUMA node must be a nonnegative integer")
        if device.type != "npu" or device.index is None:
            raise ValueError("Engram registration requires an explicit initialized NPU device")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Engram host allocation must remain outside graph capture")
        self.numa_node = numa_node
        self.device = device
        self._tensor = None
        self._mapping = None
        count = math.prod(shape)
        size = count * torch.bfloat16.itemsize
        page_size = os.sysconf("SC_PAGE_SIZE")
        allocation_size = (size + page_size - 1) // page_size * page_size
        api = _HostMemoryAPI()
        storage = _RegisteredMapping(allocation_size, api, device)
        try:
            api.bind(storage.pointer, allocation_size, numa_node)
            # VMA policy governs first-touch even when torch uses worker threads.
            tensor = torch.frombuffer(storage, dtype=torch.bfloat16, count=count).view(shape)
            if initialize is None:
                tensor.zero_()
            else:
                # A streaming checkpoint loader can populate the final table
                # here, avoiding a redundant zero-fill of tens of GiB.
                initialize(tensor)
            with torch.npu.device(device):
                api.register(storage.pointer, allocation_size)
                storage.registered = True
                if not tensor.is_pinned():
                    raise RuntimeError("Engram MAPPED host registration is not recognized as pinned by torch")
        except Exception:
            storage.release_registration()
            raise
        self._mapping = storage
        self._tensor = tensor

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("Engram pinned host owner is closed")
        return self._tensor

    def record_event(self, event) -> None:
        if self._mapping is None:
            raise RuntimeError("Engram pinned host owner is closed")
        if not any(previous is event for previous in self._mapping.events):
            self._mapping.events.append(event)

    def close(self) -> None:
        if self._mapping is None:
            return
        self._mapping.release_registration()
        self._tensor = None
        # Never call mmap.close here: tensor aliases may still own its storage.
        self._mapping = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
