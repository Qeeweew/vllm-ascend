# SPDX-License-Identifier: Apache-2.0
"""Execute the production scalar planner with host buffers; no NPU claim."""

import ctypes
import random
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def planner(tmp_path_factory):
    if not shutil.which("c++"):
        pytest.skip("A C++ compiler is required")
    directory = tmp_path_factory.mktemp("dspark_schedule")
    (directory / "kernel_operator.h").write_text(
        r"""
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#define __global__
#define __aicore__
#define __gm__
#define REGISTER_TILING_DEFAULT(T)
#define KERNEL_TASK_TYPE_DEFAULT(T)
#define GET_TILING_DATA(data, ptr) auto data = *reinterpret_cast<V41DsparkMetadataTilingData *>(ptr)
using GM_ADDR = void *;
namespace AscendC {
constexpr int EVENT_ID0 = 0;
enum class TPosition {VECCALC};
enum class HardEvent {MTE2_S, V_S, S_MTE3, MTE3_S};
template<HardEvent> void SetFlag(int) {}
template<HardEvent> void WaitFlag(int) {}
template<class T> struct LocalTensor {
    T *ptr;
    T GetValue(uint32_t i) const { return ptr[i]; }
    void SetValue(uint32_t i, T v) { ptr[i] = v; }
};
template<class T> struct GlobalTensor {
    T *ptr;
    void SetGlobalBuffer(T *p) { ptr = p; }
};
template<TPosition> struct TBuf {
    std::vector<int32_t> bytes;
    template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T *>(bytes.data())}; }
};
struct TPipe {
    template<TPosition P> void InitBuffer(TBuf<P> &b, uint32_t bytes) { b.bytes.resize(bytes / 4); }
};
struct DataCopyExtParams { uint32_t count, bytes, source_stride, target_stride, reserved; };
template<class T> struct DataCopyPadExtParams { bool pad; uint8_t left, right; T value; };
template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
                                 DataCopyExtParams p, DataCopyPadExtParams<T>) {
    std::copy_n(src.ptr, p.bytes / sizeof(T), dst.ptr);
}
template<class T> void Duplicate(LocalTensor<T> dst, T value, uint32_t n) {
    std::fill_n(dst.ptr, n, value);
}
template<class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t n) {
    std::copy_n(src.ptr, n, dst.ptr);
}
}
"""
    )
    source = (
        Path(__file__).resolve().parents[3] / "csrc/attention/v41_dspark_metadata/op_kernel/v41_dspark_metadata.cpp"
    )
    library = directory / "planner.so"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", "-I", str(directory), str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = ctypes.CDLL(str(library))
    function = loaded.v41_dspark_metadata
    function.argtypes = [ctypes.c_void_p] * 6
    function.restype = None

    def run(offsets, tokens):
        cu = (ctypes.c_int32 * len(offsets))(*offsets)
        output = (ctypes.c_int32 * 1024)(*([77] * 1024))
        tiling = (ctypes.c_uint32 * 2)(len(offsets) - 1, tokens)
        function(cu, None, None, output, None, tiling)
        return list(output)

    return run


def check_coverage(words, counts):
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    covered, sizes = [], []
    assert words[180:] == [0] * 844
    for core in range(20):
        enabled, bs, ms, ss, be, me, se, fd, max_s2 = words[9 * core : 9 * core + 9]
        if not enabled:
            assert words[9 * core : 9 * core + 9] == [0] * 9
            continue
        assert ss == se == fd == max_s2 == 0
        if core == 0:
            assert bs == ms == 0
        stop_batch = be + int(me != 0)
        queries = []
        for batch in range(bs, stop_batch):
            first = ms if batch == bs else 0
            last = me if batch == stop_batch - 1 and me else counts[batch]
            assert 0 <= first <= last <= counts[batch]
            queries.extend(range(offsets[batch] + first, offsets[batch] + last))
        assert queries
        sizes.append(len(queries))
        covered.extend(queries)
    assert covered == list(range(offsets[-1]))
    assert len(sizes) == min(20, offsets[-1])
    if sizes:
        assert max(sizes) - min(sizes) <= 1


def test_production_planner_covers_random_ragged_queries(planner):
    rng = random.Random(410)
    for batch in [0, 1, 2, 4, 8, 16, 32, 128, 4096]:
        for _ in range(20):
            counts = [rng.choice([0, 0, 1, 5, 7]) for _ in range(batch)]
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            check_coverage(planner(offsets, offsets[-1] + 3), counts)


@pytest.mark.parametrize(
    "offsets,tokens", [([0], 0), ([0, 0, 0], 8), ([1, 5], 5), ([0, -1], 8), ([0, 5, 4], 8), ([0, 9], 8)]
)
def test_empty_or_invalid_offsets_disable_all_cores(planner, offsets, tokens):
    assert planner(offsets, tokens) == [0] * 1024
