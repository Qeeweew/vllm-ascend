// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline uint32_t ScalarMin(uint32_t a, uint32_t b) { return a < b ? a : b; }
struct V41CacheMetadataTilingData {
    uint32_t batch, tokens, inputColumns, outputColumns, logicalBlock, physicalBlock, ratio, compressed, cores;
};

// Scalar consumers use explicit MTE2->S events; queue MTE2->V dependencies
// would not protect GetValue. Every GM read is a bounded DMA into UB.
template<typename T>
__aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    if (count) DataCopyPad(dst, src, DataCopyExtParams{1, count * uint32_t(sizeof(T)), 0, 0, 0},
                           DataCopyPadExtParams<T>{false, 0, 0, 0});
}
template<typename T>
__aicore__ inline void Store(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count)
{
    if (count) DataCopyPad(dst, src, DataCopyExtParams{1, count * uint32_t(sizeof(T)), 0, 0, 0});
}
template<HardEvent Event> __aicore__ inline void Fence()
{
    SetFlag<Event>(EVENT_ID0); WaitFlag<Event>(EVENT_ID0);
}

extern "C" __global__ __aicore__ void v41_cache_metadata(
    GM_ADDR positionsIn, GM_ADDR cuIn, GM_ADDR lengthsIn, GM_ADDR tableIn,
    GM_ADDR positionsOut, GM_ADDR cuOut, GM_ADDR lengthsOut, GM_ADDR tableOut,
    GM_ADDR requestsOut, GM_ADDR slotsOut, GM_ADDR cmpOut, GM_ADDR residualOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(V41CacheMetadataTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(d, tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> cuBuf, tableBuf, scalarBuf, copyBuf;
    pipe.InitBuffer(cuBuf, ((d.batch + 8) / 8) * 32);
    pipe.InitBuffer(tableBuf, 16384);
    pipe.InitBuffer(scalarBuf, 128);
    pipe.InitBuffer(copyBuf, 16384);
    auto cu = cuBuf.Get<int32_t>(); auto table = tableBuf.Get<int32_t>();
    auto pos = scalarBuf.Get<int64_t>(); auto slot = pos[4];
    auto req = scalarBuf.Get<int32_t>()[16];
    auto lengths = copyBuf.Get<int32_t>();
    GlobalTensor<int64_t> pi, po, so;
    GlobalTensor<int32_t> ci, co, li, lo, ti, to, ro, cm, re;
    pi.SetGlobalBuffer((__gm__ int64_t *)positionsIn); po.SetGlobalBuffer((__gm__ int64_t *)positionsOut);
    so.SetGlobalBuffer((__gm__ int64_t *)slotsOut); ci.SetGlobalBuffer((__gm__ int32_t *)cuIn);
    co.SetGlobalBuffer((__gm__ int32_t *)cuOut); li.SetGlobalBuffer((__gm__ int32_t *)lengthsIn);
    lo.SetGlobalBuffer((__gm__ int32_t *)lengthsOut); ti.SetGlobalBuffer((__gm__ int32_t *)tableIn);
    to.SetGlobalBuffer((__gm__ int32_t *)tableOut); ro.SetGlobalBuffer((__gm__ int32_t *)requestsOut);
    cm.SetGlobalBuffer((__gm__ int32_t *)cmpOut); re.SetGlobalBuffer((__gm__ int32_t *)residualOut);
    const uint32_t core = GetBlockIdx();
    Load(cu, ci, d.batch + 1);
    const bool cachedTable = uint64_t(d.batch) * d.inputColumns <= 4096;
    if (cachedTable) Load(table, ti, d.batch * d.inputColumns);
    Fence<HardEvent::MTE2_S>();
    if (core == 0) {
        Fence<HardEvent::S_MTE3>();
        Store(co, cu, d.batch + 1);
        Load(lengths, li, d.batch);
        Fence<HardEvent::MTE2_MTE3>();
        Store(lo, lengths, d.batch);
        Fence<HardEvent::MTE3_S>();
        for (uint32_t i = 0; i < d.batch; ++i) {
            const int32_t value = lengths.GetValue(i);
            // Match torch floor division/remainder even for sentinel negatives.
            lengths.SetValue(i, value / int32_t(d.ratio) - int32_t(value < 0 && value % int32_t(d.ratio)));
        }
        Fence<HardEvent::S_MTE3>(); Store(cm, lengths, d.batch);
        Fence<HardEvent::MTE3_MTE2>(); Load(lengths, li, d.batch);
        Fence<HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < d.batch; ++i) {
            int32_t value = lengths.GetValue(i) % int32_t(d.ratio);
            lengths.SetValue(i, value < 0 ? value + d.ratio : value);
        }
        Fence<HardEvent::S_MTE3>(); Store(re, lengths, d.batch);
        Fence<HardEvent::MTE3_S>();
    }
    // Each token has a distinct owner. Small decode batches retain one AIV per
    // token; larger buckets distribute tokens round-robin over all available AIVs.
    for (uint32_t token = core; token < d.tokens; token += d.cores) {
        Load(pos, pi[token], 1); Fence<HardEvent::MTE2_S>();
        const int64_t position = pos.GetValue(0);
        int32_t request = 0;
        int64_t physicalSlot = -1;
        if (d.batch) {
            uint32_t low = 0, high = d.batch;
            while (low < high) {
                const uint32_t mid = (low + high) / 2;
                if (cu.GetValue(mid + 1) <= int32_t(token)) low = mid + 1;
                else high = mid;
            }
            request = int32_t(low);
            const bool valid = int32_t(token) < cu.GetValue(d.batch) && position >= 0;
            if (!valid) request = -1;
            const int64_t page = position / int64_t(d.logicalBlock);
            if (valid && (!d.compressed || (position + 1) % d.ratio == 0) &&
                page < d.inputColumns && request >= 0 && uint32_t(request) < d.batch) {
                const uint64_t index = uint64_t(request) * d.inputColumns + page;
                int32_t physical;
                if (cachedTable) physical = table.GetValue(index);
                else {
                    Load(table, ti[index], 1); Fence<HardEvent::MTE2_S>();
                    physical = table.GetValue(0);
                }
                if (physical >= 0) physicalSlot = int64_t(physical) * d.physicalBlock +
                                               position % d.logicalBlock / d.ratio;
            }
        }
        req.SetValue(0, request); slot.SetValue(0, physicalSlot);
        Fence<HardEvent::S_MTE3>();
        Store(po[token], pos, 1); Store(ro[token], req, 1); Store(so[token], slot, 1);
        Fence<HardEvent::MTE3_S>();
    }
    // Full output rows are refreshed, including the unused right-hand columns.
    // Never read tableOut while another core is writing it: slot lookup uses ti.
    for (uint32_t row = core; row < d.batch; row += d.cores) {
        for (uint32_t start = 0; start < d.outputColumns; start += 4096) {
            const uint32_t count = ScalarMin(uint32_t(4096), d.outputColumns - start);
            Duplicate(lengths, int32_t(-1), (count + 7) / 8 * 8);
            Fence<HardEvent::V_MTE2>();
            if (start < d.inputColumns) {
                Load(lengths, ti[uint64_t(row) * d.inputColumns + start], ScalarMin(count, d.inputColumns - start));
            }
            Fence<HardEvent::MTE2_S>();
            if (start < d.inputColumns) {
                const uint32_t copied = ScalarMin(count, d.inputColumns - start);
                for (uint32_t i = copied; i < ScalarMin(count, (copied + 7) / 8 * 8); ++i) lengths.SetValue(i, -1);
            }
            Fence<HardEvent::S_MTE3>();
            Store(to[uint64_t(row) * d.outputColumns + start], lengths, count);
            Fence<HardEvent::MTE3_V>();
        }
    }
}
