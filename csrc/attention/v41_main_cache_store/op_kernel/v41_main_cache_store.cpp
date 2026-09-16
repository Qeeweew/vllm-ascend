// SPDX-License-Identifier: Apache-2.0
#if __has_include("../v41_rope/v41_rope_core.h")
#include "../v41_rope/v41_rope_core.h"
#else
#include "../../v41_rope/op_kernel/v41_rope_core.h"
#endif
using namespace AscendC;
struct V41MainCacheStoreTilingData {
    uint32_t tokens, page, blocks, tableRows, cores, ratio;
    uint64_t cacheStride;
};

extern "C" __global__ __aicore__ void v41_main_cache_store(
    GM_ADDR x, GM_ADDR positions, GM_ADDR slots, GM_ADDR cos, GM_ADDR sin, GM_ADDR cache,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(V41MainCacheStoreTilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<bfloat16_t> inputGm, cacheGm;
    GlobalTensor<int64_t> positionsGm, slotsGm;
    GlobalTensor<float> cosGm, sinGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    cacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(cache));
    positionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(positions));
    slotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slots));
    cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cos));
    sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sin));
    TPipe pipe;
    V41RopeCore::Row row;
    row.Init(&pipe);
    for (uint32_t token = GetBlockIdx(); token < data.tokens; token += data.cores) {
        const int64_t position = positionsGm.GetValue(token);
        const int64_t slot = slotsGm.GetValue(token);
        if (position < 0 || slot < 0 || uint64_t(slot) >= uint64_t(data.page) * data.blocks ||
            (data.ratio == 2 && (position & 1) == 0)) continue;
        const int64_t groupPosition = data.ratio == 2 ? position - 1 : position;
        if (groupPosition >= data.tableRows) continue;
        // A slot is already compressed; only decode its physical page/row.
        const uint64_t destination = uint64_t(slot / data.page) * data.cacheStride + (slot % data.page) * 512;
        row.Load(inputGm, uint64_t(token) * 512, 512);
        row.Rotate<false>(cosGm, sinGm, groupPosition, 512);
        row.Store(cacheGm, destination, 512, true);
    }
}
