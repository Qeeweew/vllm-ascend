// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

struct V41DsparkMetadataTilingData {
    uint32_t batch;
    uint32_t tokens;
};

extern "C" __global__ __aicore__ void v41_dspark_metadata(
    GM_ADDR cuQ, GM_ADDR lengths, GM_ADDR topkLengths, GM_ADDR schedule,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(V41DsparkMetadataTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> cuBuffer, outBuffer;
    pipe.InitBuffer(cuBuffer, ((data.batch + 8) / 8) * 32);
    pipe.InitBuffer(outBuffer, 4096);
    auto cu = cuBuffer.Get<int32_t>();
    auto out = outBuffer.Get<int32_t>();
    GlobalTensor<int32_t> cuGm, outGm;
    cuGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cuQ));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(schedule));
    DataCopyPad(cu, cuGm, DataCopyExtParams{1, (data.batch + 1) * 4, 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    Duplicate(out, int32_t(0), 1024);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);

    // One sparse query is one M tile (8 heads), with its entire <=256-key
    // candidate span in one 512-column S2 tile. Only query partitioning is
    // necessary. Empty candidate rows retain the attention wrapper's zero mask.
    const int32_t total = cu.GetValue(data.batch);
    bool valid = cu.GetValue(0) == 0 && total >= 0 && uint32_t(total) <= data.tokens;
    for (uint32_t b = 0; b < data.batch; ++b) {
        valid = valid && cu.GetValue(b) >= 0 && cu.GetValue(b) <= cu.GetValue(b + 1);
    }
    if (valid && total > 0 && data.batch > 0) {
        const uint32_t cores = (uint32_t(total) < 20U ? uint32_t(total) : 20U);
        const uint32_t common = uint32_t(total) / cores;
        const uint32_t extra = uint32_t(total) % cores;
        uint32_t begin = 0;
        uint32_t startBatch = 0;
        uint32_t endBatch = 0;
        for (uint32_t core = 0; core < cores; ++core) {
            const uint32_t end = begin + common + uint32_t(core < extra);
            while (startBatch + 1 < data.batch && cu.GetValue(startBatch + 1) <= int32_t(begin)) {
                ++startBatch;
            }
            while (endBatch + 1 < data.batch && cu.GetValue(endBatch + 1) < int32_t(end)) {
                ++endBatch;
            }
            const uint32_t localEnd = end - uint32_t(cu.GetValue(endBatch));
            const bool batchBoundary = int32_t(end) == cu.GetValue(endBatch + 1);
            const uint32_t offset = core * 9;
            out.SetValue(offset, 1);
            // The consumer ignores core 0's start tuple and begins at (0,0,0).
            if (core > 0) {
                out.SetValue(offset + 1, startBatch);
                out.SetValue(offset + 2, begin - uint32_t(cu.GetValue(startBatch)));
            }
            out.SetValue(offset + 4, endBatch + uint32_t(batchBoundary));
            out.SetValue(offset + 5, batchBoundary ? 0 : localEnd);
            // S2 start/end and first FD workspace index stay zero; all FD rows
            // at offset 36*9 and all inactive AIC records stay disabled.
            begin = end;
        }
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(outGm, out, 1024);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
