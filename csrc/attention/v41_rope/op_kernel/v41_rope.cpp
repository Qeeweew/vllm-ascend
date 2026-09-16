// SPDX-License-Identifier: Apache-2.0
#include "v41_rope_core.h"
#include "v41_rope_heads.h"
using namespace AscendC;

// Keep this wire layout identical to op_host/v41_rope_tiling.h.
struct V41RopeTilingData {
    uint32_t rows;
    uint32_t heads;
    uint32_t width;
    uint32_t tableRows;
    uint32_t cores;
};

template <bool Inverse>
__aicore__ inline void RunRope(GM_ADDR x, GM_ADDR positions, GM_ADDR cos, GM_ADDR sin,
                              GM_ADDR output, const V41RopeTilingData &data, TPipe *pipe)
{
    GlobalTensor<bfloat16_t> inputGm, outputGm;
    GlobalTensor<int64_t> positionsGm;
    GlobalTensor<float> cosGm, sinGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
    positionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(positions));
    cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cos));
    sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sin));
    if (data.heads == 32 && data.width == 128 && data.rows >= 128 * 32) {
        V41RopeCore::Heads32 heads;
        heads.Init(pipe);
        for (uint32_t token = GetBlockIdx(); token < data.rows / 32; token += data.cores) {
            const int64_t position = positionsGm.GetValue(token);
            const uint64_t offset = uint64_t(token) * 32 * 128;
            heads.Load(inputGm, offset);
            const bool valid = position >= 0 && position < data.tableRows;
            if (valid) {
                heads.Rotate<Inverse>(cosGm, sinGm, position);
            }
            heads.Store(outputGm, offset, valid);
        }
        return;
    }
    V41RopeCore::Row row;
    row.Init(pipe);
    // Adjacent heads reuse their rotary table. Keep contiguous, balanced row
    // ranges so the MTE2 table loads occur once per token rather than per head.
    const uint32_t core = GetBlockIdx();
    const uint32_t common = data.rows / data.cores;
    const uint32_t extra = data.rows % data.cores;
    const uint32_t begin = core * common + (core < extra ? core : extra);
    const uint32_t end = begin + common + (core < extra ? 1 : 0);
    uint32_t previousToken = UINT32_MAX;
    int64_t position = -1;
    int64_t tablePosition = -1;
    for (uint32_t index = begin; index < end; ++index) {
        const uint32_t token = index / data.heads;
        if (token != previousToken) {
            position = positionsGm.GetValue(token);
            previousToken = token;
        }
        const uint64_t offset = static_cast<uint64_t>(index) * data.width;
        row.Load(inputGm, offset, data.width);
        const bool valid = position >= 0 && position < data.tableRows;
        if (valid) {
            row.Rotate<Inverse>(cosGm, sinGm, position, data.width, position != tablePosition);
            tablePosition = position;
        }
        row.Store(outputGm, offset, data.width, valid);
    }
}

extern "C" __global__ __aicore__ void v41_rope(
    GM_ADDR x, GM_ADDR positions, GM_ADDR cos, GM_ADDR sin, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(V41RopeTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        RunRope<false>(x, positions, cos, sin, output, data, &pipe);
    } else if (TILING_KEY_IS(1)) {
        RunRope<true>(x, positions, cos, sin, output, data, &pipe);
    }
}
