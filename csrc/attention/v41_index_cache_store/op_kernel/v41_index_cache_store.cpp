// SPDX-License-Identifier: Apache-2.0
#if __has_include("../v41_rope/v41_rope_core.h")
#include "../v41_rope/v41_rope_core.h"
#else
#include "../../v41_rope/op_kernel/v41_rope_core.h"
#endif
using namespace AscendC;
struct V41IndexCacheStoreTilingData {
    uint32_t tokens, page, blocks, tableRows, cores, ratio;
    uint64_t keyStride, scaleStride;
};

class V41IndexQuantizer {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(valuesBuffer_, 2 * WIDTH * sizeof(float));
        pipe->InitBuffer(reduceBuffer_, WIDTH * sizeof(float));
        pipe->InitBuffer(scaleBuffer_, 32 * sizeof(float));
        pipe->InitBuffer(outputBuffer_, WIDTH * sizeof(int8_t));
        values_ = valuesBuffer_.Get<float>();
        work_ = values_[WIDTH];
        reduce_ = reduceBuffer_.Get<float>();
        scale_ = scaleBuffer_.Get<float>();
        output_ = outputBuffer_.Get<int8_t>();
    }

    __aicore__ inline void Compute(LocalTensor<bfloat16_t> rounded)
    {
        // RoPE must finish its BF16 cast before this roundtrip. Removing that
        // boundary changes both the quantized integers and stored scale.
        PipeBarrier<PIPE_V>();
        Cast(values_, rounded, RoundMode::CAST_NONE, WIDTH);
        PipeBarrier<PIPE_V>();
        Abs(work_, values_, WIDTH);
        PipeBarrier<PIPE_V>();
        ReduceMax(scale_, work_, reduce_, WIDTH, false);
        V41RopeCore::Sync<HardEvent::V_S>();
        const float maximum = scale_.GetValue(0);
        if (maximum == 0.0f) {
            // Frozen CANN baseline: signed-zero rows emit INT8 zero and +0 scale.
            Duplicate(output_.ReinterpretCast<uint16_t>(), uint16_t(0), WIDTH / 2);
            Duplicate(scale_[16].ReinterpretCast<half>(), half(0), 16);
            return;
        }
        Duplicate(scale_[8], 127.0f, 8);
        PipeBarrier<PIPE_V>();
        Div(scale_[8], scale_[8], scale_, 1);
        PipeBarrier<PIPE_V>();
        Muls(scale_, scale_, 1.0f / 127.0f, 1);
        V41RopeCore::Sync<HardEvent::V_S>();
        const float multiplier = scale_.GetValue(8);
        Muls(values_, values_, multiplier, WIDTH);
        PipeBarrier<PIPE_V>();
        auto integers = work_.ReinterpretCast<int32_t>();
        Cast(integers, values_, RoundMode::CAST_RINT, WIDTH);
        PipeBarrier<PIPE_V>();
        SetDeqScale(half(1));
        PipeBarrier<PIPE_V>();
        Cast(values_.ReinterpretCast<half>(), integers, RoundMode::CAST_ROUND, WIDTH);
        PipeBarrier<PIPE_V>();
        Cast(output_, values_.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, WIDTH);
        // Scale is rounded only after integer values have been computed.
        Cast(scale_[16].ReinterpretCast<half>(), scale_, RoundMode::CAST_RINT, 1);
    }

    __aicore__ inline void Store(const GlobalTensor<int8_t> &keys, const GlobalTensor<half> &scales,
                                 uint64_t keyOffset, uint64_t scaleOffset)
    {
        V41RopeCore::Sync<HardEvent::V_MTE3>();
        DataCopyExtParams keyCopy{1, WIDTH, 0, 0, 0};
        DataCopyExtParams scaleCopy{1, sizeof(half), 0, 0, 0};
        DataCopyPad(keys[keyOffset], output_, keyCopy);
        DataCopyPad(scales[scaleOffset], scale_[16].ReinterpretCast<half>(), scaleCopy);
        V41RopeCore::Sync<HardEvent::MTE3_V>();
        V41RopeCore::Sync<HardEvent::MTE3_MTE2>();
    }

private:
    static constexpr uint32_t WIDTH = 128;
    TBuf<TPosition::VECCALC> valuesBuffer_, reduceBuffer_, scaleBuffer_, outputBuffer_;
    LocalTensor<float> values_, work_, reduce_, scale_;
    LocalTensor<int8_t> output_;
};

extern "C" __global__ __aicore__ void v41_index_cache_store(
    GM_ADDR key, GM_ADDR positions, GM_ADDR slots, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR key_cache, GM_ADDR scale_cache, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(V41IndexCacheStoreTilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<int8_t> keyGm;
    GlobalTensor<half> scaleGm;
    GlobalTensor<int64_t> positionsGm, slotsGm;
    GlobalTensor<float> cosGm, sinGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key));
    keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(key_cache));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale_cache));
    positionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(positions));
    slotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slots));
    cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cos));
    sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sin));
    TPipe pipe;
    V41RopeCore::Row row;
    V41IndexQuantizer quantizer;
    row.Init(&pipe);
    quantizer.Init(&pipe);
    for (uint32_t token = GetBlockIdx(); token < data.tokens; token += data.cores) {
        const int64_t position = positionsGm.GetValue(token);
        const int64_t slot = slotsGm.GetValue(token);
        if (position < 0 || slot < 0 || uint64_t(slot) >= uint64_t(data.page) * data.blocks ||
            (data.ratio == 2 && (position & 1) == 0)) continue;
        const int64_t groupPosition = data.ratio == 2 ? position - 1 : position;
        if (groupPosition >= data.tableRows) continue;
        row.Load(inputGm, uint64_t(token) * 128, 128);
        row.Rotate<false>(cosGm, sinGm, groupPosition, 128);
        quantizer.Compute(row.Values());
        const uint64_t page = slot / data.page;
        const uint64_t within = slot % data.page;
        quantizer.Store(keyGm, scaleGm, page * data.keyStride + within * 128, page * data.scaleStride + within);
    }
}
