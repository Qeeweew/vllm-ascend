// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace V41Router {
using namespace AscendC;
constexpr uint32_t MAX_EXPERTS = 384;
constexpr uint32_t ALIGN_FLOAT = 8;
constexpr float SOFTPLUS_THRESHOLD = 20.0f;
constexpr float FP32_TINY = 1.1754943508222875e-38f;
struct TilingData {
    uint32_t rows, experts, topK, cores, vocabulary, hasTextBias, renormalize;
    float scaling;
};
template <HardEvent Event>
__aicore__ inline void Sync()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

// One row per AIV; rows stride over cores. Hash rows evaluate K scores only and
// never read biases or run a discarded dynamic top-k. No inter-core workspace.
template <uint32_t EXPERTS, uint32_t TOP_K>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR tokenIds, GM_ADDR imageMask,
        GM_ADDR table, GM_ADDR textBias, GM_ADDR imageBias, GM_ADDR weights,
        GM_ADDR expertIds, const TilingData &data, TPipe *pipe)
    {
        data_ = data;
        logits_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logits));
        tokens_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tokenIds));
        mask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(imageMask));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        textBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(textBias));
        imageBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(imageBias));
        weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weights));
        expertIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertIds));
        pipe->InitBuffer(inputBuf_, MAX_EXPERTS * sizeof(float));
        pipe->InitBuffer(scoreBuf_, MAX_EXPERTS * sizeof(float));
        pipe->InitBuffer(biasBuf_, MAX_EXPERTS * sizeof(float));
        pipe->InitBuffer(idsBuf_, MAX_EXPERTS * sizeof(uint32_t));
        pipe->InitBuffer(sortBuf_, 4 * MAX_EXPERTS * sizeof(float));
        pipe->InitBuffer(rowBuf_, 4 * 32);
        pipe->InitBuffer(cmpBuf_, 64);
        pipe->InitBuffer(outputBuf_, 2 * 32);
    }

    __aicore__ inline void Process()
    {
        auto rowData = rowBuf_.Get<uint8_t>();
        auto ids = idsBuf_.Get<uint32_t>();
        ArithProgression(ids.ReinterpretCast<int32_t>(), int32_t(0), int32_t(1), EXPERTS);
        Sync<HardEvent::V_S>();
        for (uint32_t row = GetBlockIdx(); row < data_.rows; row += data_.cores) {
            DataCopyPad(rowData, mask_[row], {1, 1, 0, 0, 0}, {false, 0, 0, 0});
            Sync<HardEvent::MTE2_S>();
            const bool image = rowData.GetValue(0) != 0;
            const bool hash = !image && data_.vocabulary != 0;
            auto selected = outputBuf_.Get<int32_t>();
            auto output = outputBuf_.Get<float>()[ALIGN_FLOAT];
            auto input = inputBuf_.Get<float>();
            auto score = scoreBuf_.Get<float>();
            bool valid = true;
            if (hash) {
                auto token = rowBuf_.Get<int64_t>()[4];
                DataCopyPad(token, tokens_[row], {1, sizeof(int64_t), 0, 0, 0}, {false, 0, 0, 0});
                Sync<HardEvent::MTE2_S>();
                const int64_t tokenId = token.GetValue(0);
                valid = tokenId >= 0 && tokenId < data_.vocabulary;
                if (valid) {
                    DataCopyPad(selected, table_[tokenId * TOP_K],
                        {1, TOP_K * uint32_t(sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
                    Sync<HardEvent::MTE2_S>();
                    for (uint32_t k = 0; k < TOP_K; ++k) {
                        const int32_t id = selected.GetValue(k);
                        valid = valid && id >= 0 && id < EXPERTS;
                    }
                }
            }
            if (!valid) {
                for (uint32_t k = 0; k < TOP_K; ++k) {
                    selected.SetValue(k, -1); output.SetValue(k, 0.0f);
                }
            } else {
                DataCopy(input, logits_[static_cast<uint64_t>(row) * EXPERTS], EXPERTS);
                Sync<HardEvent::MTE2_S>();
                if (hash) {
                    // Packing K lanes locally avoids E-wide transcendental work.
                    auto packed = biasBuf_.Get<float>();
                    for (uint32_t k = 0; k < TOP_K; ++k) {
                        packed.SetValue(k, input.GetValue(selected.GetValue(k)));
                    }
                    for (uint32_t k = TOP_K; k < 64; ++k) { packed.SetValue(k, 0.0f); }
                    Sync<HardEvent::S_V>();
                    SoftplusSqrt(packed, score, 64);
                    Sync<HardEvent::V_S>();
                    for (uint32_t k = 0; k < TOP_K; ++k) { output.SetValue(k, score.GetValue(k)); }
                } else {
                    Sync<HardEvent::MTE2_V>();
                    SoftplusSqrt(input, score, EXPERTS);
                    auto biased = biasBuf_.Get<float>();
                    if (image || data_.hasTextBias) {
                        if (image) { DataCopy(biased, imageBias_, EXPERTS); }
                        else { DataCopy(biased, textBias_, EXPERTS); }
                        Sync<HardEvent::MTE2_V>();
                        Add(biased, score, biased, EXPERTS);
                    } else { DataCopy(biased, score, EXPERTS); }
                    PipeBarrier<PIPE_V>();
                    auto sorted = sortBuf_.Get<float>();
                    // Tie ordering remains gated against installed NPU topk.
                    auto sortScratch = sorted[2 * MAX_EXPERTS];
                    Sort<float, true>(sorted, biased, ids, sortScratch, EXPERTS / 32);
                    Sync<HardEvent::V_S>();
                    auto sortedIds = sorted.ReinterpretCast<uint32_t>();
                    for (uint32_t k = 0; k < TOP_K; ++k) {
                        const uint32_t id = sortedIds.GetValue(2 * k + 1);
                        selected.SetValue(k, static_cast<int32_t>(id));
                        output.SetValue(k, score.GetValue(id));
                    }
                }
                float sum = 0.0f;
                for (uint32_t k = 0; k < TOP_K; ++k) { sum += output.GetValue(k); }
                if (sum < FP32_TINY) { sum = FP32_TINY; }
                for (uint32_t k = 0; k < TOP_K; ++k) {
                    float value = output.GetValue(k);
                    if (data_.renormalize) { value = value / sum; }
                    if (data_.scaling != 1.0f) { value = value * data_.scaling; }
                    output.SetValue(k, value);
                }
            }
            Sync<HardEvent::S_MTE3>();
            DataCopyPad(weights_[static_cast<uint64_t>(row) * TOP_K], output,
                {1, TOP_K * uint32_t(sizeof(float)), 0, 0, 0});
            DataCopyPad(expertIds_[static_cast<uint64_t>(row) * TOP_K], selected,
                {1, TOP_K * uint32_t(sizeof(int32_t)), 0, 0, 0});
            Sync<HardEvent::MTE3_S>();
            Sync<HardEvent::MTE3_V>();
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void SoftplusSqrt(LocalTensor<float> input, LocalTensor<float> output, uint32_t count)
    {
        auto mask = cmpBuf_.Get<uint8_t>();
        auto exponential = sortBuf_.Get<float>();
        auto denominator = exponential[MAX_EXPERTS];
        auto correction = exponential[2 * MAX_EXPERTS];
        // log1p(t) = log(1+t) * t / ((1+t)-1) corrects the FP32
        // cancellation in 1+t. If 1+t rounds to 1, log1p(t) rounds to t.
        // NPU baseline probing confirms negative tails must be preserved.
        Mins(exponential, input, SOFTPLUS_THRESHOLD, count);
        PipeBarrier<PIPE_V>();
        Exp(exponential, exponential, count);
        PipeBarrier<PIPE_V>();
        Adds(output, exponential, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Adds(denominator, output, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Ln(output, output, count);
        Maxs(denominator, denominator, FP32_TINY, count);
        PipeBarrier<PIPE_V>();
        Div(correction, exponential, denominator, count);
        PipeBarrier<PIPE_V>();
        Mul(output, output, correction, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, exponential, 5.9604644775390625e-8f, CMPMODE::LE, count);
        PipeBarrier<PIPE_V>();
        Select(output, mask, exponential, output, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, input, SOFTPLUS_THRESHOLD, CMPMODE::GT, count);
        PipeBarrier<PIPE_V>();
        Select(output, mask, input, output, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        Sqrt(output, output, count);
        PipeBarrier<PIPE_V>();
    }

private:
    TilingData data_;
    GlobalTensor<float> logits_, textBias_, imageBias_, weights_;
    GlobalTensor<int64_t> tokens_;
    GlobalTensor<uint8_t> mask_;
    GlobalTensor<int32_t> table_, expertIds_;
    TBuf<TPosition::VECCALC> inputBuf_, scoreBuf_, biasBuf_, idsBuf_, sortBuf_, rowBuf_, cmpBuf_, outputBuf_;
};
}  // namespace V41Router

extern "C" __global__ __aicore__ void v41_moe_router(GM_ADDR logits, GM_ADDR token_ids,
    GM_ADDR image_mask, GM_ADDR tid2eid, GM_ADDR text_bias, GM_ADDR image_bias,
    GM_ADDR weights, GM_ADDR expert_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(V41Router::TilingData);
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    if (data.experts == 128) {
        V41Router::Kernel<128, 3> kernel;
        kernel.Init(logits, token_ids, image_mask, tid2eid, text_bias, image_bias, weights, expert_ids, data, &pipe);
        kernel.Process();
    } else {
        V41Router::Kernel<384, 6> kernel;
        kernel.Init(logits, token_ids, image_mask, tid2eid, text_bias, image_bias, weights, expert_ids, data, &pipe);
        kernel.Process();
    }
}
