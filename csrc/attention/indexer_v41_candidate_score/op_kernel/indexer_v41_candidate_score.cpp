// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace CandidateScore {
using namespace AscendC;
constexpr uint32_t HEADS = 32;
constexpr uint32_t TILE = 256;
struct TilingData { uint32_t positions, cores; };
template <HardEvent Event>
__aicore__ inline void Sync()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

// No matmul or topk here: only the frozen QLI score numerical contract.
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR qk, GM_ADDR weights, GM_ADDR queryScale,
        GM_ADDR scale, GM_ADDR positions, GM_ADDR scores, const TilingData &data, TPipe *pipe)
    {
        data_ = data;
        qk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qk));
        weights_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weights));
        queryScale_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(queryScale));
        scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
        positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(positions));
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));
        pipe->InitBuffer(qkBuf_, HEADS * TILE * sizeof(float));
        pipe->InitBuffer(roundBuf_, HEADS * TILE * sizeof(half));
        pipe->InitBuffer(weightHalfBuf_, 2 * HEADS * sizeof(half));
        pipe->InitBuffer(weightFloatBuf_, HEADS * sizeof(float));
        pipe->InitBuffer(scaleBuf_, TILE * sizeof(float));
        pipe->InitBuffer(positionBuf_, TILE * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        auto halfWeight = weightHalfBuf_.Get<half>();
        auto weight = weightFloatBuf_.Get<float>();
        DataCopy(halfWeight, weights_, HEADS);
        DataCopy(halfWeight[HEADS], queryScale_, HEADS);
        Sync<HardEvent::MTE2_V>();
        Mul(halfWeight, halfWeight, halfWeight[HEADS], HEADS);
        PipeBarrier<PIPE_V>();
        Cast(weight, halfWeight, RoundMode::CAST_NONE, HEADS);
        Sync<HardEvent::V_S>();
        for (uint32_t offset = GetBlockIdx() * TILE; offset < data_.positions; offset += data_.cores * TILE) {
            const uint32_t remaining = data_.positions - offset;
            const uint32_t count = remaining < TILE ? remaining : TILE;
            auto qk = qkBuf_.Get<float>();
            auto rounded = roundBuf_.Get<half>();
            auto scale = scaleBuf_.Get<float>();
            auto ids = positionBuf_.Get<int32_t>();
            DataCopyExtParams copy{HEADS, count * uint32_t(sizeof(float)),
                (data_.positions - count) * uint32_t(sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> padding{false, 0, 0, 0};
            DataCopyPad(qk, qk_[offset], copy, padding);
            DataCopy(scale, scale_[offset], count);
            DataCopy(ids, positions_[offset], count);
            Sync<HardEvent::MTE2_V>();
            Muls(qk, qk, 1.0f / 1024.0f, HEADS * count);
            PipeBarrier<PIPE_V>();
            Maxs(qk, qk, 0.0f, HEADS * count);
            PipeBarrier<PIPE_V>();
            Cast(rounded, qk, RoundMode::CAST_RINT, HEADS * count);
            PipeBarrier<PIPE_V>();
            Cast(qk, rounded, RoundMode::CAST_NONE, HEADS * count);
            PipeBarrier<PIPE_V>();
            for (uint32_t head = 0; head < HEADS; ++head) {
                Muls(qk[head * count], qk[head * count], weight.GetValue(head), count);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t halfHeads = HEADS / 2; halfHeads; halfHeads /= 2) {
                Add(qk, qk, qk[halfHeads * count], halfHeads * count);
                PipeBarrier<PIPE_V>();
            }
            Mul(qk, qk, scale, count);
            Sync<HardEvent::V_S>();
            Sync<HardEvent::MTE2_S>();
            // Scalar writes only the rare invalid lanes. Signed-weight scores
            // can be negative: zero-filled invalid K is never a valid mask.
            for (uint32_t p = 0; p < count; ++p) {
                if (ids.GetValue(p) < 0) { qk.ReinterpretCast<uint32_t>().SetValue(p, 0xff800000U); }
            }
            Sync<HardEvent::S_MTE3>();
            Sync<HardEvent::V_MTE3>();
            DataCopy(scores_[offset], qk, count);
            Sync<HardEvent::MTE3_V>();
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    TilingData data_;
    GlobalTensor<float> qk_, scale_, scores_;
    GlobalTensor<half> weights_, queryScale_;
    GlobalTensor<int32_t> positions_;
    TBuf<TPosition::VECCALC> qkBuf_, roundBuf_, weightHalfBuf_, weightFloatBuf_, scaleBuf_, positionBuf_;
};
}  // namespace CandidateScore

extern "C" __global__ __aicore__ void indexer_v41_candidate_score(GM_ADDR qk,
    GM_ADDR weights, GM_ADDR query_scale, GM_ADDR gathered_scale, GM_ADDR positions,
    GM_ADDR scores, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(CandidateScore::TilingData);
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    CandidateScore::Kernel kernel;
    kernel.Init(qk, weights, query_scale, gathered_scale, positions, scores, data, &pipe);
    kernel.Process();
}
