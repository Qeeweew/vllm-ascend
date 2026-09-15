// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"



namespace EngramGate {
using namespace AscendC;
constexpr uint32_t DIM = 5120;
constexpr uint32_t COPIES = 4;
constexpr uint32_t BLOCK = 8;
struct EngramGateTilingData {
    uint32_t tokens;
    uint32_t cores;
    float eps;
};

template <HardEvent Event>
__aicore__ inline void Sync()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

// One task owns a complete (token, hc copy). There is no inter-core barrier,
// global reduction workspace, matmul, or hidden state mutation.
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR kv, GM_ADDR q, GM_ADDR k,
        GM_ADDR mask, GM_ADDR output, const EngramGateTilingData &data, TPipe *pipe)
    {
        data_ = data;
        hidden_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(hidden));
        kv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kv));
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        mask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(mask));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
        // 153696 bytes total UB, below the arch22 192 KiB limit.
        pipe->InitBuffer(bfBuf_, 5 * DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(workBuf_, 4 * DIM * sizeof(float));
        pipe->InitBuffer(reduceBuf_, DIM * sizeof(float));
        pipe->InitBuffer(scalarBuf_, 3 * BLOCK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t row = GetBlockIdx(); row < static_cast<uint64_t>(data_.tokens) * COPIES;
             row += data_.cores) {
            Row(row);
        }
    }

private:

    __aicore__ inline void Row(uint64_t row)
    {
        const uint64_t token = row / COPIES;
        const uint32_t hc = row % COPIES;
        auto bf = bfBuf_.Get<bfloat16_t>();
        auto h = workBuf_.Get<float>();
        auto key = h[DIM];
        auto temp = h[2 * DIM];
        auto product = h[3 * DIM];
        auto scratch = reduceBuf_.Get<float>();
        auto stats = scalarBuf_.Get<float>();
        DataCopy(bf, hidden_[row * DIM], DIM);
        if (mask_.GetValue(token) == 0) {
            // Exact pass-through for graph padding / non-text tokens.
            Sync<HardEvent::MTE2_MTE3>();
            DataCopy(output_[row * DIM], bf, DIM);
            Sync<HardEvent::MTE3_MTE2>();
            Sync<HardEvent::MTE3_V>();
            return;
        }
        DataCopy(bf[DIM], kv_[token * (COPIES + 1) * DIM + hc * DIM], DIM);
        DataCopy(bf[2 * DIM], q_[hc * DIM], DIM);
        DataCopy(bf[3 * DIM], k_[hc * DIM], DIM);
        DataCopy(bf[4 * DIM], kv_[token * (COPIES + 1) * DIM + COPIES * DIM], DIM);
        Sync<HardEvent::MTE2_V>();
        Cast(h, bf, RoundMode::CAST_NONE, DIM);
        Cast(key, bf[DIM], RoundMode::CAST_NONE, DIM);
        Cast(temp, bf[2 * DIM], RoundMode::CAST_NONE, DIM);
        Cast(product, bf[3 * DIM], RoundMode::CAST_NONE, DIM);
        PipeBarrier<PIPE_V>();
        // Preserve model.py association: weight=q*k, dot=sum((h*weight)*key).
        Mul(product, temp, product, DIM);
        PipeBarrier<PIPE_V>();
        Mul(product, h, product, DIM);
        Mul(temp, h, h, DIM);
        PipeBarrier<PIPE_V>();
        ReduceSum(stats, temp, scratch, DIM);
        Mul(product, product, key, DIM);
        PipeBarrier<PIPE_V>();
        Mul(temp, key, key, DIM);
        ReduceSum(stats[2 * BLOCK], product, scratch, DIM);
        PipeBarrier<PIPE_V>();
        ReduceSum(stats[BLOCK], temp, scratch, DIM);
        PipeBarrier<PIPE_V>();
        Muls(stats, stats, 1.0f / DIM, 1);
        Muls(stats[BLOCK], stats[BLOCK], 1.0f / DIM, 1);
        PipeBarrier<PIPE_V>();
        Adds(stats, stats, data_.eps, 1);
        Adds(stats[BLOCK], stats[BLOCK], data_.eps, 1);
        PipeBarrier<PIPE_V>();
        // Basic arch22 vrsqrt(1) returns 0.998046875. Use the precise
        // Sqrt + Div sequence from CANN normalization instead of that estimate.
        Duplicate(temp, 1.0f, 1);
        Sqrt(stats, stats, 1);
        Sqrt(stats[BLOCK], stats[BLOCK], 1);
        PipeBarrier<PIPE_V>();
        Div(stats, temp, stats, 1);
        Div(stats[BLOCK], temp, stats[BLOCK], 1);
        PipeBarrier<PIPE_V>();
        Mul(stats, stats, stats[BLOCK], 1);
        PipeBarrier<PIPE_V>();
        Mul(stats, stats[2 * BLOCK], stats, 1);
        PipeBarrier<PIPE_V>();
        Muls(stats, stats, 0.013975424859373685f, 1);  // 5120**-0.5
        Sync<HardEvent::V_S>();
        const float dot = stats.GetValue(0);
        Abs(stats, stats, 1);
        PipeBarrier<PIPE_V>();
        Maxs(stats, stats, 1e-6f, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(stats, stats, 1);
        PipeBarrier<PIPE_V>();
        Muls(stats, stats, dot < 0.0f ? 1.0f : -1.0f, 1);
        PipeBarrier<PIPE_V>();
        Exp(stats, stats, 1);
        PipeBarrier<PIPE_V>();
        Adds(stats, stats, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        // Match CANN Sigmoid: divide a FP32 one by the denominator using vdiv.
        Duplicate(stats[BLOCK], 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(stats, stats[BLOCK], stats, 1);
        Sync<HardEvent::V_S>();
        const float gate = stats.GetValue(0);
        Cast(temp, bf[4 * DIM], RoundMode::CAST_NONE, DIM);
        PipeBarrier<PIPE_V>();
        Muls(temp, temp, gate, DIM);
        PipeBarrier<PIPE_V>();
        Add(temp, h, temp, DIM);  // Separate FP32 multiply/add, no FMA contraction.
        PipeBarrier<PIPE_V>();
        Cast(bf, temp, RoundMode::CAST_RINT, DIM);
        Sync<HardEvent::V_MTE3>();
        DataCopy(output_[row * DIM], bf, DIM);
        // Next row reuses both the BF16 MTE destination and FP32 Vector space.
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
    }

    EngramGateTilingData data_;
    GlobalTensor<bfloat16_t> hidden_, kv_, q_, k_, output_;
    GlobalTensor<uint8_t> mask_;
    TBuf<TPosition::VECCALC> bfBuf_, workBuf_, reduceBuf_, scalarBuf_;
};
}  // namespace EngramGate

extern "C" __global__ __aicore__ void engram_gate(GM_ADDR hidden, GM_ADDR kv,
    GM_ADDR q_weight, GM_ADDR k_weight, GM_ADDR token_mask, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(EngramGate::EngramGateTilingData);
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    EngramGate::Kernel kernel;
    kernel.Init(hidden, kv, q_weight, k_weight, token_mask, output, data, &pipe);
    kernel.Process();
}
