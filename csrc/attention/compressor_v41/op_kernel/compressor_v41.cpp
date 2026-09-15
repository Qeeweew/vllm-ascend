// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace CompressorV41 {
using namespace AscendC;

// Keep this wire layout identical to op_host/compressor_v41_tiling.h.
struct CompressorV41TilingData {
    uint32_t tokens;
    uint32_t requests;
    uint32_t capacity;
    uint32_t stateBlocks;
    uint32_t cores;
    float eps;
};

constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t STATE_DIM = 2 * HEAD_DIM;
constexpr uint32_t FLOAT_BLOCK = 8;

// Each use sets and immediately consumes one local event. No arming, delayed
// wait, cross-core flag, or Cube dependency is hidden in this helper.
template <HardEvent Event>
__aicore__ inline void Sync()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

template <bool HasGate>
class Kernel {
public:
    __aicore__ inline void Init(
        GM_ADDR raw, GM_ADDR positions, GM_ADDR slots, GM_ADDR starts, GM_ADDR reqIds,
        GM_ADDR weight, GM_ADDR state, GM_ADDR output,
        const CompressorV41TilingData &data, TPipe *pipe)
    {
        data_ = data;
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(raw));
        rawBf16_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(raw));
        positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(positions));
        slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slots));
        starts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(starts));
        reqIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(reqIds));
        weight_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weight));
        state_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(state));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
        pipe->InitBuffer(rowsBuf_, 2 * STATE_DIM * sizeof(float));
        pipe->InitBuffer(workBuf_, 4 * HEAD_DIM * sizeof(float));
        pipe->InitBuffer(weightBuf_, HEAD_DIM * sizeof(float));
        pipe->InitBuffer(weightBf16Buf_, HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(bf16Buf_, HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(reduceBuf_, HEAD_DIM * sizeof(float));
        rows_ = rowsBuf_.Get<float>();
        work_ = workBuf_.Get<float>();
        normWeight_ = weightBuf_.Get<float>();
        bf16_ = bf16Buf_.Get<bfloat16_t>();
        reduction_ = reduceBuf_.Get<float>();
        auto weightBf16 = weightBf16Buf_.Get<bfloat16_t>();
        DataCopy(weightBf16, weight_, HEAD_DIM);
        Sync<HardEvent::MTE2_V>();
        Cast(normWeight_, weightBf16, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        // Interior tasks read raw only. A request task owns both the read of
        // the prior ring and all writes to that request's ring. Long prefills
        // therefore use all AIVs without a global barrier or state write race.
        const uint64_t boundaryTasks = HasGate ? data_.requests : 0;
        const uint64_t tasks = boundaryTasks + data_.tokens;
        for (uint64_t task = GetBlockIdx(); task < tasks; task += data_.cores) {
            if constexpr (HasGate) {
                if (task < boundaryTasks) {
                    Boundary(task);
                    continue;
                }
            }
            Token(task - boundaryTasks);
        }
    }

private:
    __aicore__ inline bool ValidSlot(int64_t slot) const
    {
        return slot >= 0 && static_cast<uint64_t>(slot) <
            static_cast<uint64_t>(data_.stateBlocks) * data_.capacity;
    }

    __aicore__ inline void Store(uint32_t token)
    {
        Sync<HardEvent::V_MTE3>();
        DataCopy(output_[static_cast<uint64_t>(token) * HEAD_DIM], bf16_, HEAD_DIM);
        // The next task may reuse bf16_ as either an MTE2 destination or a
        // Vector destination. Release it to both pipelines after this store.
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
    }

    __aicore__ inline void StoreZero(uint32_t token)
    {
        Duplicate(bf16_.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), HEAD_DIM);
        Store(token);
    }

    __aicore__ inline void Normalize(uint32_t token, LocalTensor<float> value)
    {
        auto squared = work_[2 * HEAD_DIM];
        auto sum = work_[3 * HEAD_DIM];
        Mul(squared, value, value, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        ReduceSum(sum, squared, reduction_, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Muls(sum, sum, 1.0f / HEAD_DIM, 1);
        PipeBarrier<PIPE_V>();
        Adds(sum, sum, data_.eps, 1);
        PipeBarrier<PIPE_V>();
        // Basic arch22 vrsqrt is an estimate (vrsqrt(1) = 0.998046875).
        // Reuse the dead square buffer for the precise reciprocal numerator.
        Duplicate(squared, 1.0f, 1);
        Sqrt(sum, sum, 1);
        PipeBarrier<PIPE_V>();
        Div(sum, squared, sum, 1);
        Sync<HardEvent::V_S>();
        const float inverseRms = sum.GetValue(0);
        Muls(value, value, inverseRms, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Mul(value, value, normWeight_, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Cast(bf16_, value, RoundMode::CAST_RINT, HEAD_DIM);
        Store(token);
    }

    __aicore__ inline void PoolAndNormalize(uint32_t token)
    {
        auto kv0 = rows_;
        auto score0 = rows_[HEAD_DIM];
        auto kv1 = rows_[STATE_DIM];
        auto score1 = rows_[STATE_DIM + HEAD_DIM];
        auto peak = work_;
        auto denominator = work_[HEAD_DIM];
        // Softmax is across the two tokens, separately in each feature lane.
        Max(peak, score0, score1, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Sub(score0, score0, peak, HEAD_DIM);
        Sub(score1, score1, peak, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Exp(score0, score0, HEAD_DIM);
        Exp(score1, score1, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Add(denominator, score0, score1, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        // Normalize weights before multiplying KV, preserving the reference
        // evaluation order instead of dividing a weighted numerator at end.
        Div(score0, score0, denominator, HEAD_DIM);
        Div(score1, score1, denominator, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Mul(kv0, kv0, score0, HEAD_DIM);
        Mul(kv1, kv1, score1, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Add(kv0, kv0, kv1, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        // This roundtrip is part of inference/model.py, not a removable cast.
        Cast(bf16_, kv0, RoundMode::CAST_RINT, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Cast(kv0, bf16_, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Normalize(token, kv0);
    }

    __aicore__ inline void Token(uint32_t token)
    {
        const int64_t slot = slots_.GetValue(token);
        if constexpr (!HasGate) {
            if (slot < 0) {
                StoreZero(token);
                return;
            }
            DataCopy(bf16_, rawBf16_[static_cast<uint64_t>(token) * HEAD_DIM], HEAD_DIM);
            Sync<HardEvent::MTE2_V>();
            Cast(rows_, bf16_, RoundMode::CAST_NONE, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Normalize(token, rows_);
        } else {
            const int64_t position = positions_.GetValue(token);
            if (!ValidSlot(slot) || position < 0 || (position & 1) == 0) {
                StoreZero(token);
                return;
            }
            const int32_t req = reqIds_.GetValue(token);
            if (req < 0 || req >= data_.requests) {
                StoreZero(token);
                return;
            }
            const int32_t start = starts_.GetValue(req);
            if (token == start) {
                return;  // This output belongs exclusively to Boundary(req).
            }
            if (start < 0 || token <= start || !ValidSlot(slots_.GetValue(token - 1))) {
                StoreZero(token);
                return;
            }
            DataCopy(rows_, raw_[static_cast<uint64_t>(token - 1) * STATE_DIM], 2 * STATE_DIM);
            Sync<HardEvent::MTE2_V>();
            PoolAndNormalize(token);
        }
    }

    __aicore__ inline void Boundary(uint32_t req)
    {
        const int32_t start = starts_.GetValue(req);
        const int32_t end = starts_.GetValue(req + 1);
        if (start < 0 || start >= end || end > data_.tokens) {
            return;
        }
        const int64_t slot = slots_.GetValue(start);
        const int64_t position = positions_.GetValue(start);
        if (!ValidSlot(slot) || position < 0) {
            return;
        }
        if ((position & 1) != 0) {
            const uint64_t block = slot / data_.capacity;
            const uint64_t previous = block * data_.capacity + (position - 1) % data_.capacity;
            DataCopy(rows_, state_[previous * STATE_DIM], STATE_DIM);
            DataCopy(rows_[STATE_DIM], raw_[static_cast<uint64_t>(start) * STATE_DIM], STATE_DIM);
            Sync<HardEvent::MTE2_V>();
            PoolAndNormalize(start);
        }
        // Read of the old ring above is complete before any tail overwrite.
        const int32_t count = end - start < data_.capacity ? end - start : data_.capacity;
        for (int32_t token = end - count; token < end; ++token) {
            const int64_t tailSlot = slots_.GetValue(token);
            if (!ValidSlot(tailSlot)) {
                continue;
            }
            DataCopy(rows_, raw_[static_cast<uint64_t>(token) * STATE_DIM], STATE_DIM);
            Sync<HardEvent::MTE2_MTE3>();
            DataCopy(state_[static_cast<uint64_t>(tailSlot) * STATE_DIM], rows_, STATE_DIM);
            Sync<HardEvent::MTE3_MTE2>();
        }
        // rows_ is reused by Vector in later tasks as well as by MTE2.
        Sync<HardEvent::MTE3_V>();
    }

    CompressorV41TilingData data_;
    GlobalTensor<float> raw_, state_;
    GlobalTensor<bfloat16_t> rawBf16_, weight_, output_;
    GlobalTensor<int64_t> positions_, slots_;
    GlobalTensor<int32_t> starts_, reqIds_;
    TBuf<TPosition::VECCALC> rowsBuf_, workBuf_, weightBuf_, weightBf16Buf_, bf16Buf_, reduceBuf_;
    LocalTensor<float> rows_, work_, normWeight_, reduction_;
    LocalTensor<bfloat16_t> bf16_;
};
}  // namespace CompressorV41

extern "C" __global__ __aicore__ void compressor_v41(
    GM_ADDR kv_score, GM_ADDR positions, GM_ADDR slot_mapping, GM_ADDR query_start_loc,
    GM_ADDR token_to_req_indices, GM_ADDR norm_weight, GM_ADDR state_cache, GM_ADDR latent_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CompressorV41::CompressorV41TilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(1)) {
        CompressorV41::Kernel<false> kernel;
        kernel.Init(kv_score, positions, slot_mapping, query_start_loc, token_to_req_indices,
                    norm_weight, state_cache, latent_out, data, &pipe);
        kernel.Process();
    } else if (TILING_KEY_IS(2)) {
        CompressorV41::Kernel<true> kernel;
        kernel.Init(kv_score, positions, slot_mapping, query_start_loc, token_to_req_indices,
                    norm_weight, state_cache, latent_out, data, &pipe);
        kernel.Process();
    }
}
