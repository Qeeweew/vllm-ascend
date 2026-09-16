// SPDX-License-Identifier: Apache-2.0
#ifndef V41_ROPE_CORE_H
#define V41_ROPE_CORE_H
#include "kernel_operator.h"

namespace V41RopeCore {
using namespace AscendC;
constexpr uint32_t ROTARY = 64;
constexpr uint32_t PAIRS = ROTARY / 2;
constexpr uint32_t MAX_WIDTH = 512;

// Each event is set and immediately consumed. No delayed ownership protocol,
// cross-core flags, or global scratch is needed by these independent rows.
template <HardEvent Event>
__aicore__ inline void Sync()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

class Row {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(rowBuffer_, MAX_WIDTH * sizeof(bfloat16_t));
        pipe->InitBuffer(floatBuffer_, 5 * ROTARY * sizeof(float));
        pipe->InitBuffer(tableBuffer_, ROTARY * sizeof(float));
        pipe->InitBuffer(mapBuffer_, 2 * ROTARY * sizeof(uint32_t));
        row_ = rowBuffer_.Get<bfloat16_t>();
        floats_ = floatBuffer_.Get<float>();
        tables_ = tableBuffer_.Get<float>();
        maps_ = mapBuffer_.Get<uint32_t>();
        for (uint32_t i = 0; i < PAIRS; ++i) {
            maps_.SetValue(i, 2 * i * sizeof(float));
            maps_.SetValue(PAIRS + i, (2 * i + 1) * sizeof(float));
            maps_.SetValue(ROTARY + 2 * i, i * sizeof(float));
            maps_.SetValue(ROTARY + 2 * i + 1, (PAIRS + i) * sizeof(float));
        }
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline LocalTensor<bfloat16_t> Values() { return row_; }

    __aicore__ inline void Load(const GlobalTensor<bfloat16_t> &input, uint64_t offset, uint32_t width)
    {
        DataCopyExtParams copy{1, width * uint32_t(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, bfloat16_t(0)};
        DataCopyPad(row_, input[offset], copy, padding);
    }

    template <bool Inverse>
    __aicore__ inline void Rotate(const GlobalTensor<float> &cos, const GlobalTensor<float> &sin,
                                  uint64_t position, uint32_t width, bool refreshTables = true)
    {
        if (refreshTables) {
            DataCopyExtParams copy{1, PAIRS * uint32_t(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
            DataCopyPad(tables_, cos[position * PAIRS], copy, padding);
            DataCopyPad(tables_[PAIRS], sin[position * PAIRS], copy, padding);
        }
        Sync<HardEvent::MTE2_V>();
        Cast(floats_, row_[width - ROTARY], RoundMode::CAST_NONE, ROTARY);
        if constexpr (Inverse) {
            if (refreshTables) {
                Muls(tables_[PAIRS], tables_[PAIRS], -1.0f, PAIRS);
            }
        }
        PipeBarrier<PIPE_V>();
        auto planar = floats_[ROTARY];
        Gather(planar, floats_, maps_, 0, ROTARY);
        PipeBarrier<PIPE_V>();
        auto products = floats_[2 * ROTARY];
        Mul(products, planar, tables_, PAIRS);
        Mul(products[PAIRS], planar[PAIRS], tables_[PAIRS], PAIRS);
        Mul(products[ROTARY], planar, tables_[PAIRS], PAIRS);
        Mul(products[ROTARY + PAIRS], planar[PAIRS], tables_, PAIRS);
        // Separate vector instructions plus this dependency barrier preserve
        // the four FP32 products; do not replace these with MulAdd/FMA.
        PipeBarrier<PIPE_V>();
        Sub(planar, products, products[PAIRS], PAIRS);
        Add(planar[PAIRS], products[ROTARY], products[ROTARY + PAIRS], PAIRS);
        PipeBarrier<PIPE_V>();
        Gather(floats_, planar, maps_[ROTARY], 0, ROTARY);
        PipeBarrier<PIPE_V>();
        Cast(row_[width - ROTARY], floats_, RoundMode::CAST_RINT, ROTARY);
    }

    __aicore__ inline void Store(const GlobalTensor<bfloat16_t> &output, uint64_t offset,
                                 uint32_t width, bool rotated)
    {
        if (rotated) {
            Sync<HardEvent::V_MTE3>();
        } else {
            Sync<HardEvent::MTE2_MTE3>();
        }
        DataCopyExtParams copy{1, width * uint32_t(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(output[offset], row_, copy);
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
    }

private:
    TBuf<TPosition::VECCALC> rowBuffer_, floatBuffer_, tableBuffer_, mapBuffer_;
    LocalTensor<bfloat16_t> row_;
    LocalTensor<float> floats_, tables_;
    LocalTensor<uint32_t> maps_;
};
}  // namespace V41RopeCore
#endif
