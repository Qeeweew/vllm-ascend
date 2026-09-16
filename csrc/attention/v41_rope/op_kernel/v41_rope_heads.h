// SPDX-License-Identifier: Apache-2.0
#ifndef V41_ROPE_HEADS_H
#define V41_ROPE_HEADS_H
#include "v41_rope_core.h"

namespace V41RopeCore {
// The wide-head query case moves one complete token per DMA and applies the
// same table through vector repeat strides. Prefix BF16 bytes never change.
class Heads32 {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(rowBuffer_, ELEMENTS * sizeof(bfloat16_t));
        pipe->InitBuffer(floatBuffer_, (ELEMENTS + 3 * ROTATED) * sizeof(float));
        pipe->InitBuffer(tableBuffer_, ROTARY * sizeof(float));
        pipe->InitBuffer(mapBuffer_, 2 * ROTATED * sizeof(uint32_t));
        row_ = rowBuffer_.Get<bfloat16_t>();
        floats_ = floatBuffer_.Get<float>();
        planar_ = floats_[ELEMENTS];
        products_ = planar_[ROTATED];
        tables_ = tableBuffer_.Get<float>();
        maps_ = mapBuffer_.Get<uint32_t>();
        for (uint32_t head = 0; head < HEADS; ++head) {
            for (uint32_t pair = 0; pair < PAIRS; ++pair) {
                const uint32_t base = head * ROTARY;
                maps_.SetValue(base + pair, (head * WIDTH + ROTARY + 2 * pair) * sizeof(float));
                maps_.SetValue(base + PAIRS + pair, (head * WIDTH + ROTARY + 2 * pair + 1) * sizeof(float));
                maps_.SetValue(ROTATED + base + 2 * pair, (base + pair) * sizeof(float));
                maps_.SetValue(ROTATED + base + 2 * pair + 1, (base + PAIRS + pair) * sizeof(float));
            }
        }
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Load(const GlobalTensor<bfloat16_t> &input, uint64_t offset)
    {
        DataCopyExtParams copy{1, ELEMENTS * uint32_t(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, bfloat16_t(0)};
        DataCopyPad(row_, input[offset], copy, padding);
    }

    template <bool Inverse>
    __aicore__ inline void Rotate(const GlobalTensor<float> &cos, const GlobalTensor<float> &sin, uint64_t position)
    {
        DataCopyExtParams copy{1, PAIRS * uint32_t(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
        DataCopyPad(tables_, cos[position * PAIRS], copy, padding);
        DataCopyPad(tables_[PAIRS], sin[position * PAIRS], copy, padding);
        Sync<HardEvent::MTE2_V>();
        Cast(floats_, row_, RoundMode::CAST_NONE, ELEMENTS);
        if constexpr (Inverse) {
            Muls(tables_[PAIRS], tables_[PAIRS], -1.0f, PAIRS);
        }
        PipeBarrier<PIPE_V>();
        Gather(planar_, floats_, maps_, 0, ROTATED);
        PipeBarrier<PIPE_V>();
        // Each head occupies 64 FP32 elements; the shared table stays at the
        // same address for all 32 repeats (src1RepStride = 0).
        BinaryRepeatParams tableRepeat{1, 1, 1, 8, 8, 0};
        Mul(products_, planar_, tables_, uint64_t(PAIRS), HEADS, tableRepeat);
        Mul(products_[PAIRS], planar_[PAIRS], tables_[PAIRS], uint64_t(PAIRS), HEADS, tableRepeat);
        Mul(products_[ROTATED], planar_, tables_[PAIRS], uint64_t(PAIRS), HEADS, tableRepeat);
        Mul(products_[ROTATED + PAIRS], planar_[PAIRS], tables_, uint64_t(PAIRS), HEADS, tableRepeat);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams arithmeticRepeat{1, 1, 1, 8, 8, 8};
        Sub(planar_, products_, products_[PAIRS], uint64_t(PAIRS), HEADS, arithmeticRepeat);
        Add(planar_[PAIRS], products_[ROTATED], products_[ROTATED + PAIRS],
            uint64_t(PAIRS), HEADS, arithmeticRepeat);
        PipeBarrier<PIPE_V>();
        Gather(floats_, planar_, maps_[ROTATED], 0, ROTATED);
        PipeBarrier<PIPE_V>();
        // Scatter only the last 64 BF16 values of each 128-wide destination.
        UnaryRepeatParams castRepeat{1, 1, 8, 8};
        Cast(row_[ROTARY], floats_, RoundMode::CAST_RINT, uint64_t(ROTARY), HEADS, castRepeat);
    }

    __aicore__ inline void Store(const GlobalTensor<bfloat16_t> &output, uint64_t offset, bool rotated)
    {
        if (rotated) {
            Sync<HardEvent::V_MTE3>();
        } else {
            Sync<HardEvent::MTE2_MTE3>();
        }
        DataCopyExtParams copy{1, ELEMENTS * uint32_t(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(output[offset], row_, copy);
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
    }

private:
    static constexpr uint32_t HEADS = 32;
    static constexpr uint32_t WIDTH = 128;
    static constexpr uint32_t ELEMENTS = HEADS * WIDTH;
    static constexpr uint32_t ROTATED = HEADS * ROTARY;
    TBuf<TPosition::VECCALC> rowBuffer_, floatBuffer_, tableBuffer_, mapBuffer_;
    LocalTensor<bfloat16_t> row_;
    LocalTensor<float> floats_, planar_, products_, tables_;
    LocalTensor<uint32_t> maps_;
};
}  // namespace V41RopeCore
#endif
