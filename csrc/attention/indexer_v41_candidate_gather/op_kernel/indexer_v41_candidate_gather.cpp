// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace CandidateGather {
using namespace AscendC;
constexpr uint32_t CANDIDATES = 2048;
constexpr uint32_t BLOCK_POSITIONS = 8;
constexpr uint32_t TILE_BLOCKS = 8;
constexpr uint32_t DIM = 128;
constexpr uint32_t TILE_VALUES = TILE_BLOCKS * BLOCK_POSITIONS * DIM;
struct TilingData {
    uint32_t positions, cores, pageSize, pages, tablePages, reserved;
    uint64_t keyStride, scaleStride;
};
template <HardEvent Event>
__aicore__ inline void Sync()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

// Pure AIV page gather. Matrix multiplication is a separate operation.
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR scale, GM_ADDR candidates,
        GM_ADDR table, GM_ADDR length, GM_ADDR boundaries, GM_ADDR outKey,
        GM_ADDR outScale, GM_ADDR positions, const TilingData &data, TPipe *pipe)
    {
        data_ = data;
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(key));
        scale_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale));
        candidates_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(candidates));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        length_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(length));
        boundaries_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(boundaries));
        outKey_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(outKey));
        outScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outScale));
        positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(positions));
        pipe->InitBuffer(candidatesBuf_, CANDIDATES * sizeof(float));
        pipe->InitBuffer(intBuf_, TILE_VALUES * sizeof(int8_t));
        pipe->InitBuffer(halfBuf_, TILE_VALUES * sizeof(half));
        pipe->InitBuffer(floatBuf_, TILE_VALUES * sizeof(float));
        pipe->InitBuffer(outBuf_, TILE_VALUES * sizeof(bfloat16_t));
        pipe->InitBuffer(scaleHalfBuf_, TILE_BLOCKS * 16 * sizeof(half));
        pipe->InitBuffer(scaleFloatBuf_, TILE_BLOCKS * 16 * sizeof(float));
        pipe->InitBuffer(positionBuf_, TILE_BLOCKS * BLOCK_POSITIONS * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        auto candidates = candidatesBuf_.Get<float>();
        DataCopy(candidates, candidates_, CANDIDATES);
        Sync<HardEvent::MTE2_S>();
        const int32_t length = length_.GetValue(0);
        const bool active = boundaries_.GetValue(0) == 0 && boundaries_.GetValue(1) == 1;
        for (uint32_t tile = GetBlockIdx() * TILE_BLOCKS; tile < data_.positions / BLOCK_POSITIONS;
             tile += data_.cores * TILE_BLOCKS) {
            const uint32_t remaining = data_.positions / BLOCK_POSITIONS - tile;
            const uint32_t blocks = remaining < TILE_BLOCKS ? remaining : TILE_BLOCKS;
            GatherTile(tile, blocks, candidates, active ? length : 0);
        }
    }

private:
    __aicore__ inline bool Contains(LocalTensor<float> sorted, int32_t value)
    {
        // Descending order. Duplicates and negative padding are permitted.
        uint32_t lo = 0, hi = CANDIDATES;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (sorted.GetValue(mid) > static_cast<float>(value)) { lo = mid + 1; }
            else { hi = mid; }
        }
        return lo < CANDIDATES && sorted.GetValue(lo) == static_cast<float>(value);
    }

    __aicore__ inline void GatherTile(uint32_t tile, uint32_t blocks,
        LocalTensor<float> candidates, int32_t length)
    {
        auto packed = intBuf_.Get<int8_t>();
        auto halfKey = halfBuf_.Get<half>();
        auto floatKey = floatBuf_.Get<float>();
        auto output = outBuf_.Get<bfloat16_t>();
        auto scaleHalf = scaleHalfBuf_.Get<half>();
        auto scaleFloat = scaleFloatBuf_.Get<float>();
        auto ids = positionBuf_.Get<int32_t>();
        Duplicate(packed.ReinterpretCast<int16_t>(), int16_t(0), TILE_VALUES / 2);
        Duplicate(scaleHalf, half(0), TILE_BLOCKS * 16);
        Sync<HardEvent::V_MTE2>();
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint32_t slot = tile + b;
            int32_t block;
            bool selected;
            if (data_.positions < CANDIDATES * BLOCK_POSITIONS) {
                // Short-context bound avoids computing 16K padded positions.
                // Logical enumeration also preserves all unique candidates
                // when duplicates consume more than the short output capacity.
                block = slot;
                selected = Contains(candidates, block);
            } else {
                block = static_cast<int32_t>(candidates.GetValue(slot));
                selected = slot == 0 || candidates.GetValue(slot - 1) != static_cast<float>(block);
            }
            const int64_t first = static_cast<int64_t>(block) * BLOCK_POSITIONS;
            bool valid = selected && block >= 0 && first < length &&
                first / data_.pageSize < data_.tablePages;
            int32_t physical = -1;
            if (valid) {
                physical = table_.GetValue(first / data_.pageSize);
                valid = physical >= 0 && physical < static_cast<int32_t>(data_.pages);
            }
            if (valid) {
                const uint32_t within = first % data_.pageSize;
                DataCopy(packed[b * BLOCK_POSITIONS * DIM],
                    key_[static_cast<uint64_t>(physical) * data_.keyStride + within * DIM], BLOCK_POSITIONS * DIM);
                DataCopyExtParams copy{1, BLOCK_POSITIONS * sizeof(half), 0, 0, 0};
                DataCopyPadExtParams<half> padding{true, 0, 0, half(0)};
                DataCopyPad(scaleHalf[b * 16],
                    scale_[static_cast<uint64_t>(physical) * data_.scaleStride + within], copy, padding);
            }
            for (uint32_t p = 0; p < BLOCK_POSITIONS; ++p) {
                ids.SetValue(b * BLOCK_POSITIONS + p, valid && first + p < length ? first + p : -1);
            }
        }
        Sync<HardEvent::MTE2_V>();
        const uint32_t elements = blocks * BLOCK_POSITIONS * DIM;
        Cast(halfKey, packed, RoundMode::CAST_NONE, elements);
        Cast(scaleFloat, scaleHalf, RoundMode::CAST_NONE, blocks * 16);
        PipeBarrier<PIPE_V>();
        Cast(floatKey, halfKey, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Cast(output, floatKey, RoundMode::CAST_RINT, elements);
        Sync<HardEvent::V_MTE3>();
        Sync<HardEvent::S_MTE3>();
        DataCopy(outKey_[static_cast<uint64_t>(tile) * BLOCK_POSITIONS * DIM], output, elements);
        DataCopy(positions_[tile * BLOCK_POSITIONS], ids, blocks * BLOCK_POSITIONS);
        for (uint32_t b = 0; b < blocks; ++b) {
            DataCopy(outScale_[(tile + b) * BLOCK_POSITIONS], scaleFloat[b * 16], BLOCK_POSITIONS);
        }
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_S>();
    }

    TilingData data_;
    GlobalTensor<int8_t> key_;
    GlobalTensor<half> scale_;
    GlobalTensor<float> candidates_, outScale_;
    GlobalTensor<int32_t> table_, length_, boundaries_, positions_;
    GlobalTensor<bfloat16_t> outKey_;
    TBuf<TPosition::VECCALC> candidatesBuf_, intBuf_, halfBuf_, floatBuf_, outBuf_;
    TBuf<TPosition::VECCALC> scaleHalfBuf_, scaleFloatBuf_, positionBuf_;
};
}  // namespace CandidateGather

extern "C" __global__ __aicore__ void indexer_v41_candidate_gather(GM_ADDR key_cache,
    GM_ADDR key_scale_cache, GM_ADDR sorted_blocks, GM_ADDR block_table, GM_ADDR seqused_k,
    GM_ADDR cu_seqlens_q, GM_ADDR gathered_key, GM_ADDR gathered_scale, GM_ADDR positions,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(CandidateGather::TilingData);
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    CandidateGather::Kernel kernel;
    kernel.Init(key_cache, key_scale_cache, sorted_blocks, block_table, seqused_k,
        cu_seqlens_q, gathered_key, gathered_scale, positions, data, &pipe);
    kernel.Process();
}
