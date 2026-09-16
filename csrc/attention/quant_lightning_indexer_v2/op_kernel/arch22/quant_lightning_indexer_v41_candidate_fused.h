// SPDX-License-Identifier: Apache-2.0
#ifndef QUANT_LIGHTNING_INDEXER_V41_CANDIDATE_FUSED_H
#define QUANT_LIGHTNING_INDEXER_V41_CANDIDATE_FUSED_H

#include "quant_lightning_indexer_v41_candidate_cube.h"
#include "quant_lightning_indexer_v2_vector.h"

namespace QLIV41Candidate {
using namespace AscendC;
constexpr uint32_t HEADS = 32;
constexpr uint32_t DIM = 128;
constexpr uint32_t CANDIDATES = 2048;
constexpr uint32_t BLOCK = 8;
constexpr uint32_t POSITIONS = CANDIDATES * BLOCK;
constexpr uint32_t TILE = 1024;
constexpr uint32_t TOPK = 512;
constexpr uint32_t SEGMENTS = POSITIONS / TILE;
constexpr uint32_t WEIGHT_OFFSET = 0;
constexpr uint32_t OFFSET_OFFSET = HEADS * 16 * sizeof(half);
constexpr uint32_t POSITION_OFFSET = OFFSET_OFFSET + CANDIDATES * sizeof(uint64_t);
constexpr uint32_t SCALE_OFFSET = POSITION_OFFSET + CANDIDATES * sizeof(int32_t);
constexpr uint32_t SCORE_OFFSET = SCALE_OFFSET + CANDIDATES * 16 * sizeof(half);
constexpr uint32_t STATE_OFFSET = SCORE_OFFSET + POSITIONS * sizeof(float);
constexpr uint32_t ROW_BYTES = STATE_OFFSET + 32;
static_assert(ROW_BYTES == 156704, "Keep host workspace calculation in sync");
constexpr uint32_t READY = 6;
constexpr uint32_t SCORED = 7;
constexpr uint32_t ACK = 8;

template <HardEvent Event>
__aicore__ inline void Sync()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

__aicore__ inline void Sort(LocalTensor<float> &pairs, LocalTensor<float> &scratch, uint32_t count)
{
    QLIV2ServiceVec::SortAll(pairs, scratch, count);
    PipeBarrier<PIPE_ALL>();
}

// Each query has a distinct record for this invocation. The early producer ACK
// never permits overwriting a score/scale record still used by a topk owner.
class Vector {
public:
    __aicore__ inline void Init(GM_ADDR weights, GM_ADDR queryScale, GM_ADDR keyScale,
        GM_ADDR candidates, GM_ADDR table, GM_ADDR length, GM_ADDR boundaries,
        GM_ADDR output, GM_ADDR workspace, const QLIV2TilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        workspace_ = workspace;
        weights_.SetGlobalBuffer((__gm__ half *)weights);
        queryScale_.SetGlobalBuffer((__gm__ half *)queryScale);
        keyScale_.SetGlobalBuffer((__gm__ half *)keyScale);
        candidates_.SetGlobalBuffer((__gm__ int32_t *)candidates);
        table_.SetGlobalBuffer((__gm__ int32_t *)table);
        length_.SetGlobalBuffer((__gm__ int32_t *)length);
        boundaries_.SetGlobalBuffer((__gm__ int32_t *)boundaries);
        output_.SetGlobalBuffer((__gm__ int32_t *)output);
        // IB event 0 uses slots indexed by physical even AIV ID, hence 2*P
        // 32-byte records. Every participating AIV clears its own slot.
        if (tiling.candidateFused != 2) {
            mailbox_.SetGlobalBuffer((__gm__ int32_t *)(workspace + uint64_t(tiling.s1Size) * ROW_BYTES));
        }
        // Exact static UB allocation: 71,840 B, below the arch22 192 KiB limit.
        pipe->InitBuffer(candidateBuf_, CANDIDATES * 2 * sizeof(float));
        pipe->InitBuffer(sortBuf_, CANDIDATES * 2 * sizeof(float));
        pipe->InitBuffer(inputBuf_, CANDIDATES * sizeof(int32_t));
        pipe->InitBuffer(weightBuf_, HEADS * 2 * sizeof(half));
        pipe->InitBuffer(expandedBuf_, OFFSET_OFFSET);
        pipe->InitBuffer(offsetBuf_, TILE / BLOCK * sizeof(uint64_t));
        pipe->InitBuffer(scaleHalfBuf_, TILE * 2 * sizeof(half));
        pipe->InitBuffer(scaleFloatBuf_, TILE * 2 * sizeof(float));
        pipe->InitBuffer(positionBuf_, TILE * sizeof(int32_t));
        pipe->InitBuffer(scoreBuf_, TILE * 2 * sizeof(float));
        pipe->InitBuffer(bestBuf_, TOPK * 2 * sizeof(float));
        pipe->InitBuffer(flagBuf_, 32);
    }

    __aicore__ inline void BindRow(uint32_t row)
    {
        BindRecord(workspace_ + uint64_t(row) * ROW_BYTES);
    }

    __aicore__ inline void BindRecord(GM_ADDR record)
    {
        expandedWeights_.SetGlobalBuffer((__gm__ half *)(record + WEIGHT_OFFSET));
        offsets_.SetGlobalBuffer((__gm__ uint64_t *)(record + OFFSET_OFFSET));
        blockPositions_.SetGlobalBuffer((__gm__ int32_t *)(record + POSITION_OFFSET));
        scales_.SetGlobalBuffer((__gm__ half *)(record + SCALE_OFFSET));
        scores_.SetGlobalBuffer((__gm__ float *)(record + SCORE_OFFSET));
        state_.SetGlobalBuffer((__gm__ int32_t *)(record + STATE_OFFSET));
    }

    __aicore__ inline void Prepare(uint32_t row)
    {
        PrepareAt<false>(row, workspace_ + uint64_t(row) * ROW_BYTES);
    }

    template <bool TRUSTED_UNIQUE>
    __aicore__ inline void PrepareAt(uint32_t row, GM_ADDR record)
    {
        BindRecord(record);
        auto input = inputBuf_.Get<int32_t>();
        auto candidates = candidateBuf_.Get<float>();
        auto scratch = sortBuf_.Get<float>();
        auto weight = weightBuf_.Get<half>();
        auto expanded = expandedBuf_.Get<half>();
        DataCopy(input, candidates_[uint64_t(row) * CANDIDATES], CANDIDATES);
        DataCopy(weight, weights_[uint64_t(row) * HEADS], HEADS);
        DataCopy(weight[HEADS], queryScale_[uint64_t(row) * HEADS], HEADS);
        Sync<HardEvent::MTE2_V>();
        if constexpr (!TRUSTED_UNIQUE) {
            Cast(candidates, input, RoundMode::CAST_NONE, CANDIDATES);
        }
        Mul(weight, weight, weight[HEADS], HEADS);
        PipeBarrier<PIPE_V>();
        Brcb(expanded, weight, HEADS / 8, {1, 8});
        if constexpr (!TRUSTED_UNIQUE) {
            Duplicate(candidates[CANDIDATES].ReinterpretCast<int32_t>(), int32_t(0), CANDIDATES);
            PipeBarrier<PIPE_V>();
            Sort(candidates, scratch, CANDIDATES);
            Sync<HardEvent::V_S>();
        } else {
            // Source-produced candidates are unique. Keep their order and
            // read exact INT32 IDs; no sorting, conversion or deduplication.
            Sync<HardEvent::MTE2_S>();
        }
        Sync<HardEvent::V_MTE3>();
        DataCopy(expandedWeights_, expanded, HEADS * 16);

        // Empty requests are skipped naturally. Padding rows have visible=0.
        int32_t visible = 0;
        uint32_t request = 0;
        for (; request < tiling_.bSize; ++request) {
            const int32_t begin = boundaries_.GetValue(request);
            const int32_t end = boundaries_.GetValue(request + 1);
            if (begin <= int32_t(row) && int32_t(row) < end) {
                const int32_t causal = length_.GetValue(request) - (end - begin) + int32_t(row) - begin + 1;
                visible = causal > 0 ? causal : 0;
                break;
            }
        }
        const uint64_t keyStride = tiling_.keyStride0 ? tiling_.keyStride0 : tiling_.blockSize * DIM;
        const uint64_t scaleStride = tiling_.keyDequantScaleStride0 ? tiling_.keyDequantScaleStride0 : tiling_.blockSize;
        auto offsets = offsetBuf_.Get<uint64_t>();
        auto firstPositions = positionBuf_.Get<int32_t>();
        auto scales = scaleHalfBuf_.Get<half>();
        uint32_t validSegments = 0;
        uint32_t validTiles[4] = {0, 0, 0, 0};
        for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
            Duplicate(scales, half(0), TILE * 2);
            Sync<HardEvent::V_MTE2>();
            bool any = false;
            for (uint32_t b = 0; b < TILE / BLOCK; ++b) {
                const uint32_t slot = segment * (TILE / BLOCK) + b;
                int32_t id;
                bool unique = true;
                if constexpr (TRUSTED_UNIQUE) {
                    id = input.GetValue(slot);
                } else {
                    const uint32_t sortedSlot = CANDIDATES - 1 - slot;
                    const float value = candidates.GetValue(2 * sortedSlot);
                    unique = slot == 0 || candidates.GetValue(2 * (sortedSlot + 1)) != value;
                    id = value >= 0 && value < float(int32_t(tiling_.s2Size / BLOCK))
                        ? static_cast<int32_t>(value) : -1;
                }
                bool valid = visible > 0 && unique && id >= 0 && uint32_t(id) < tiling_.s2Size / BLOCK;
                const int32_t first = valid ? id * BLOCK : 0;
                valid = valid && first < visible;
                int32_t page = -1;
                if (valid) {
                    page = table_.GetValue(uint64_t(request) * tiling_.maxBlockNumPerBatch + first / tiling_.blockSize);
                    valid = page >= 0 && uint32_t(page) < tiling_.candidatePhysicalPages;
                }
                const uint32_t within = first % tiling_.blockSize;
                offsets.SetValue(b, valid ? uint64_t(page) * keyStride + within * DIM : ~uint64_t(0));
                firstPositions.SetValue(b, valid ? first : -1);
                if (valid) {
                    any = true;
                    const uint32_t tile = slot / 16;  // 16 blocks8 = one N128 Cube tile.
                    validTiles[tile / 32] |= 1U << (tile % 32);
                    DataCopyExtParams copy{1, BLOCK * sizeof(half), 0, 0, 0};
                    DataCopyPadExtParams<half> pad{true, 0, 0, half(0)};
                    DataCopyPad(scales[b * 16], keyScale_[uint64_t(page) * scaleStride + within], copy, pad);
                }
            }
            validSegments |= uint32_t(any) << segment;
            Sync<HardEvent::S_MTE3>();
            Sync<HardEvent::MTE2_MTE3>();
            DataCopy(offsets_[segment * (TILE / BLOCK)], offsets, TILE / BLOCK);
            DataCopy(blockPositions_[segment * (TILE / BLOCK)], firstPositions, TILE / BLOCK);
            DataCopy(scales_[segment * TILE * 2], scales, TILE * 2);
            // The same small staging buffers are reused in the next segment.
            Sync<HardEvent::MTE3_S>();
            Sync<HardEvent::MTE3_V>();
        }
        input.SetValue(0, static_cast<int32_t>(validSegments));
        input.SetValue(1, visible);
        for (uint32_t word = 0; word < 4; ++word) { input.SetValue(2 + word, int32_t(validTiles[word])); }
        Sync<HardEvent::S_MTE3>();
        DataCopy(state_, input, 8);
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void ClearMailbox()
    {
        auto flag = flagBuf_.Get<int32_t>();
        Duplicate(flag, int32_t(0), 8);
        Sync<HardEvent::V_MTE3>();
        DataCopy(mailbox_[GetBlockIdx() * 8], flag, 8);
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void Relay()
    {
        IBSet<false>(mailbox_, flagBuf_.Get<int32_t>(), GetBlockIdx(), 0);
    }

    __aicore__ inline void WaitPartitions(uint32_t group, uint32_t split)
    {
        for (uint32_t part = 0; part < split; ++part) {
            IBWait<false>(mailbox_, flagBuf_.Get<int32_t>(), 2 * (group * split + part), 0);
        }
    }

    __aicore__ inline void Topk(uint32_t row)
    {
        TopkAt(row, workspace_ + uint64_t(row) * ROW_BYTES);
    }

    __aicore__ inline void TopkAt(uint32_t row, GM_ADDR record)
    {
        BindRecord(record);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(state_);
        const uint32_t validSegments = static_cast<uint32_t>(state_.GetValue(0));
        const int32_t visible = state_.GetValue(1);
        auto score = scoreBuf_.Get<float>();
        auto best = bestBuf_.Get<float>();
        auto scratch = sortBuf_.Get<float>();
        auto halfScales = scaleHalfBuf_.Get<half>();
        auto floatScales = scaleFloatBuf_.Get<float>();
        auto positions = positionBuf_.Get<int32_t>();
        auto blockPositions = inputBuf_.Get<int32_t>();
        QLIV2ServiceVec::InitSortOutBuf(best, TOPK * 2);
        for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
            if ((validSegments & (1U << segment)) == 0) { continue; }
            Sync<HardEvent::V_MTE2>();
            DataCopy(score, scores_[segment * TILE], TILE);
            DataCopy(halfScales, scales_[segment * TILE * 2], TILE * 2);
            DataCopy(blockPositions, blockPositions_[segment * (TILE / BLOCK)], TILE / BLOCK);
            Sync<HardEvent::MTE2_S>();
            for (uint32_t b = 0; b < TILE / BLOCK; ++b) {
                const int32_t first = blockPositions.GetValue(b);
                for (uint32_t p = 0; p < BLOCK; ++p) {
                    positions.SetValue(b * BLOCK + p, first >= 0 && first + p < visible ? first + p : -1);
                }
            }
            Sync<HardEvent::MTE2_V>();
            Cast(floatScales, halfScales, RoundMode::CAST_NONE, TILE * 2);
            PipeBarrier<PIPE_V>();
            for (uint32_t b = 0; b < TILE / BLOCK; ++b) {
                Mul(score[b * BLOCK], score[b * BLOCK], floatScales[b * 16], BLOCK);
            }
            auto positionFloat = candidateBuf_.Get<float>();
            auto validMask = scratch.ReinterpretCast<uint8_t>();
            Sync<HardEvent::S_V>();
            Cast(positionFloat, positions, RoundMode::CAST_NONE, TILE);
            Adds(score[TILE].ReinterpretCast<int32_t>(), positions, int32_t(0), TILE);
            PipeBarrier<PIPE_V>();
            CompareScalar(validMask, positionFloat, 0.0f, CMPMODE::GE, TILE);
            PipeBarrier<PIPE_V>();
            Select(score, validMask, score, GetScalarBitcodeValue<uint32_t, float>(0xff800000U),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE);
            PipeBarrier<PIPE_V>();
            Sort(score, scratch, TILE);
            QLIV2ServiceVec::MergeSortVecCopy(best, TOPK, score, TOPK, scratch);
        }
        QLIV2ServiceVec::ExtractIndex(positions.ReinterpretCast<uint32_t>(), best.ReinterpretCast<uint32_t>(), TOPK);
        Sync<HardEvent::V_MTE3>();
        DataCopy(output_[uint64_t(row) * TOPK], positions, TOPK);
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_V>();
    }

private:
    QLIV2TilingData tiling_;
    GM_ADDR workspace_;
    GlobalTensor<half> weights_, queryScale_, keyScale_, expandedWeights_, scales_;
    GlobalTensor<int32_t> candidates_, table_, length_, boundaries_, output_, blockPositions_, state_, mailbox_;
    GlobalTensor<uint64_t> offsets_;
    GlobalTensor<float> scores_;
    TBuf<TPosition::VECCALC> candidateBuf_, sortBuf_, inputBuf_, weightBuf_, expandedBuf_, offsetBuf_;
    TBuf<TPosition::VECCALC> scaleHalfBuf_, scaleFloatBuf_, positionBuf_, scoreBuf_, bestBuf_, flagBuf_;
};

__aicore__ inline void Run(GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryScale,
    GM_ADDR keyScale, GM_ADDR boundaries, GM_ADDR length, GM_ADDR table, GM_ADDR candidates,
    GM_ADDR output, GM_ADDR workspace, const QLIV2TilingData &tiling, TPipe *pipe)
{
    const uint32_t split = tiling.candidateSplit;
    const uint32_t producers = tiling.candidateProducerCores;
    const uint32_t groups = producers / split;
    if ASCEND_IS_AIV {
        const uint32_t core = GetBlockIdx() / 2;
        const uint32_t sub = GetBlockIdx() % 2;
        const uint32_t group = core / split;
        Vector vector;
        vector.Init(weights, queryScale, keyScale, candidates, table, length, boundaries,
                    output, workspace, tiling, pipe);
        // Preparation is balanced over all AIVs, once/query, independently of
        // the later producer groups and topk owners.
        for (uint32_t row = GetBlockIdx(); row < tiling.s1Size; row += producers * 2) {
            vector.Prepare(row);
        }
        vector.ClearMailbox();
        SyncAll();
        CrossCoreSetFlag<2, PIPE_MTE3>(READY);
        uint32_t iteration = 0;
        for (uint32_t row = group; row < tiling.s1Size; row += groups, ++iteration) {
            CrossCoreWaitFlag(SCORED);
            const bool owner = split == 1 ? sub == iteration % 2 : sub == 1 && core % split == 0;
            if (split > 1) {
                if (sub == 0) { vector.Relay(); }
                if (owner) { vector.WaitPartitions(group, split); }
            }
            // No workspace slot is recycled here. Both halves ACK once per
            // query; the producer can score the next query during this topk.
            CrossCoreSetFlag<2, PIPE_MTE3>(ACK);
            if (owner) { vector.Topk(row); }
        }
    } else {
        using Type = QLIV2Common::QLIV2Type<int8_t, int8_t, float, uint16_t, int32_t,
            true, QLIV2Common::LI_LAYOUT::TND, QLIV2Common::LI_LAYOUT::PA_BBND>;
        QLIV41CandidateCube::QLIMatmul<Type> cube;
        QLIV2Common::ConstInfo info{};
        info.gSize = info.qHeadNum = HEADS;
        info.kHeadNum = 1;
        info.headDim = DIM;
        info.s1BaseSize = 4;
        info.mBaseSize = 4 * HEADS;
        info.s2BaseSize = POSITIONS / split;
        const uint32_t group = GetBlockIdx() / split;
        const uint32_t part = GetBlockIdx() % split;
        GlobalTensor<int32_t> tableGm, state;
        GlobalTensor<int8_t> keyGm, queryGm;
        GlobalTensor<float> scoreGm;
        GlobalTensor<half> weightGm;
        GlobalTensor<uint64_t> offsets;
        tableGm.SetGlobalBuffer((__gm__ int32_t *)table);
        keyGm.SetGlobalBuffer((__gm__ int8_t *)key);
        queryGm.SetGlobalBuffer((__gm__ int8_t *)query);
        cube.InitParams(info);
        cube.InitBuffers(pipe);
        cube.AllocEventID();
        CrossCoreWaitFlag(READY);
        for (uint32_t row = group; row < tiling.s1Size; row += groups) {
            GM_ADDR record = workspace + uint64_t(row) * ROW_BYTES;
            weightGm.SetGlobalBuffer((__gm__ half *)(record + WEIGHT_OFFSET));
            state.SetGlobalBuffer((__gm__ int32_t *)(record + STATE_OFFSET));
            const uint32_t segmentMask = ((1U << (SEGMENTS / split)) - 1) << (part * (SEGMENTS / split));
            if (uint32_t(state.GetValue(0)) & segmentMask) {
                // Trim only leading/trailing all-invalid N128 tiles. Logical
                // score slots stay unchanged, including internal holes. The
                // topk owner never reads a segment marked wholly invalid.
                uint32_t live[4] = {uint32_t(state.GetValue(2)), uint32_t(state.GetValue(3)),
                                    uint32_t(state.GetValue(4)), uint32_t(state.GetValue(5))};
                uint32_t firstTile = part * (POSITIONS / 128 / split);
                uint32_t lastTile = (part + 1) * (POSITIONS / 128 / split) - 1;
                while ((live[firstTile / 32] & (1U << (firstTile % 32))) == 0) { ++firstTile; }
                while ((live[lastTile / 32] & (1U << (lastTile % 32))) == 0) { --lastTile; }
                // The Cube prologue always primes two N128 tiles. Keep that
                // invariant for a singleton, entirely within this partition.
                if (firstTile == lastTile) {
                    if (firstTile > part * (POSITIONS / 128 / split)) { --firstTile; }
                    else { ++lastTile; }
                }
                const uint32_t firstPosition = firstTile * 128;
                const uint32_t livePositions = (lastTile - firstTile + 1) * 128;
                offsets.SetGlobalBuffer((__gm__ uint64_t *)(record + OFFSET_OFFSET) + firstPosition / BLOCK);
                scoreGm.SetGlobalBuffer((__gm__ float *)(record + SCORE_OFFSET) + firstPosition);
                cube.InitMm1GlobalTensor(tableGm, keyGm, queryGm, scoreGm, weightGm);
                cube.InitCandidateOffsets(offsets,
                    tiling.keyStride0 ? tiling.keyStride0 : tiling.blockSize * DIM, tiling.blockSize);
                QLIV2Common::RunInfo run{};
                run.actMBaseSize = HEADS;
                run.actualSingleProcessSInnerSize = livePositions;
                run.actualSingleProcessSInnerSizeAlign = livePositions;
                run.isFirstS2InnerLoop = true;
                run.isLastS2InnerLoop = true;
                run.tensorQueryOffset = uint64_t(row) * HEADS * DIM;
                cube.ComputeMm1(run);
            }
            CrossCoreSetFlag<2, PIPE_FIX>(SCORED);
            CrossCoreWaitFlag(ACK);
        }
        cube.FreeEventID();
    }
}
} // namespace QLIV41Candidate
#endif
