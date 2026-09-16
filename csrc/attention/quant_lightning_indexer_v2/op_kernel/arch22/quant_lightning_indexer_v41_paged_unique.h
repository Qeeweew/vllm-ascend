// SPDX-License-Identifier: Apache-2.0
#ifndef QUANT_LIGHTNING_INDEXER_V41_PAGED_UNIQUE_H
#define QUANT_LIGHTNING_INDEXER_V41_PAGED_UNIQUE_H

#include "quant_lightning_indexer_v41_candidate_fused.h"

namespace QLIV41PagedUnique {
using namespace AscendC;
using namespace QLIV41Candidate;
constexpr uint32_t SLOTS = 2;
constexpr uint32_t METADATA_READY = 6;
constexpr uint32_t FREE = 9;

// A slot contains only candidate offsets, positions, scales, expanded weights
// and reduced scores. There is no K staging buffer and no AIV access to K.
// The metadata producer and the full-query top-k owner share the established
// two-slot ownership protocol. READY refers only to the small metadata record.
__aicore__ inline void Run(GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryScale,
    GM_ADDR keyScale, GM_ADDR boundaries, GM_ADDR length, GM_ADDR table, GM_ADDR candidates,
    GM_ADDR output, GM_ADDR workspace, const QLIV2TilingData &tiling, TPipe *pipe)
{
    const uint32_t cores = tiling.candidateProducerCores;
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV { core /= 2; }
    const uint32_t count = (tiling.s1Size - 1 - core) / cores + 1;
    GM_ADDR records = workspace + uint64_t(core) * SLOTS * ROW_BYTES;
    if ASCEND_IS_AIV {
        if (GetBlockIdx() % 2 == 0) {
            Vector metadata;
            metadata.Init(weights, queryScale, keyScale, candidates, table, length, boundaries,
                          output, workspace, tiling, pipe);
            for (uint32_t i = 0; i < count; ++i) {
                if (i >= SLOTS) { CrossCoreWaitFlag(FREE); }
                metadata.PrepareAt<true>(core + i * cores, records + (i % SLOTS) * ROW_BYTES);
                CrossCoreSetFlag<2, PIPE_MTE3>(METADATA_READY);
                // Other ACK half comes only after the topk consumer is done.
                CrossCoreSetFlag<2, PIPE_MTE3>(ACK);
                // One-generation delay allows metadata(i+1) to overlap Cube(i).
                if (i > 0) { CrossCoreWaitFlag(SCORED); }
            }
            CrossCoreWaitFlag(SCORED);
        } else {
            Vector vector;
            vector.Init(weights, queryScale, keyScale, candidates, table, length, boundaries,
                        output, workspace, tiling, pipe);
            for (uint32_t i = 0; i < SLOTS && i < count; ++i) { CrossCoreSetFlag<2, PIPE_MTE3>(METADATA_READY); }
            for (uint32_t i = 0; i < count; ++i) {
                if (i >= SLOTS) { CrossCoreWaitFlag(FREE); }
                CrossCoreWaitFlag(SCORED);
                vector.TopkAt(core + i * cores, records + (i % SLOTS) * ROW_BYTES);
                CrossCoreSetFlag<2, PIPE_MTE3>(ACK);
                if (i + SLOTS < count) { CrossCoreSetFlag<2, PIPE_MTE3>(METADATA_READY); }
            }
        }
    } else {
        // Each query keeps its own candidate mapping. K moves only from the
        // original paged GM cache into Cube L1. QK, ReLU and the weighted head
        // reduction use the established two-matmul Cube service.
        using Type = QLIV2Common::QLIV2Type<int8_t, int8_t, float, uint16_t, int32_t,
            true, QLIV2Common::LI_LAYOUT::TND, QLIV2Common::LI_LAYOUT::PA_BBND>;
        using Cube = QLIV41CandidateCube::QLIMatmul<Type>;
        Cube cube;
        QLIV2Common::ConstInfo info{};
        info.gSize = info.qHeadNum = HEADS;
        info.kHeadNum = 1;
        info.headDim = DIM;
        info.s1BaseSize = 4;
        info.mBaseSize = 4 * HEADS;
        info.s2BaseSize = POSITIONS;
        GlobalTensor<int32_t> tableGm, state;
        GlobalTensor<int8_t> keyGm, queryGm;
        GlobalTensor<float> scoreGm;
        GlobalTensor<half> weightGm;
        GlobalTensor<uint64_t> offsets;
        tableGm.SetGlobalBuffer((__gm__ int32_t *)table);
        queryGm.SetGlobalBuffer((__gm__ int8_t *)query);
        keyGm.SetGlobalBuffer((__gm__ int8_t *)key);
        cube.InitParams(info);
        cube.InitBuffers(pipe);
        cube.AllocEventID();
        for (uint32_t i = 0; i < count; ++i) {
            if (i >= SLOTS) {
                CrossCoreWaitFlag(ACK);
                CrossCoreSetFlag<2, PIPE_FIX>(FREE);
            }
            CrossCoreWaitFlag(METADATA_READY);
            GM_ADDR record = records + (i % SLOTS) * ROW_BYTES;
            state.SetGlobalBuffer((__gm__ int32_t *)(record + STATE_OFFSET));
            // State and offset records are recycled. Invalidate AIC scalar
            // GM cache before reading this generation's metadata; ND2NZ K
            // transfers use MTE2 and never write the source cache.
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(state);
            if (uint32_t(state.GetValue(0))) {
                // Preserve candidate slots, including internal holes. Trim
                // only all-invalid N128 tiles at the edges. The baseline
                // Cube pipeline primes two N128 tiles.
                uint32_t live[4] = {uint32_t(state.GetValue(2)), uint32_t(state.GetValue(3)),
                                    uint32_t(state.GetValue(4)), uint32_t(state.GetValue(5))};
                uint32_t firstTile = 0;
                uint32_t lastTile = POSITIONS / 128 - 1;
                while ((live[firstTile / 32] & (1U << (firstTile % 32))) == 0) { ++firstTile; }
                while ((live[lastTile / 32] & (1U << (lastTile % 32))) == 0) { --lastTile; }
                if (firstTile == lastTile) {
                    if (firstTile > 0) { --firstTile; }
                    else { ++lastTile; }
                }
                const uint32_t firstPosition = firstTile * 128;
                const uint32_t active = (lastTile - firstTile + 1) * 128;
                weightGm.SetGlobalBuffer((__gm__ half *)(record + WEIGHT_OFFSET));
                scoreGm.SetGlobalBuffer((__gm__ float *)(record + SCORE_OFFSET) + firstPosition);
                offsets.SetGlobalBuffer((__gm__ uint64_t *)(record + OFFSET_OFFSET) + firstPosition / BLOCK);
                cube.InitMm1GlobalTensor(tableGm, keyGm, queryGm, scoreGm, weightGm);
                cube.InitCandidateOffsets(offsets,
                    tiling.keyStride0 ? tiling.keyStride0 : tiling.blockSize * DIM, tiling.blockSize);
                QLIV2Common::RunInfo run{};
                run.actMBaseSize = HEADS;
                run.actualSingleProcessSInnerSize = active;
                run.actualSingleProcessSInnerSizeAlign = active;
                run.isFirstS2InnerLoop = true;
                run.isLastS2InnerLoop = true;
                run.tensorQueryOffset = uint64_t(core + i * cores) * HEADS * DIM;
                cube.ComputeMm1(run);
            }
            CrossCoreSetFlag<2, PIPE_FIX>(SCORED);
        }
        // Every MODE2 token is consumed, including the last one/two slots.
        for (uint32_t i = 0; i < SLOTS && i < count; ++i) { CrossCoreWaitFlag(ACK); }
        cube.FreeEventID();
    }
}
} // namespace QLIV41PagedUnique
#endif
