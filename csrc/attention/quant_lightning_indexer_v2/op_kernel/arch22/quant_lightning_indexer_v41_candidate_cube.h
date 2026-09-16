/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_service_cube.h
 * \brief Candidate-only adaptation of G_W_E/ops-transformer qli_opt 2ed905f.
 * QK stays in Score L1; only reduced logits are written to GM. The original
 * non-candidate v2 Cube service is intentionally unchanged.
 */
#ifndef QUANT_LIGHTNING_INDEXER_V41_CANDIDATE_CUBE_H
#define QUANT_LIGHTNING_INDEXER_V41_CANDIDATE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"

namespace QLIV41CandidateCube {
using namespace AscendC;
using namespace QLIV2Common;
struct MmInfo {
    int64_t s2L0LoopId;
    int64_t s1gL0LoopId;
    int64_t s2L0RealSize;
    int64_t s2GmOffset;
};

template <typename QLIT>
class QLIMatmul {
public:
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;

    // Physical INT8 element offsets prepared by the paired vector core.
    __aicore__ inline void InitCandidateOffsets(const GlobalTensor<uint64_t> &offsets,
        uint64_t pageStride, uint64_t pageSize)
    {
        candidateOffsets_ = offsets;
        candidatePageStride_ = pageStride;
        candidatePageSize_ = pageSize;
    }

    __aicore__ inline QLIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm, const GlobalTensor<float> &mm1ResGm,
                                               const GlobalTensor<half> &weightWorkspaceGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLIV2Common::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};  // isSetFMatrix isSetPadding;
    static constexpr uint64_t DOUBLE_BUF_NUM = 2;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t L0C_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 4;
    static constexpr uint64_t TILES_PER_SLOT = 4;
    static constexpr uint64_t S2_PER_STEP = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QW_MTE1_MTE2_EVENT = EVENT_ID5;  // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_MTE1_EVENT = EVENT_ID4;

    static constexpr uint64_t S8_BLOCK_CUBE = 32;

    static constexpr uint64_t S1_L0C_OFFSET = 64;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S1G_BASIC_BLOCK_L1 = 256;

    static constexpr uint64_t S1G_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t TILE_SIZE = S1G_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK;
    static constexpr uint64_t SL1_BUFFER_OFFSET = TILES_PER_SLOT * TILE_SIZE;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK;
    static constexpr uint64_t WEIGHT_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * BLOCK_CUBE;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S8_16K = 16 * 1024;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_FP16_16K = 16 * 512;
    static constexpr uint64_t L0C_BUFFER_OFFSET = 64 * 256;

private:
    __aicore__ inline void WeightDmaCopy(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0RealSize, uint64_t l0Slot);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                          uint64_t l0Slot);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                         const QLIV2Common::RunInfo &runInfo, uint64_t keyBufIdx);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo &mmInfo,
                                    const QLIV2Common::RunInfo &runInfo, uint64_t keyBufIdx);
    __aicore__ inline void PrefetchKey(const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline uint64_t AcquireKey(const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void PrefetchNextAndReleaseKey(const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo,
                                                      uint64_t keySlot);
    __aicore__ inline void FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                     uint64_t s1L0LoopCnt, uint64_t scoreTileIdx, uint64_t cSlot);
    __aicore__ inline void LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                      int64_t mStartPt, uint64_t mExtension, uint64_t l0Slot);
    __aicore__ inline void LoadWeightToL0a(uint64_t s1gL1Offset, uint64_t l0Slot);
    __aicore__ inline void ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, int64_t s1gOffset,
                                     uint64_t aSlot, uint64_t bSlot, uint64_t cSlot);
    __aicore__ inline void FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize, uint64_t s1GmOffset,
                                       uint64_t s2GmOffset, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                     uint64_t aSlot, uint64_t bSlot, uint64_t cSlot);
    __aicore__ inline void ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1BufIdx,
                                     const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ProcessWsPair(uint64_t s1gL0RealSize, uint64_t s1gL1Offset,
                                         uint64_t firstSL1BufIdx,
                                         const MmInfo &firstMmInfo, const MmInfo &secondMmInfo,
                                         const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ProcessWsStagePair(const int64_t *s1gL0RealSize, const int64_t *s1gL1Offset,
                                              uint64_t swBaseTile, const MmInfo *pairInfo,
                                              const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                     const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ProcessQkPair(const int64_t *s1gL0RealSize, const int64_t *s1gL1Offset,
                                         const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void CalcMmInfo(MmInfo &mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt, const MmInfo &lastMmInfo,
                                      const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ProduceQkTiles(int64_t &loopIdx, int64_t s2TileCnt, int64_t s1L0LoopCnt,
                                          const int64_t *s1gL0RealSize, const int64_t *s1gL1Offset,
                                          MmInfo *mmInfo, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ConsumeWsFullPair(int64_t pairIdx, int64_t s1L0LoopCnt,
                                             const int64_t *s1gL0RealSize, const int64_t *s1gL1Offset,
                                             uint64_t scoreStageBase, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ConsumeWsTail(int64_t remS2Idx, int64_t s2Remainder, int64_t s1L0LoopCnt,
                                         const int64_t *s1gL0RealSize, const int64_t *s1gL1Offset,
                                         uint64_t scoreStageBase, const QLIV2Common::RunInfo &runInfo);
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    GlobalTensor<uint64_t> candidateOffsets_;
    uint64_t candidatePageStride_;
    uint64_t candidatePageSize_;
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<half> weightGm_;
    GlobalTensor<float> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A1> bufWeightL1_;
    LocalTensor<half> weightL1_;
    TBuf<TPosition::B1> bufSL1_;
    LocalTensor<half> sL1_;

    TBuf<TPosition::A2> bufL0A_;
    LocalTensor<Q_T> l0a_;
    TBuf<TPosition::B2> bufL0B_;
    LocalTensor<K_T> l0b_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<int32_t> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t keyL1Mte2BufIdx_ = 0;
    uint64_t qwL1Mte2BufIdx_ = 0;
    uint64_t sL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufWeightL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * BLOCK_CUBE * sizeof(half));
    weightL1_ = bufWeightL1_.Get<half>();
    pipe->InitBuffer(bufSL1_, DOUBLE_BUF_NUM * TILES_PER_SLOT * S2_BASIC_BLOCK_L0 * S1G_BASIC_BLOCK_L0 * sizeof(half));
    sL1_ = bufSL1_.Get<half>();

    pipe->InitBuffer(bufL0A_, 64 * 1024);
    l0a_ = bufL0A_.Get<Q_T>();
    pipe->InitBuffer(bufL0B_, 64 * 1024);
    l0b_ = bufL0B_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, 128 * 1024);
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm,
                                                            const GlobalTensor<K_T> &keyGm,
                                                            const GlobalTensor<Q_T> &queryGm,
                                                            const GlobalTensor<float> &mm1ResGm,
                                                            const GlobalTensor<half> &weightWorkspaceGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
    weightGm_ = weightWorkspaceGm;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1BufIdx,
                                                  const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo)
{
    uint64_t cSlot = l0cBufIdx_ % L0C_BUF_NUM;
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + cSlot);
    for (int64_t s1gOffset = 0; s1gOffset < s1gL0RealSize; s1gOffset += constInfo_.gSize) {
        uint64_t l0Slot = l0BufIdx_ % L0AB_BUF_NUM;
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0Slot);
        LoadSToL0b(s1gL0RealSize, mmInfo.s2L0RealSize, sL1BufIdx, s1gOffset, constInfo_.gSize, l0Slot);
        LoadWeightToL0a(s1gOffset + s1gL1Offset, l0Slot);
        ComputeWs(s1gL0RealSize, mmInfo.s2L0RealSize, s1gOffset, l0Slot, l0Slot, cSlot);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0Slot);
        l0BufIdx_++;
    }

    FixpResToGm(s1gL0RealSize / constInfo_.gSize, mmInfo.s2L0RealSize, s1gL1Offset / constInfo_.gSize,
                mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0, runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + cSlot);
    l0cBufIdx_++;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProcessWsPair(
    uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t firstSL1BufIdx,
    const MmInfo &firstMmInfo, const MmInfo &secondMmInfo, const QLIV2Common::RunInfo &runInfo)
{
    uint64_t s1Count = s1gL0RealSize / constInfo_.gSize;
    uint64_t s2RealSize = firstMmInfo.s2L0RealSize + secondMmInfo.s2L0RealSize;
    uint64_t cSlot = l0cBufIdx_ % L0C_BUF_NUM;
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + cSlot);
    for (uint64_t s1Idx = 0; s1Idx < s1Count; s1Idx++) {
        uint64_t s1gOffset = s1Idx * constInfo_.gSize;
        uint64_t firstEventSlot = ((l0BufIdx_ / 2) % (L0AB_BUF_NUM / 2)) * 2;
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + firstEventSlot);
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + firstEventSlot + 1);
        LoadSToL0b(s1gL0RealSize, 2 * S2_BASIC_BLOCK_L0, firstSL1BufIdx, s1gOffset,
                   constInfo_.gSize, firstEventSlot);
        LoadWeightToL0a(s1gOffset + s1gL1Offset, firstEventSlot);
        ComputeWs(s1gL0RealSize, 2 * S2_BASIC_BLOCK_L0, s1gOffset,
                  firstEventSlot, firstEventSlot, cSlot);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + firstEventSlot);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + firstEventSlot + 1);
        l0BufIdx_ += 2;
    }

    FixpResToGm(s1Count, s2RealSize, s1gL1Offset / constInfo_.gSize,
                firstMmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0, runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + cSlot);
    l0cBufIdx_++;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProcessWsStagePair(
    const int64_t *s1gL0RealSize, const int64_t *s1gL1Offset, uint64_t swBaseTile,
    const MmInfo *pairInfo, const QLIV2Common::RunInfo &runInfo)
{
    uint64_t s1L0LoopCnt = CeilDiv(runInfo.actMBaseSize / constInfo_.gSize,
                                   constInfo_.s1BaseSize / S2_PER_STEP);
    for (uint64_t s1g = 0; s1g < s1L0LoopCnt; s1g++) {
        ProcessWsPair(s1gL0RealSize[s1g], s1gL1Offset[s1g],
                      swBaseTile + s1g * S2_PER_STEP,
                      pairInfo[s1g], pairInfo[s1L0LoopCnt + s1g], runInfo);
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::PrefetchKey(const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo)
{
    uint64_t keyBufIdx = keyL1Mte2BufIdx_ % KEY_BUF_NUM;
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyBufIdx);
    if constexpr (K_LAYOUT_T == LI_LAYOUT::PA_BBND) {
        KeyNd2NzForPA(mmInfo.s2L0RealSize, runInfo.s2Idx * constInfo_.s2BaseSize + mmInfo.s2GmOffset, runInfo,
                      keyBufIdx);
    } else {
        KeyNd2Nz(mmInfo.s2L0RealSize, mmInfo, runInfo, keyBufIdx);
    }
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT + keyBufIdx);
    keyL1Mte2BufIdx_++;
}

template <typename QLIT>
__aicore__ inline uint64_t QLIMatmul<QLIT>::AcquireKey(const MmInfo &mmInfo,
                                                        const QLIV2Common::RunInfo &runInfo)
{
    if (keyL1Mte2BufIdx_ == keyL1BufIdx_) {
        PrefetchKey(mmInfo, runInfo);
    }
    uint64_t keySlot = keyL1BufIdx_ % KEY_BUF_NUM;
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT + keySlot);
    return keySlot;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::PrefetchNextAndReleaseKey(
    const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo, uint64_t keySlot)
{
    if (mmInfo.s2GmOffset + mmInfo.s2L0RealSize < runInfo.actualSingleProcessSInnerSize) {
        MmInfo nextMmInfo;
        nextMmInfo.s2L0LoopId = mmInfo.s2L0LoopId + 1;
        nextMmInfo.s1gL0LoopId = 0;
        nextMmInfo.s2GmOffset = mmInfo.s2GmOffset + mmInfo.s2L0RealSize;
        nextMmInfo.s2L0RealSize =
            nextMmInfo.s2GmOffset + S2_BASIC_BLOCK_L0 > runInfo.actualSingleProcessSInnerSize
                ? runInfo.actualSingleProcessSInnerSize - nextMmInfo.s2GmOffset
                : S2_BASIC_BLOCK_L0;
        PrefetchKey(nextMmInfo, runInfo);
    }
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keySlot);
    keyL1BufIdx_++;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                                  const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo)
{
    uint64_t keySlot = AcquireKey(mmInfo, runInfo);

    uint64_t l0Slot = l0BufIdx_ % L0AB_BUF_NUM;
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0Slot);
    LoadQueryToL0a(s1gL1Offset, runInfo.actMBaseSize, s1gL0RealSize, l0Slot);
    LoadKeyToL0b(mmInfo.s2L0RealSize, l0Slot);

    PrefetchNextAndReleaseKey(mmInfo, runInfo, keySlot);

    uint64_t cSlot = l0cBufIdx_ % L0C_BUF_NUM;
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + cSlot);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    ComputeQk(s1gL0RealSize, mmInfo.s2L0RealSize, l0Slot, l0Slot, cSlot);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0Slot);
    FixpSToL1(s1gL0RealSize, mmInfo.s2L0RealSize, s1L0LoopCnt, sL1BufIdx_, cSlot);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + cSlot);
    sL1BufIdx_++;
    l0BufIdx_++;
    l0cBufIdx_++;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProcessQkPair(
    const int64_t *s1gL0RealSize, const int64_t *s1gL1Offset,
    const MmInfo &mmInfo, const QLIV2Common::RunInfo &runInfo)
{
    uint64_t keySlot = AcquireKey(mmInfo, runInfo);

    uint64_t firstL0Slot = l0BufIdx_ % L0AB_BUF_NUM;
    uint64_t secondL0Slot = (l0BufIdx_ + 1) % L0AB_BUF_NUM;
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + firstL0Slot);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + secondL0Slot);
    LoadQueryToL0a(s1gL1Offset[0], runInfo.actMBaseSize, s1gL0RealSize[0], firstL0Slot);
    LoadQueryToL0a(s1gL1Offset[1], runInfo.actMBaseSize, s1gL0RealSize[1], secondL0Slot);
    LoadKeyToL0b(mmInfo.s2L0RealSize, firstL0Slot);
    PrefetchNextAndReleaseKey(mmInfo, runInfo, keySlot);

    uint64_t firstCSlot = l0cBufIdx_ % L0C_BUF_NUM;
    uint64_t secondCSlot = (l0cBufIdx_ + 1) % L0C_BUF_NUM;
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + firstCSlot);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

    ComputeQk(s1gL0RealSize[0], mmInfo.s2L0RealSize, firstL0Slot, firstL0Slot, firstCSlot);
    FixpSToL1(s1gL0RealSize[0], mmInfo.s2L0RealSize, 2, sL1BufIdx_, firstCSlot);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + firstCSlot);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + secondCSlot);
    ComputeQk(s1gL0RealSize[1], mmInfo.s2L0RealSize, secondL0Slot, firstL0Slot, secondCSlot);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + firstL0Slot);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + secondL0Slot);
    FixpSToL1(s1gL0RealSize[1], mmInfo.s2L0RealSize, 2, sL1BufIdx_ + 1, secondCSlot);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + secondCSlot);

    sL1BufIdx_ += 2;
    l0BufIdx_ += 2;
    l0cBufIdx_ += 2;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::CalcMmInfo(MmInfo &mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt,
                                                   const MmInfo &lastMmInfo, const QLIV2Common::RunInfo &runInfo)
{
    mmInfo.s2L0LoopId = loopIdx / s1L0LoopCnt;
    mmInfo.s1gL0LoopId = loopIdx % s1L0LoopCnt;

    if (mmInfo.s1gL0LoopId == 0) {
        mmInfo.s2GmOffset = mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0;
        mmInfo.s2L0RealSize = mmInfo.s2GmOffset + S2_BASIC_BLOCK_L0 > runInfo.actualSingleProcessSInnerSize
                                  ? runInfo.actualSingleProcessSInnerSize - mmInfo.s2GmOffset
                                  : S2_BASIC_BLOCK_L0;
    } else {
        mmInfo.s2L0RealSize = lastMmInfo.s2L0RealSize;
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProduceQkTiles(int64_t &loopIdx, int64_t s2TileCnt, int64_t s1L0LoopCnt,
                                                       const int64_t *s1gL0RealSize,
                                                       const int64_t *s1gL1Offset, MmInfo *mmInfo,
                                                       const QLIV2Common::RunInfo &runInfo)
{
    for (int64_t s2Sub = 0; s2Sub < s2TileCnt; s2Sub++) {
        if (s1L0LoopCnt == 2) {
            uint64_t cur = static_cast<uint64_t>(loopIdx) & 1;
            CalcMmInfo(mmInfo[cur], loopIdx, s1L0LoopCnt, mmInfo[cur ^ 1], runInfo);
            ProcessQkPair(s1gL0RealSize, s1gL1Offset, mmInfo[cur], runInfo);
            loopIdx += 2;
            continue;
        }
        for (int64_t s1g = 0; s1g < s1L0LoopCnt; s1g++) {
            uint64_t cur = static_cast<uint64_t>(loopIdx) & 1;
            CalcMmInfo(mmInfo[cur], loopIdx, s1L0LoopCnt, mmInfo[cur ^ 1], runInfo);
            uint64_t s1gIdx = mmInfo[cur].s1gL0LoopId;
            ProcessQk(s1gL0RealSize[s1gIdx], s1gL1Offset[s1gIdx], s1L0LoopCnt, mmInfo[cur], runInfo);
            loopIdx++;
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ConsumeWsFullPair(int64_t pairIdx, int64_t s1L0LoopCnt,
                                                          const int64_t *s1gL0RealSize,
                                                          const int64_t *s1gL1Offset,
                                                          uint64_t scoreStageBase,
                                                          const QLIV2Common::RunInfo &runInfo)
{
    int64_t swBaseTile = scoreStageBase + pairIdx * static_cast<int64_t>(TILES_PER_SLOT);
    int64_t swLoopBase = pairIdx * S2_PER_STEP * s1L0LoopCnt;
    MmInfo pairInfo[TILES_PER_SLOT];
    for (int64_t idx = 0; idx < S2_PER_STEP * s1L0LoopCnt; idx++) {
        CalcMmInfo(pairInfo[idx], swLoopBase + idx, s1L0LoopCnt,
                   pairInfo[idx == 0 ? 0 : idx - 1], runInfo);
    }
    ProcessWsStagePair(s1gL0RealSize, s1gL1Offset, swBaseTile, pairInfo, runInfo);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ConsumeWsTail(int64_t remS2Idx, int64_t s2Remainder,
                                                      int64_t s1L0LoopCnt,
                                                      const int64_t *s1gL0RealSize,
                                                      const int64_t *s1gL1Offset,
                                                      uint64_t scoreStageBase,
                                                      const QLIV2Common::RunInfo &runInfo)
{
    MmInfo mmInfo[2];
    int64_t remSwLoopIdx = remS2Idx * s1L0LoopCnt;
    for (int64_t remSub = 0; remSub < s2Remainder; remSub++) {
        for (int64_t s1g = 0; s1g < s1L0LoopCnt; s1g++) {
            uint64_t cur = static_cast<uint64_t>(remSwLoopIdx) & 1;
            CalcMmInfo(mmInfo[cur], remSwLoopIdx, s1L0LoopCnt, mmInfo[cur ^ 1], runInfo);
            uint64_t s1gIdx = mmInfo[cur].s1gL0LoopId;
            ProcessWs(s1gL0RealSize[s1gIdx], s1gL1Offset[s1gIdx],
                      scoreStageBase + static_cast<uint64_t>(
                          remS2Idx / S2_PER_STEP * TILES_PER_SLOT + s1g * S2_PER_STEP + remSub),
                      mmInfo[cur], runInfo);
            remSwLoopIdx++;
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeMm1(const QLIV2Common::RunInfo &runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        QueryNd2Nz(runInfo.actMBaseSize, runInfo);  // 256 * 128 // L1BasicBlock
        WeightDmaCopy(runInfo.actMBaseSize, runInfo);
    }
    int64_t s2L0LoopCnt = CeilDiv(runInfo.actualSingleProcessSInnerSize, S2_BASIC_BLOCK_L0);  // 2048取128
    int64_t s1L0LoopCnt = CeilDiv(runInfo.actMBaseSize / constInfo_.gSize, constInfo_.s1BaseSize / 2);
    int64_t s1gL1Offset[2] = {0, static_cast<int64_t>(constInfo_.gSize * constInfo_.s1BaseSize / 2)};
    int64_t s1gL0RealSize[2] = {s1L0LoopCnt > 1 ? static_cast<int64_t>(constInfo_.gSize * constInfo_.s1BaseSize / 2) : runInfo.actMBaseSize,
                                runInfo.actMBaseSize - s1gL1Offset[1]};

    int64_t s2Pairs = s2L0LoopCnt / S2_PER_STEP;
    int64_t s2Remainder = s2L0LoopCnt % S2_PER_STEP;
    MmInfo mmInfo[2];
    uint64_t scoreStageBase = sL1BufIdx_;
    uint64_t firstScoreSlot = (scoreStageBase / TILES_PER_SLOT) % DOUBLE_BUF_NUM;

    // Prologue: produce the first full score stage, i.e. S2_PER_STEP S2 blocks.
    int64_t loopIdx = 0;
    ProduceQkTiles(loopIdx, S2_PER_STEP, s1L0LoopCnt, s1gL0RealSize, s1gL1Offset, mmInfo, runInfo);
    // A stage owns a complete Score slot. Decode has only one M tile, so it
    // produces two real tiles per stage; advance over the unused half before
    // publishing the slot, otherwise the next stage overwrites it.
    sL1BufIdx_ = CeilDiv(sL1BufIdx_, TILES_PER_SLOT) * TILES_PER_SLOT;
    SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + firstScoreSlot);
    int64_t qkSlotIdx = 1 - firstScoreSlot;

    // Pipeline: produce the current QK tiles while consuming the previous score tiles.
    for (int64_t pair = 1; pair < s2Pairs; pair++) {
        ProduceQkTiles(loopIdx, S2_PER_STEP, s1L0LoopCnt, s1gL0RealSize, s1gL1Offset, mmInfo, runInfo);
        sL1BufIdx_ = CeilDiv(sL1BufIdx_, TILES_PER_SLOT) * TILES_PER_SLOT;
        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + qkSlotIdx);
        int64_t swSlot = 1 - qkSlotIdx;
        qkSlotIdx = 1 - qkSlotIdx;
        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + swSlot);

        ConsumeWsFullPair(pair - 1, s1L0LoopCnt, s1gL0RealSize, s1gL1Offset, scoreStageBase, runInfo);
    }

    // Produce the partial stage in the free score slot before consuming the
    // last full stage, so Tail QK/FixPipe overlaps that SW stage.
    int64_t remS2Idx = s2Pairs * S2_PER_STEP;
    uint64_t tailSlot = (firstScoreSlot + s2Pairs) % DOUBLE_BUF_NUM;
    if (s2Remainder > 0) {
        int64_t remLoopIdx = remS2Idx * s1L0LoopCnt;
        ProduceQkTiles(remLoopIdx, s2Remainder, s1L0LoopCnt, s1gL0RealSize, s1gL1Offset, mmInfo, runInfo);
        sL1BufIdx_ = CeilDiv(sL1BufIdx_, TILES_PER_SLOT) * TILES_PER_SLOT;
        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + tailSlot);
    }

    // Epilogue: consume the last complete pair.
    if (s2Pairs > 0) {
        int64_t swSlot = 1 - qkSlotIdx;
        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + swSlot);
        ConsumeWsFullPair(s2Pairs - 1, s1L0LoopCnt, s1gL0RealSize, s1gL1Offset, scoreStageBase, runInfo);
    }

    // Consume the partial stage after the final full-stage SW.
    if (s2Remainder > 0) {
        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + tailSlot);
        ConsumeWsTail(remS2Idx, s2Remainder, s1L0LoopCnt, s1gL0RealSize, s1gL1Offset, scoreStageBase, runInfo);
    }

    // A partial stage occupies a complete logical Score slot. Keep the next
    // outer iteration aligned with the two-slot event state machine.
    sL1BufIdx_ = CeilDiv(sL1BufIdx_, TILES_PER_SLOT) * TILES_PER_SLOT;

    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        qwL1Mte2BufIdx_++;
    }
}

// blkNum, blkSize, N2, D
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                      const QLIV2Common::RunInfo &runInfo, uint64_t keyBufIdx)
{
    // Merge adjacent physical blocks into one ND2NZ transfer. Runs never
    // cross a page or the current L1 tile. Invalid columns need no key load:
    // INT8 lanes cannot create NaNs and the vector epilogue masks those IDs.
    for (uint64_t offset = 0; offset < s2L1RealSize;) {
        const uint64_t keyOffset = candidateOffsets_.GetValue((s2GmOffset + offset) / 8);
        if (keyOffset == ~uint64_t(0)) {
            offset += 8;
            continue;
        }
        uint64_t rows = 8;
        const uint64_t withinPage = (keyOffset % candidatePageStride_) / constInfo_.headDim;
        while (offset + rows + 8 <= s2L1RealSize && withinPage + rows + 8 <= candidatePageSize_) {
            const uint64_t next = candidateOffsets_.GetValue((s2GmOffset + offset + rows) / 8);
            if (next != keyOffset + rows * constInfo_.headDim) { break; }
            rows += 8;
        }
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = rows;
        params.dValue = constInfo_.headDim;
        params.srcDValue = constInfo_.headDim;
        params.dstNzC0Stride = CeilAlign(s2L1RealSize, uint64_t(BLOCK_CUBE));
        params.dstNzNStride = 1;
        params.srcNdMatrixStride = 0;
        params.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyBufIdx % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + offset * S8_BLOCK_CUBE],
                 keyGm_[keyOffset], params);
        offset += rows;
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo &mmInfo,
                                                 const QLIV2Common::RunInfo &runInfo, uint64_t keyBufIdx)
{
    uint64_t dStride = constInfo_.headDim;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::BSND || K_LAYOUT_T == LI_LAYOUT::TND) {
        dStride = constInfo_.headDim * constInfo_.kHeadNum; // constInfo_.kHeadNum
    }
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize;  // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = dStride;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[keyBufIdx * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + mmInfo.s2GmOffset * constInfo_.headDim], nd2nzPara);
}

// batch, s1, g, 1
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::WeightDmaCopy(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo &runInfo)
{
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = s1gL1RealSize;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopy(weightL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * WEIGHT_BUFFER_OFFSET],
             weightGm_[runInfo.loop % DOUBLE_BUF_NUM * BLOCK_CUBE * constInfo_.mBaseSize], copyInParams);
}

// batch, s1, n2, g, d
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::QueryNd2Nz(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize;  // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET], queryGm_[runInfo.tensorQueryOffset],
             nd2nzPara);
}

// s1g, d
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                       uint64_t s1gL0RealSize, uint64_t l0Slot)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(s1gL1RealSize, BLOCK_CUBE);  // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                          // Win=M0
    loadData3DParams.channelSize = constInfo_.headDim;          // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = s1gL0RealSize;                         // M height维度目的
    loadData3DParams.kExtension = constInfo_.headDim;                    // K   width维度目的
    loadData3DParams.mStartPt = s1gL1Offset;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(
        l0a_[l0Slot * L0AB_BUFFER_OFFSET_S8_16K],
        queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData3DParams);
}

// s1, g, s2  -->  2 * 64* 128
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize,
                                                   uint64_t sL1BufIdx, int64_t mStartPt,
                                                   uint64_t mExtension, uint64_t l0Slot)
{
    LoadData3DParamsV2<half> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = S1G_BASIC_BLOCK_L0 / BLOCK_CUBE;              // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                                   // Win=M0
    loadData3DParams.channelSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);  // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = mExtension;                           // M height维度目的
    loadData3DParams.kExtension = CeilAlign(s2L0RealSize, BLOCK_CUBE);  // K   width维度目的
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 1;
    loadData3DParams.fMatrixCtrl = 0;

    loadData3DParams.mStartPt = mStartPt;
    uint64_t slot = (sL1BufIdx / TILES_PER_SLOT) % DOUBLE_BUF_NUM;
    uint64_t tileOff = (sL1BufIdx % TILES_PER_SLOT) * TILE_SIZE;
    LoadData<half, LOAD3DV2_CONFIG>(
        l0b_.template ReinterpretCast<half>()[l0Slot * L0AB_BUFFER_OFFSET_FP16_16K],
        sL1_[slot * SL1_BUFFER_OFFSET + tileOff], loadData3DParams);
}

// s1,g,1(16), 2,64,16
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadWeightToL0a(uint64_t s1gL1Offset, uint64_t l0Slot)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(constInfo_.gSize, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;
    LoadData(l0a_.template ReinterpretCast<half>()[l0Slot * L0AB_BUFFER_OFFSET_FP16_16K],
             weightL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * WEIGHT_BUFFER_OFFSET + s1gL1Offset* BLOCK_CUBE],
             loadData2DParams);
}

// s2, d -> 128,128
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadKeyToL0b(uint64_t s2L0RealSize, uint64_t l0Slot)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S8_BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0b_[l0Slot * L0AB_BUFFER_OFFSET_S8_16K],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParams);
}

// A: s1,g,1(16) B: s1,g,s2  C: s1, 1(16), s2
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                  int64_t s1gOffset, uint64_t aSlot, uint64_t bSlot,
                                                  uint64_t cSlot)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = BLOCK_CUBE;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.gSize;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    uint32_t s1gOffsetNum = s1gOffset / constInfo_.gSize;
    Mmad(cL0_.template ReinterpretCast<float>()[cSlot * L0C_BUFFER_OFFSET +
                                                s1gOffsetNum * S1_L0C_OFFSET * S2_BASIC_BLOCK_L0],
            l0a_.template ReinterpretCast<half>()[aSlot * L0AB_BUFFER_OFFSET_FP16_16K],
            l0b_.template ReinterpretCast<half>()[bSlot * L0AB_BUFFER_OFFSET_FP16_16K],
            mmadParams);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                  uint64_t aSlot, uint64_t bSlot, uint64_t cSlot)
{
    MmadParams params;
    params.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    params.n = s2L0RealSize;
    params.k = constInfo_.headDim;
    params.cmatrixInitVal = true;
    params.cmatrixSource = false;
    params.unitFlag = 0b11;
    Mmad(cL0_[cSlot * L0C_BUFFER_OFFSET],
         l0a_[aSlot * L0AB_BUFFER_OFFSET_S8_16K],
         l0b_[bSlot * L0AB_BUFFER_OFFSET_S8_16K], params);
    if ((params.m / 16) * (params.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                  uint64_t s1L0LoopCnt, uint64_t scoreTileIdx,
                                                  uint64_t cSlot)
{
    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = S1G_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::DEQF16;
    params.unitFlag = 0b11;
    params.reluPre = 1;
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3a800000);
    uint64_t slot = (scoreTileIdx / TILES_PER_SLOT) % DOUBLE_BUF_NUM;
    uint64_t logicalTile = scoreTileIdx % TILES_PER_SLOT;
    uint64_t s2Sub = logicalTile / s1L0LoopCnt;
    uint64_t s1Sub = logicalTile % s1L0LoopCnt;
    uint64_t physicalTile = s1Sub * S2_PER_STEP + s2Sub;
    uint64_t tileOff = physicalTile * TILE_SIZE;
    DataCopy(sL1_[slot * SL1_BUFFER_OFFSET + tileOff], cL0_[cSlot * L0C_BUFFER_OFFSET], params);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize, uint64_t s1GmOffset,
                                                    uint64_t s2GmOffset, const QLIV2Common::RunInfo &runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = 1;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = constInfo_.s2BaseSize;
    intriParams.srcStride = 16;
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.reluPre = 0;
    AscendC::SetFixpipeNz2ndFlag(s1L0RealCount, CeilDiv(constInfo_.gSize, BLOCK_CUBE) * S2_BASIC_BLOCK_L0 / BLOCK_CUBE,
                                 constInfo_.s2BaseSize);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSize / constInfo_.gSize * constInfo_.s2BaseSize +
                                s1GmOffset * intriParams.dstStride + s2GmOffset],
                      cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % L0C_BUF_NUM) * L0C_BUFFER_OFFSET],
                      intriParams);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
    SetMMLayoutTransform(false);
}
}  // namespace QLIV41CandidateCube
#endif
