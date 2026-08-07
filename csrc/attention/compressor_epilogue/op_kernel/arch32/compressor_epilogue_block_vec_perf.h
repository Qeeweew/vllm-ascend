/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_epilogue_block_vec_perf.h
 * \brief
 */

#ifndef COMPRESSOR_EPILOGUE_BLOCK_VEC_PERF_H
#define COMPRESSOR_EPILOGUE_BLOCK_VEC_PERF_H

#include "compressor_epilogue_comm.h"
#include "compressor_epilogue_tools.h"
#include "compressor_epilogue_vector_comm.h"
#include "rms_norm.h"
#include "rope.h"
#include "soft_max.h"


using namespace AscendC;

namespace CompressorEpilogue {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

struct Vec1SplitInfo {
    uint32_t dealSeqStartIdx = 0;
    uint32_t dBaseSize = 0;
    uint32_t vec1GroupSize = 0;
    uint32_t vec1GroupNum = 0;
    uint32_t dealTcSize = 0;
    uint32_t preDealTcSize = 0;
    uint32_t curBStart = 0;
    uint32_t curSStart = 0;
    uint32_t curCompressedCnt = 0;
    uint32_t totalCompressedCnt = 0;
    uint32_t tcSplitSize = 0;
    uint32_t dSplitSize = 0;
    uint32_t dLoopCount = 0;
};


template <typename COMP>
class CompressorEpilogueBlockVectorPerf {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    float SOFTMAX_MIN_NUM = static_cast<float>(-1.0 / 0.0);
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;
    using ROPE_T = typename AscendC::Conditional<COMP::ropeDtype == ROPE_DTYPE::FP32, float, X_T>::type;

    __aicore__ inline CompressorEpilogueBlockVectorPerf(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo, const CompressorEpilogueTools<COMP> &tools);
    __aicore__ inline void Init(
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *stateBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);
    __aicore__ inline uint32_t GetScSize();
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<X_T> kvMmGm, GlobalTensor<X_T> scoreMmGm);

protected:
    // MatMul 结果直接来自用户输入的 GEMM 输出（[token, coff, headDim] 展平），替代原 cube 写的 workspace
    GlobalTensor<X_T> mmScoreGm_;
    GlobalTensor<X_T> mmKvGm_;

private:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline void CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd, uint32_t scEnd,
                                             uint64_t &globalScStart);
    __aicore__ inline void UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &dealScSize,
                                           uint32_t &curDealScSize);
    __aicore__ inline void DealVec1BaseBlock(const Vec1RunInfo &info, CompressorEpilogueVec1SliceIterator<COMP> &sliceIterator,
                                             uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t dBaseSize);
    __aicore__ inline void CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void AddApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                         const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    __aicore__ inline void AddSingleApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                               const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToUb(const LocalTensor<O> dstLocal, const LocalTensor<O> srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignGmToUb(const LocalTensor<O> dstLocal, const GlobalTensor<O> srcGm,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToGm(const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    __aicore__ inline void PadAlign(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                   const GlobalTensor<X_T> &srcGm, const GlobalTensor<T> &stateGm,
                                   const GlobalTensor<int32_t> &blockTableGm,
                                   const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                   uint32_t globalSeqIdx, uint32_t dDealSize);
    // 输入双缓冲：CopyInMm 只发射 GM->UB（MTE2），同步由 queue EnQue/DeQue 承担；
    // CastMm 在 DeQue 之后执行纯 V 侧 Cast
    __aicore__ inline void CopyInMm(const LocalTensor<T> &dstLocal, const GlobalTensor<X_T> &srcGm,
                                    const Vec1SliceInfo &sliceInfo, const StatisticInfo &statisticInfo,
                                    uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CastMm(const LocalTensor<T> &dstLocal, const StatisticInfo &statisticInfo,
                                  uint32_t dDealSize);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
                                             const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx,
                                             uint32_t endSeqIdx, uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state,
                                              const GlobalTensor<int32_t> &blockTableGm, uint32_t batchIdx,
                                              uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                              uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void LoadFromWorkSpace(const LocalTensor<T> dstLocal,
                                             const GlobalTensor<X_T> &srcGm, const LocalTensor<T> srcLocal,
                                             const Vec1SliceInfo &sliceInfo,
                                             uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void SoftmaxDN(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                     uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpUb,
                                            uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal,
                                          const Vec1RunInfo &info,
                                          const StatisticInfo &statisticInfo,
                                          const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize,
                                          uint32_t dBaseSize, uint32_t needDealTcSize);
    __aicore__ inline void FinishCompressedRows(const LocalTensor<T> &compressedUb, uint32_t scCnt,
                                                const LocalTensor<T> &tmpUb);
    __aicore__ inline void CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1(const Vec1RunInfo &info);
    __aicore__ inline void CopyFinalResultOut(const LocalTensor<X_T> &cmpKvOutUb, uint32_t dealRowCount);
    __aicore__ inline void SingleCalRope(const LocalTensor<X_T> &outputUb, const LocalTensor<T> &normResUb,
                                         uint32_t rowCnt, uint32_t curDealScSize, uint32_t globalScStart);
    __aicore__ inline void SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    template <bool IS_SCORE>
    __aicore__ inline void DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
                                               uint32_t duplicateColCount, uint32_t singleRowCount);
    template <bool IS_SCORE>
    __aicore__ inline void ReadState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t compressedCnt_ = 0;
    bool isExistSeqUsed = false;
    bool isExistStartPos = false;
    // vec2
    uint32_t v2MBaseSize = 16; // 行块大小：32 * 1024 / (512 * 4)
    uint32_t OutputBStartIdx, OutputSStartIdx, OutputSize;
    bool v2OutInited = false;  // 输出游标（BSH 布局）是否已按核起点初始化
    CompressorEpilogueTools<COMP> tools_;
    ConstInfo constInfo_ = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> stateBlockTableGm_;
    GlobalTensor<T> stateCacheGm_;
    GlobalTensor<T> apeGm_;
    GlobalTensor<X_T> normWeightGm_;
    GlobalTensor<ROPE_T> ropeSinGm_;
    GlobalTensor<ROPE_T> ropeCosGm_;
    GlobalTensor<X_T> cmpKvOutGm_;

    // ================================Local Buffer区====================================
    LocalTensor<T> normWeightUb;
    LocalTensor<T> apeUb;
    LocalTensor<uint32_t> gatherOffsetCastUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    TBuf<TPosition::VECCALC> gatherOffsetBuf;
    TBuf<TPosition::VECCALC> apeBuf;
    // in queue：score/kv 独立队列实现输入双缓冲（kv GM copy 与 score Cast/OverLap 重叠）
    TQue<QuePosition::VECIN, 1> inputQueScore;
    TQue<QuePosition::VECIN, 1> inputQueKv;
    TBuf<TPosition::VECIN> normWeightBuf;
};


template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::InitParams(const ConstInfo &constInfo,
                                                                   const CompressorEpilogueTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    v2MBaseSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.headDim * sizeof(float));
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::Init(
    __gm__ uint8_t *stateCache,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *cmpKvOut)
{
    stateBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)stateBlockTable);
    stateCacheGm_.SetGlobalBuffer((__gm__ T *)stateCache);
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
    normWeightGm_.SetGlobalBuffer((__gm__ X_T *)normWeight);
    ropeSinGm_.SetGlobalBuffer((__gm__ ROPE_T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ ROPE_T *)ropeCos);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ X_T *)cmpKvOut);
    isExistSeqUsed = (seqUsed != nullptr);
    isExistStartPos = (startPos != nullptr);
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if (isExistStartPos) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
    coff_ = static_cast<uint32_t>(COMP::coff);
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQueScore, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQueKv, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(normWeightBuf, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(gatherOffsetBuf, BUFFER_SIZE_BYTE_1K);
    // ape 实际用量 coff*cmpRatio*dDealSize fp32 = 16KB（buf 按 UB 预算收缩）
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_16K);
    normWeightUb = normWeightBuf.Get<T>();
    apeUb = apeBuf.Get<T>();
    LocalTensor<X_T> normweightInUb = inputQueScore.AllocTensor<X_T>();
    LocalTensor<int32_t> gatherOffsetUb = gatherOffsetBuf.Get<int32_t>();
    DataCopy(normweightInUb, normWeightGm_, constInfo_.headDim); // 获取normWeight，常驻
    inputQueScore.EnQue(normweightInUb);
    inputQueScore.DeQue<X_T>();
    Cast(normWeightUb, normweightInUb, RoundMode::CAST_NONE, constInfo_.headDim);
    inputQueScore.FreeTensor(normweightInUb);
    if constexpr (COMP::rotaryMode == CompressorEpilogue::ROTARY_MODE::INTERLEAVE) {
        SetGatherSrcOffset<float>(gatherOffsetUb, constInfo_.ropeHeadDim);
    }
    gatherOffsetCastUb = gatherOffsetUb.ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::InitVec1GlobalTensor(GlobalTensor<X_T> kvMmGm, GlobalTensor<X_T> scoreMmGm)
{
    this->mmKvGm_ = kvMmGm;
    this->mmScoreGm_ = scoreMmGm;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueBlockVectorPerf<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return constInfo_.sSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueBlockVectorPerf<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueBlockVectorPerf<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (isExistSeqUsed) {
        return sequsedGm_.GetValue(bIdx);
    } else if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}


template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueBlockVectorPerf<COMP>::GetScSize()
{
    uint32_t curBasicNum = (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx,
                                                                  uint32_t dDealSize)
{
    LocalTensor<T> apeUbTmp = inputQueScore.AllocTensor<T>();

    uint32_t copyRowCount = coff_ * constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = dDealSize;
    uint32_t srcSingleRowCount = constInfo_.headDim;

    uint64_t gmOffset = dStartIdx;
    DataCopyAlignGmToUb(apeUbTmp, apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
    inputQueScore.EnQue(apeUbTmp);
    inputQueScore.DeQue<T>();
    DataCopy(apeUb, apeUbTmp, coff_ * dDealSize * constInfo_.cmpRatio);
    inputQueScore.FreeTensor(apeUbTmp);
}

template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                               const Vec1SliceInfo &sliceInfo, uint32_t dDealSize)
{
    uint32_t singleRowElemNum = dDealSize * coff_;
    uint64_t scoreOffset = sliceInfo.dealedSeqCnt * singleRowElemNum;

    uint32_t tcDealSize = sliceInfo.dealTcSize;
    if (sliceInfo.headHolderSeqCnt > 0) {
        uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleRowElemNum;
        uint32_t rCnt = tcDealSize == 1 ? sliceInfo.validSeqCnt * singleRowElemNum :
                                          (constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt) * singleRowElemNum;
        Add(scoreLocal[scoreOffset], scoreLocal[scoreOffset], apeUb[apeOffset], rCnt);
        scoreOffset += rCnt;
        tcDealSize -= 1;
    }
    if (tcDealSize == 0) {
        return;
    }
    if (sliceInfo.tailHolderSeqCnt > 0) {
        tcDealSize -= 1;
        uint64_t apeOffset = 0;
        uint32_t rCnt = (constInfo_.cmpRatio - sliceInfo.tailHolderSeqCnt) * singleRowElemNum;
        uint32_t tailScoreOffset = scoreOffset + tcDealSize * constInfo_.cmpRatio * singleRowElemNum;
        Add(scoreLocal[tailScoreOffset], scoreLocal[tailScoreOffset], apeUb[apeOffset], rCnt);
    }
    if (tcDealSize == 0) {
        return;
    }
    uint32_t rCnt = constInfo_.cmpRatio * singleRowElemNum;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        Add(scoreLocal[scoreOffset + r * rCnt], scoreLocal[scoreOffset + r * rCnt], apeUb, rCnt);
    }
}

template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::AddSingleApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                                     const Vec1SliceInfo &sliceInfo, uint32_t dDealSize)
{
    uint32_t SingleRowElemNum = dDealSize * coff_;
    uint32_t dealRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t scoreOffset = (constInfo_.cmpRatio - dealRowCount) * SingleRowElemNum;
    uint64_t apeOffset = (constInfo_.cmpRatio - dealRowCount) * SingleRowElemNum;
    for (uint32_t dOffset = 0; dOffset < dDealSize; dOffset += FP32_REPEAT_ELEMENT_NUM) {
        uint32_t curAddColCount = min(dDealSize - dOffset, FP32_REPEAT_ELEMENT_NUM);
        Add(scoreLocal[scoreOffset + dOffset], scoreLocal[scoreOffset + dOffset], apeUb[apeOffset + dOffset],
            curAddColCount, dealRowCount,
            {1, 1, 1, static_cast<uint8_t>(SingleRowElemNum / FP32_BLOCK_ELEMENT_NUM),
             static_cast<uint8_t>(SingleRowElemNum / FP32_BLOCK_ELEMENT_NUM),
             static_cast<uint8_t>(SingleRowElemNum / FP32_BLOCK_ELEMENT_NUM)});
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::DataCopyAlignUbToUb(const LocalTensor<O> dstLocal, const LocalTensor<O> srcLocal,
                                                     uint32_t copyRowCount, uint32_t copyColCount,
                                                     uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    // blockLen/srcGap/dstGap 单位为 32B 块，需按元素类型换算（fp32: 8 元素/块，bf16/fp16: 16 元素/块）
    constexpr uint32_t blockElemNum = BYTE_BLOCK / sizeof(O);
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / blockElemNum;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / blockElemNum;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / blockElemNum;
    DataCopy(dstLocal, srcLocal, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::DataCopyAlignGmToUb(const LocalTensor<O> dstLocal, const GlobalTensor<O> srcGm,
                                                     uint32_t copyRowCount, uint32_t copyColCount,
                                                     uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    // blockLen/srcGap/dstGap 单位为 32B 块，需按元素类型换算（fp32: 8 元素/块，bf16/fp16: 16 元素/块）
    constexpr uint32_t blockElemNum = BYTE_BLOCK / sizeof(O);
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / blockElemNum;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / blockElemNum;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / blockElemNum;
    DataCopy(dstLocal, srcGm, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::DataCopyAlignUbToGm(const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal,
                                                     uint32_t copyRowCount, uint32_t copyColCount,
                                                     uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    // blockLen/srcGap/dstGap 单位为 32B 块，需按元素类型换算（fp32: 8 元素/块，bf16/fp16: 16 元素/块）
    constexpr uint32_t blockElemNum = BYTE_BLOCK / sizeof(O);
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / blockElemNum;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / blockElemNum;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / blockElemNum;
    DataCopy(dstGm, srcLocal, intriParams);
}


template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::PadAlign(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                          const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    // Ub data layout after overlap when r = 4 and coff = 2:
    //  Tc0_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq04: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq04: |--- --D_L--- -|------D_R-----|
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = sliceInfo.compressTcSize * constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = srcSingleRowElemNum; // left和right在seq方向是交错存储的
    uint64_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum;

    uint64_t dstUbOffset = sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左侧
        uint64_t preSrcLocalOffset = srcLocalOffset;
        uint64_t preDstUbOffset = dstUbOffset + (sliceInfo.headHolderSeqCnt + constInfo_.cmpRatio) * dstSingleRowCount;
        DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
                            copyRowCount - min(copyRowCount, constInfo_.cmpRatio), copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
        dstUbOffset += dDealSize;
        srcLocalOffset += dDealSize;
    }
    // 右侧
    dstUbOffset += sliceInfo.headHolderSeqCnt * dstSingleRowCount;
    DataCopyAlignUbToUb(dstLocal[dstUbOffset], srcLocal[srcLocalOffset], copyRowCount, copyColCount, srcSingleRowCount,
                        dstSingleRowCount);
}


template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::OverLap(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                         const GlobalTensor<X_T> &srcGm, const GlobalTensor<T> &stateGm,
                                         const GlobalTensor<int32_t> &blockTableGm,
                                         const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                         uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    if constexpr (IS_SCORE) {
        AddApeToScore(srcLocal, apeUb, sliceInfo, dDealSize);
        PipeBarrier<PIPE_V>();
    }
    // srcLocal 的最终生产者是 V pipe（Cast/AddApe），SaveState 的中转 copy（UB->UB/UB->GM）与后续
    // ReadState/PadAlign（MTE2 写 dstLocal）都会读/写相关 UB。实测去掉此屏障最后一个 slice 的窗口
    // 确定性损坏（state 正常但输出错），必须在此排空 V 后再进入窗口装配（保留原算子的 V_MTE2 对亦不足）
    AscendC::PipeBarrier<PIPE_ALL>();
    SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, static_cast<uint32_t>(IS_SCORE));
    // SaveState 直通 UbToGm：MTE3 读 srcLocal 本段区，与后续 PadAlign（V 写窗口区，score 分支与 srcLocal
    // 同 buffer 重叠）及 LoadFromWorkSpace/下个基本块 stage copy（MTE2 写）存在竞争，此处排空 MTE3。
    // 注意：两个 WaitFlag 必须与 SetFlag 在同一位置成对出现（延后 Wait 曾因 flag 配对错乱死锁）
    event_t eventIdMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3V);
    event_t eventIdMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3Mte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3Mte2);

    event_t eventId_V_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId_V_MTE2);
    WaitFlag<HardEvent::V_MTE2>(eventId_V_MTE2);
    ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, static_cast<uint32_t>(IS_SCORE));

    if (sliceInfo.compressTcSize > 0) {
        PadAlign(dstLocal, srcLocal, sliceInfo, dStartIdx, dDealSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            event_t eventId_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            LoadFromWorkSpace(dstLocal, srcGm, srcLocal, sliceInfo, dStartIdx, dDealSize);
        }
    }
    event_t eventId_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
}

// CopyInMm：GM->UB 发射即返回（MTE2），数据就绪由 queue EnQue/DeQue 事件保证
template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CopyInMm(const LocalTensor<T> &dstLocal,
                                                   const GlobalTensor<X_T> &srcGm,
                                                   const Vec1SliceInfo &sliceInfo, const StatisticInfo &statisticInfo,
                                                   uint32_t dStartIdx, uint32_t dDealSize)
{
    // 用户 mm GM 的展平行序与原 cube workspace 一致（按 (batch, token) 排列，行内 [coff, headDim]），
    // 且 slice 起点 (bIdx, sIdx) 的展平序号 = GetTIdxByBatch(bIdx) + sIdx，因此直接按 slice 起点寻址即可。
    uint32_t copyRowCount = statisticInfo.dealSeqCnt * coff_;
    uint32_t copyColCount = dDealSize;
    uint32_t totalCnt = copyRowCount * copyColCount;
    uint64_t srcGmOffset =
        ((uint64_t)tools_.GetTIdxByBatch(sliceInfo.bIdx) + sliceInfo.sIdx) * coff_ * constInfo_.headDim + dStartIdx;
    // X_T 源数据暂存到 fp32 目标 buffer 的后半段（字节偏移 2N 起）：Cast 时写字节 4i 始终小于读字节 2N+2i（i < N），无重叠
    LocalTensor<X_T> stageUb = dstLocal[totalCnt / 2].template ReinterpretCast<X_T>();
    DataCopyAlignGmToUb(stageUb, srcGm[srcGmOffset], copyRowCount, copyColCount, constInfo_.headDim, copyColCount);
}

// CastMm：GM->UB 数据已由 queue DeQue 保证就绪，纯 V 侧 Cast。末尾 PipeBarrier<PIPE_V> 排空 Cast；
// buffer 复用（下块 stage copy 的 MTE2 写）由 queue Free 事件（V->MTE2）接管
template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CastMm(const LocalTensor<T> &dstLocal,
                                                   const StatisticInfo &statisticInfo, uint32_t dDealSize)
{
    uint32_t totalCnt = statisticInfo.dealSeqCnt * coff_ * dDealSize;
    LocalTensor<X_T> stageUb = dstLocal[totalCnt / 2].template ReinterpretCast<X_T>();
    Cast(dstLocal, stageUb, RoundMode::CAST_NONE, totalCnt);
    PipeBarrier<PIPE_V>();
}


template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::LoadFromWorkSpace(const LocalTensor<T> dstLocal,
                                                   const GlobalTensor<X_T> &srcGm, const LocalTensor<T> srcLocal,
                                                   const Vec1SliceInfo &sliceInfo,
                                                   uint32_t dStartIdx, uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t dstLocalOffset =
        (sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio + constInfo_.cmpRatio - copyRowCount) * dstSingleRowElemNum;
    if (sliceInfo.isFirst) { // 从用户 mm GM 中获取（本 call 前驱行，跨基本块/跨 tc 块同样适用）
        uint64_t srcRowBase = (uint64_t)tools_.GetTIdxByBatch(sliceInfo.bIdx) + sliceInfo.sIdx - copyRowCount;
        uint64_t srcGmOffset = srcRowBase * coff_ * constInfo_.headDim + dStartIdx;
        // 目标行本身即 fp32 区域（每行 4*dDealSize 字节）：X_T 数据直接拷到每行后半段（2*dDealSize 字节起），
        // 再逐行原地前向 Cast：写字节 4i < 读字节 2*dDealSize+2i（i < dDealSize），互不覆盖，
        // 无需中转 buffer，对任意 dDealSize（16~512）安全
        constexpr uint32_t X_T_PER_T = sizeof(T) / sizeof(X_T); // 一个 fp32 位宽容纳的 X_T 个数
        LocalTensor<X_T> dstX = dstLocal.template ReinterpretCast<X_T>();
        // 注意 GM 行 stride 是 coff_*headDim（每 token 一行，只取 coff0 半边作为窗口 D_L），
        // 与 FromWokrSpaceToUb 的 coff 交错读（stride=headDim）不同
        DataCopyAlignGmToUb(dstX[X_T_PER_T * dstLocalOffset + dDealSize], srcGm[srcGmOffset], copyRowCount,
                            dDealSize, coff_ * constInfo_.headDim, X_T_PER_T * coff_ * dDealSize);
        // 裸 GM copy(MTE2) 与 Cast(V) 混用，显式同步
        event_t eventIdMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2V);
        for (uint32_t r = 0; r < copyRowCount; r++) {
            uint32_t rowOffset = dstLocalOffset + r * coff_ * dDealSize;
            Cast(dstLocal[rowOffset], dstX[X_T_PER_T * rowOffset + dDealSize], RoundMode::CAST_NONE, dDealSize);
        }
    } else { // 从UB中获取
        uint32_t srcSingleRowElemNum = dDealSize * coff_;
        uint64_t srcLocalOffset = (sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum;
        DataCopyAlignUbToUb(dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], copyRowCount, dDealSize,
                            coff_ * dDealSize, coff_ * dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::ReadFromCacheState(
    const LocalTensor<T> &output, const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
    uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCount > seqCnt) {
            copyRowCount = seqCnt - copyFinishRowCnt;
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                stateIdx * coff_ * constInfo_.headDim + dStartIdx;

        DataCopyAlignGmToUb(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount,
                                dDealSize, coff_ * constInfo_.headDim * 2, coff_ * dDealSize);
        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::WriteToCacheState(
    const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm, const LocalTensor<T> &input,
    uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCount > seqCnt) {
            copyRowCount = seqCnt - copyFinishRowCnt;
        }
        if (idInBlockTable != 0) { // 32
            uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                    remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                    stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            // 直通 UbToGm（srcGap 支持行 stride）：源行 stride = coff*dDealSize，跳过中转 UbToUb 与 queue 同步。
            // V->MTE3 由调用侧（OverLap）SaveState 前的 PipeBarrier<PIPE_ALL> 保证；
            // MTE3->后续 V/MTE2 由 OverLap 里 SaveState 后的 MTE3_V/MTE3_MTE2 flag 保证。
            DataCopyAlignUbToGm(state[stateOffset], input[copyFinishRowCnt * coff_ * dDealSize], copyRowCount,
                                dDealSize, coff_ * dDealSize, coff_ * constInfo_.headDim * 2);
        }

        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                           const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                           uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx)
{
    uint32_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dDealSize;

    if constexpr (COMP::coff == COFF::OVERLAP) {
        WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                          dStartIdx, dDealSize, stateIdx);
        srcBaseOffset += dDealSize;
        dStartIdx += constInfo_.headDim;
    }

    WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx,
                      dDealSize, stateIdx);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
                                                     uint32_t duplicateColCount, uint32_t singleRowCount)
{
    for (uint32_t offset = 0; offset < duplicateColCount; offset += FP32_REPEAT_ELEMENT_NUM) {
        uint32_t curDuplicateColCount = min(duplicateColCount - offset, FP32_REPEAT_ELEMENT_NUM);
        if constexpr (IS_SCORE) {
            Duplicate(dstLocal[offset], SOFTMAX_MIN_NUM, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        } else {
            Duplicate(dstLocal[offset], FLOAT_ZERO, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        }
    }
}


template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::ReadState(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &stateGm,
                                           const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                           uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx)
{
    // 没有需要压缩的块时, 不需要读state的信息
    if (sliceInfo.compressTcSize == 0) {
        return;
    }
    // 填充右边
    if (sliceInfo.headHolderSeqCnt > 0) {
        // 整个batch的第一块
        uint32_t startSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, constInfo_.cmpRatio);
        uint32_t endSeqIdx = sliceInfo.bStartPos;
        uint64_t dstBaseOffset = sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
        if constexpr (COMP::coff == CompressorEpilogue::COFF::OVERLAP) {
            dstBaseOffset += (coff_ - 1) * dDealSize;
        }
        ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                           dStartIdx + (coff_ - 1) * constInfo_.headDim, dDealSize, stateIdx);
    }

    // 填充左边
    if constexpr (COMP::coff == CompressorEpilogue::COFF::OVERLAP) {
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < constInfo_.cmpRatio;
        if (isFirst) {
            // 无历史数据
            // dDealSize必须为64
            uint64_t dstBaseOffset = sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
            DuplicateFirstBlock<IS_SCORE>(dstLocal[dstBaseOffset], constInfo_.cmpRatio, dDealSize, coff_ * dDealSize);
        }
        if (sliceInfo.sIdx < constInfo_.cmpRatio && (!isFirst || sliceInfo.compressTcSize > 1)) {
            uint32_t startSeqIdx =
                sliceInfo.bStartPos < constInfo_.cmpRatio ?
                    0 :
                    Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, constInfo_.cmpRatio) - constInfo_.cmpRatio;
            uint32_t endSeqIdx =
                min(Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, constInfo_.cmpRatio) -
                        constInfo_.cmpRatio,
                    sliceInfo.bStartPos);
            uint64_t dstBaseOffset = sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
            if (isFirst) {
                dstBaseOffset += constInfo_.cmpRatio * coff_ * dDealSize;
            }
            ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                               dStartIdx, dDealSize, stateIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal,
                                                                  const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                                                  uint32_t dDealSize)
{
    float minValue = -2e38;
    uint32_t ReduceSize = coff_ * constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSoftMax(scoreLocal[r * rCnt], scoreLocal[r * rCnt], tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                                  const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpUb,
                                                  uint32_t tcDealSize, uint32_t dDealSize)
{
    uint32_t ReduceSize = coff_ * constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    Mul(kvLocal, kvLocal, scoreLocal, tcDealSize * rCnt);
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSum(dstLocal[r * dDealSize], kvLocal[r * rCnt], tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const Vec1RunInfo &info,
    const StatisticInfo &statisticInfo,
    const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
    uint32_t needDealTcSize)
{
    CompressorEpilogueVec1SliceIterator overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<X_T> scoreMmGm = mmScoreGm_;
    LocalTensor<T> scoreUb = inputQueScore.AllocTensor<T>();
    CopyInMm(scoreUb, scoreMmGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQueScore.EnQue(scoreUb);
    // kv 的 GM copy 与 score 的 Cast/OverLap 重叠（输入双缓冲）：两个独立 queue，
    // EnQue(score) 的 MTE2->V 事件在 score copy 之后、kv copy 之前，DeQue(score) 不等 kv copy
    GlobalTensor<X_T> kvMmGm = mmKvGm_;
    LocalTensor<T> kvUb = inputQueKv.AllocTensor<T>();
    CopyInMm(kvUb, kvMmGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQueKv.EnQue(kvUb);
    inputQueScore.DeQue<T>();
    CastMm(scoreUb, statisticInfo, dDealSize);
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreMmGm, stateCacheGm_, stateBlockTableGm_,
                      info, overLapSliceInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    inputQueScore.FreeTensor(scoreUb);

    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 原算子此处门控 (!isCoreRowFirst || !isCoreLoopFirst)：该情形下左半行来自 cacheTc（已含 ape）。
        // 本算子左半行一律读裸 mm GM（不含 ape），只要 LoadFromWorkSpace 可能填了左半就必须补 ape
        if (originSliceInfo.sIdx != 0 && originSliceInfo.compressTcSize > 0) {
            AddSingleApeToScore(scoreLocal, apeUb, originSliceInfo, dDealSize);
        }
    }

    inputQueKv.DeQue<T>();
    CastMm(kvUb, statisticInfo, dDealSize);
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvMmGm, stateCacheGm_, stateBlockTableGm_, info, overLapSliceInfo,
                       dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    inputQueKv.FreeTensor(kvUb);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::DealVec1BaseBlock(
    const Vec1RunInfo &info, CompressorEpilogueVec1SliceIterator<COMP> &sliceIterator,
    uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = tmpBuff1.Get<T>();
    LocalTensor<T> kvLocal = tmpBuff2.Get<T>();

    OverLapScoreKv(scoreLocal, kvLocal, info, statisticInfo, originSliceInfo, dStartIdx,
                   dDealSize, dBaseSize, needDealTcSize);

    if (statisticInfo.compressor_epilogueScCnt > 0) {
        LocalTensor<T> tmpUb = kvLocal[BUFFER_SIZE_BYTE_32K / sizeof(T)];
        SoftmaxDN(scoreLocal, tmpUb, statisticInfo.compressor_epilogueScCnt, dDealSize);
        // 压缩行直接落到 tmpBuff1 区域：scoreLocal 窗口在 KvMulReduceScore 的 Mul 之后即废弃
        LocalTensor<T> compressedUb = tmpBuff1.Get<T>();
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, compressedUb, tmpUb, statisticInfo.compressor_epilogueScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        FinishCompressedRows(compressedUb, statisticInfo.compressor_epilogueScCnt, tmpUb);
    }
    compressedCnt_ += statisticInfo.compressor_epilogueScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo)
{
    // 每核独占完整 headDim 维（行并行）：窗口装配、softmax、加权和、rms_norm、rope 全部在核内完成，
    // 无跨核数据依赖，因此无需 vec1Res workspace 中转与 SyncAll 全局同步
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;
    splitInfo.dBaseSize = constInfo_.headDim;
    splitInfo.vec1GroupSize = 1;
    splitInfo.vec1GroupNum = min(aiCoreNum, info.dealTcNum);
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CalcTaskDistribution(const Vec1RunInfo &info,
                                                                             Vec1SplitInfo &splitInfo)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t groupSize = splitInfo.vec1GroupSize;
    uint32_t groupNum = splitInfo.vec1GroupNum;
    uint32_t dealTcNum = info.dealTcNum;

    if (blockIdx < groupSize * (dealTcNum % groupNum)) {
        splitInfo.dealTcSize = dealTcNum / groupNum + 1;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize);
    } else if (blockIdx < groupSize * groupNum) {
        splitInfo.dealTcSize = dealTcNum / groupNum;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize) + dealTcNum % groupNum;
    } else {
        splitInfo.dealTcSize = 0;
        splitInfo.preDealTcSize = dealTcNum;
    }
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::UpdateIteratorState(const Vec1RunInfo &info,
                                                                            Vec1SplitInfo &splitInfo)
{
    CompressorEpilogueVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.Reset(info.bStart, info.sStart, 0U, 0U);
    Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();

    // 处理前序任务量，更新起始索引
    if (splitInfo.preDealTcSize > 0) {
        sliceIterator.SetNeedDealTcSize(splitInfo.preDealTcSize);
        StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
        splitInfo.curCompressedCnt = statisticInfo.compressor_epilogueScCnt;
        splitInfo.dealSeqStartIdx = sliceInfo.dealedSeqCnt;
        splitInfo.curBStart = sliceInfo.bIdx;
        splitInfo.curSStart = sliceInfo.sIdx;
    } else {
        splitInfo.curCompressedCnt = 0;
        splitInfo.dealSeqStartIdx = 0;
        splitInfo.curBStart = info.bStart;
        splitInfo.curSStart = info.sStart;
    }

    // 处理当前核实际要跑的任务量
    sliceIterator.SetNeedDealTcSize(info.dealTcNum - splitInfo.preDealTcSize);
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    splitInfo.totalCompressedCnt = splitInfo.curCompressedCnt + statisticInfo.compressor_epilogueScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
{
    // 计算headDim和Tc方向切分大小
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_32K / (constInfo_.cmpRatio * coff_ * sizeof(T));

    // 切块逻辑
    if (maxDealColNum < splitInfo.dBaseSize) {
        splitInfo.tcSplitSize = 1;
        splitInfo.dLoopCount = CeilDivT(splitInfo.dBaseSize, maxDealColNum);
        splitInfo.dSplitSize = splitInfo.dBaseSize / splitInfo.dLoopCount;
    } else {
        splitInfo.dSplitSize = splitInfo.dBaseSize;
        splitInfo.dLoopCount = splitInfo.dBaseSize / splitInfo.dSplitSize; // 此处常等于1，保留原逻辑
        splitInfo.tcSplitSize = maxDealColNum / splitInfo.dBaseSize;
    }
}

template <typename COMP>
__aicore__ inline Vec1SplitInfo CompressorEpilogueBlockVectorPerf<COMP>::SplitCoreV1(const Vec1RunInfo &info)
{
    Vec1SplitInfo splitInfo;

    // 1. 计算基础分组和分片大小
    CalcGroupInfo(info, splitInfo);

    // 2. 根据当前的 BlockIdx 计算任务分配（负载均衡）
    CalcTaskDistribution(info, splitInfo);

    // 3. 刷新迭代器并获取当前核的起始位置状态
    UpdateIteratorState(info, splitInfo);

    if (splitInfo.dealTcSize == 0) {
        return splitInfo;
    }

    // 4. 计算具体在内存中的切块（Tiling）逻辑
    CalcTilingStrategy(splitInfo);

    return splitInfo;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::ComputeVec1(const Vec1RunInfo &info)
{
    if (info.dealTcNum == 0) {
        return;
    }
    uint32_t preCompressedCnt = compressedCnt_;
    Vec1SplitInfo splitInfo = SplitCoreV1(info);
    // 计算当前VecCore的任务量
    if (splitInfo.dealTcSize == 0) {
        compressedCnt_ += splitInfo.totalCompressedCnt;
        return;
    }
    // 输出位置游标：本核起点 slice 对应的 (batch, batch 内压缩行起点)，仅 BSH 输出映射使用
    if (!v2OutInited) {
        uint32_t startPos = tools_.GetStartPos(splitInfo.curBStart);
        OutputBStartIdx = splitInfo.curBStart;
        OutputSStartIdx = (startPos + splitInfo.curSStart) / constInfo_.cmpRatio - startPos / constInfo_.cmpRatio;
        v2OutInited = true;
    }

    CompressorEpilogueVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    // 切块循环（D=512 全维，dLoopCount 恒为 1，dBaseOffset 恒为 0）
    uint64_t baseOffset = 0;
    for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
        uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;

        CopyInApe(apeUb, dBaseOffset, splitInfo.dSplitSize);

        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
        compressedCnt_ = preCompressedCnt + splitInfo.curCompressedCnt;
        for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcSize; tcIdx += splitInfo.tcSplitSize) {
            uint32_t actDealTcSize = min(splitInfo.tcSplitSize, splitInfo.dealTcSize - tcIdx);

            // 处理单个切块
            sliceIterator.SetNeedDealTcSize(actDealTcSize);
            sliceIterator.SetDealedTcCnt(0U);
            DealVec1BaseBlock(info, sliceIterator, dBaseOffset, splitInfo.dSplitSize, splitInfo.dBaseSize);
        }
    }
    compressedCnt_ = preCompressedCnt + splitInfo.totalCompressedCnt;
}


template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::SingleCalRope(const LocalTensor<X_T> &outputUb,
                                                                      const LocalTensor<T> &normResUb, uint32_t rowCnt,
                                                                      uint32_t curDealScSize, uint32_t globalScStart)
{
    uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
    uint64_t SinCosOffset = globalScStart * constInfo_.ropeHeadDim;
    // sin/cos each reserves 16KB so fp32 rope can use the same compute tile.
    LocalTensor<ROPE_T> cosUb = inputQueScore.AllocTensor<ROPE_T>();
    LocalTensor<ROPE_T> sinUb = cosUb[BUFFER_SIZE_BYTE_16K / sizeof(ROPE_T)];
    DataCopy(cosUb, ropeCosGm_[SinCosOffset], computeSize);
    DataCopy(sinUb, ropeSinGm_[SinCosOffset], computeSize);
    inputQueScore.EnQue(sinUb);
    inputQueScore.DeQue<ROPE_T>();

    LocalTensor<T> ropeCosFp32Local = tmpBuff2.Get<T>();
    LocalTensor<T> ropeSinFp32Local = ropeCosFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
    LocalTensor<T> tempLocal = ropeSinFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<ROPE_T, T>::value) {
        DataCopy(ropeCosFp32Local, cosUb, computeSize);
        DataCopy(ropeSinFp32Local, sinUb, computeSize);
    } else {
        Cast(ropeCosFp32Local, cosUb, RoundMode::CAST_NONE, computeSize);
        Cast(ropeSinFp32Local, sinUb, RoundMode::CAST_NONE, computeSize);
    }
    PipeBarrier<PIPE_V>();
    inputQueScore.FreeTensor(sinUb);
    RotaryPosEmb<COMP::rotaryMode>(normResUb[rowCnt * constInfo_.headDim], normResUb[rowCnt * constInfo_.headDim],
                                   ropeCosFp32Local, ropeSinFp32Local, tempLocal, gatherOffsetCastUb, curDealScSize,
                                   constInfo_.ropeHeadDim, constInfo_.headDim,
                                   constInfo_.headDim - constInfo_.ropeHeadDim);
    PipeBarrier<PIPE_V>();
}


template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CalcGlobalScStart(uint32_t bStart, uint32_t scStart,
                                                                          uint32_t bEnd, uint32_t scEnd,
                                                                          uint64_t &globalScStart)
{
    for (uint32_t bIdx = bStart; bIdx < bEnd; ++bIdx) {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            globalScStart += GetScSize();
        } else {
            curActSeqLength_ = constInfo_.sSize;
            globalScStart += CeilDivT(curActSeqLength_, constInfo_.cmpRatio);
        }
    }
    globalScStart -= scStart;
    globalScStart += scEnd;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart,
                                                                        uint32_t &dealScSize, uint32_t &curDealScSize)
{
    curActSeqLength_ = GetSeqLength(outputBStart);
    curStartPos_ = GetStartPos(outputBStart);
    uint32_t curBatchScSize =
        (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    uint32_t curBatchRemainScSize = curBatchScSize - outputSStart;
    curDealScSize = curBatchRemainScSize > dealScSize ? dealScSize : curBatchRemainScSize;
    dealScSize -= curDealScSize;
    outputSStart += curDealScSize;
    if (outputSStart == curBatchScSize) {
        outputBStart++;
        outputSStart = 0;
    }
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::FinishCompressedRows(const LocalTensor<T> &compressedUb,
                                                                             uint32_t scCnt, const LocalTensor<T> &tmpUb)
{
    // 完全串行化：本核独占完整 headDim，压缩行即完整行，核内直接完成 rms_norm + rope + cast + 写 cmp_kv，
    // 无 vec1Res workspace 中转、无 SyncAll 全局同步
    RmsNormParam rmsNormParams;
    rmsNormParams.reciprocal = constInfo_.reciprocalD;
    rmsNormParams.epsilon = constInfo_.normEps;
    rmsNormParams.row = scCnt;
    rmsNormParams.col = constInfo_.headDim;
    RmsNorm(compressedUb, compressedUb, normWeightUb, tmpUb, rmsNormParams);
    PipeBarrier<PIPE_V>();
    // 输出 X_T 中转借用 tmpBuff2 前 16K（rope 临时区在 SingleCalRope 后已用完），省掉独立 output queue
    LocalTensor<X_T> outputUb = tmpBuff2.Get<X_T>();
    // rope：sin/cos 按全局压缩行号取（与输出布局无关）
    SingleCalRope(outputUb, compressedUb, 0, scCnt, compressedCnt_);
    Cast(outputUb, compressedUb, RoundMode::CAST_RINT, scCnt * constInfo_.headDim);
    PipeBarrier<PIPE_V>();
    // 裸 buffer：V->MTE3 / MTE3->V 显式同步（替代原 outputQue1 的 EnQue/DeQue/Free）
    event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
    CopyFinalResultOut(outputUb, scCnt);
    event_t eventIdMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3V);
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CopyFinalResultOut(const LocalTensor<X_T> &cmpKvOutUb,
                                                                           uint32_t dealRowCount)
{
    uint32_t dealScSize = dealRowCount;
    uint32_t curDealScSize = 0;
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        // TH：cmp_kv 按全局压缩行号紧凑排列（compressedCnt_ 起点即本核产出行的全局行号）
        DataCopy(cmpKvOutGm_[compressedCnt_ * constInfo_.headDim], cmpKvOutUb, dealRowCount * constInfo_.headDim);
        while (dealScSize > 0) {
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
        }
    } else {
        // BSH：逐 batch 写出（游标在 ComputeVec1 按核起点初始化，UpdateOutputIdx 推进）
        uint64_t globalScStart = 0;
        CalcGlobalScStart(0, 0, OutputBStartIdx, OutputSStartIdx, globalScStart);
        uint32_t ubProcessedCount = 0;
        uint32_t preOutputBStartIdx = 0;
        uint32_t preOutputSStartIdx = 0;
        while (dealScSize > 0) {
            // 逐batch计算写出索引
            preOutputBStartIdx = OutputBStartIdx;
            preOutputSStartIdx = OutputSStartIdx;
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
            DataCopy(cmpKvOutGm_[globalScStart * constInfo_.headDim], cmpKvOutUb[ubProcessedCount * constInfo_.headDim],
                     curDealScSize * constInfo_.headDim);
            CalcGlobalScStart(preOutputBStartIdx, preOutputSStartIdx, OutputBStartIdx, OutputSStartIdx, globalScStart);
            ubProcessedCount += curDealScSize;
        }
    }
}
} // namespace CompressorEpilogue
#endif // COMPRESSOR_EPILOGUE_BLOCK_VECTOR_PREF_H
