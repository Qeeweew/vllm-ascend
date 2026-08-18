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
    // copy / 计算严格分离的 helper（同步 flag 一律在主流程）：
    template <bool IS_SCORE>
    __aicore__ inline void CopyInState(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &stateGm,
                                       const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                       uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    template <bool IS_SCORE>
    __aicore__ inline void FillFirstBlock(const LocalTensor<T> &dstLocal, const Vec1SliceInfo &sliceInfo,
                                          uint32_t dDealSize);
    __aicore__ inline void CopyInHistoryGm(const LocalTensor<T> dstLocal, const GlobalTensor<X_T> &srcGm,
                                           const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    template <bool IS_SCORE>
    __aicore__ inline void CastHistoryGm(const LocalTensor<T> dstLocal, const Vec1SliceInfo &sliceInfo,
                                         uint32_t dDealSize);
    template <bool IS_SCORE>
    __aicore__ inline void CopyHistoryUb(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                         const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    __aicore__ inline void CopyInRopeCosSin(uint32_t globalScStart, uint32_t curDealScSize);
    __aicore__ inline void CalRope(const LocalTensor<T> &normResUb, uint32_t rowCnt, uint32_t curDealScSize);
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
    __aicore__ inline void CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1(const Vec1RunInfo &info);
    __aicore__ inline void CopyFinalResultOut(const LocalTensor<X_T> &cmpKvOutUb, uint32_t dealRowCount);
    __aicore__ inline void SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    template <bool IS_SCORE>
    __aicore__ inline void DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
                                               uint32_t duplicateColCount, uint32_t singleRowCount);
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
    TBuf<TPosition::VECCALC> ropeBuf; // rope cos/sin 中转（copy in 阶段就绪，计算阶段使用）
    // in buffer：score/kv 独立 buffer 实现输入双缓冲（kv GM copy 与 score Cast/OverLap 重叠）。
    // 同步用常驻 event id（InitBuffers 一次性 Fetch）：MTE2_V = GM copy 就绪；V_MTE2 = buffer 复用保护
    TBuf<TPosition::VECIN> inBufScore;
    TBuf<TPosition::VECIN> inBufKv;
    // 常驻同步事件 id：InitBuffers 一次性 Fetch，热路径只允许 SetFlag/WaitFlag。
    // 每类 pipe 方向一个 id，Set 在 producer 后 / Wait 在 consumer 前严格交替（allocate 时预 Set，结束 Wait 释放）
    event_t evMte2V_;   // MTE2 copy 完成 -> V 可读（本轮所有 copy in 统一发射、统一等待）
    event_t evVMte2_;   // V 用完 buffer -> 允许 MTE2 覆盖写（轮末 Set，下轮 copy in 前 Wait）
    event_t evMte3V_;   // MTE3（SaveState/输出）读 buffer 完成 -> V 可覆盖写（轮末 Set/Wait 紧贴）
    event_t evVMte3_;   // V（cast/add ape/输出 cast）完成 -> MTE3 可读/写（块内 Set/Wait）
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
    pipe->InitBuffer(inBufScore, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inBufKv, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(normWeightBuf, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(gatherOffsetBuf, BUFFER_SIZE_BYTE_1K);
    // ape 实际用量 coff*cmpRatio*dDealSize fp32 = 16KB（buf 按 UB 预算收缩）
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(ropeBuf, BUFFER_SIZE_BYTE_4K);
    normWeightUb = normWeightBuf.Get<T>();
    apeUb = apeBuf.Get<T>();
    evMte2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    evVMte2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    evMte3V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    evVMte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    LocalTensor<X_T> normweightInUb = inBufScore.Get<X_T>();
    LocalTensor<int32_t> gatherOffsetUb = gatherOffsetBuf.Get<int32_t>();
    DataCopy(normweightInUb, normWeightGm_, constInfo_.headDim); // 获取normWeight，常驻
    SetFlag<HardEvent::MTE2_V>(evMte2V_);
    WaitFlag<HardEvent::MTE2_V>(evMte2V_);
    Cast(normWeightUb, normweightInUb, RoundMode::CAST_NONE, constInfo_.headDim);

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
    uint32_t copyRowCount = coff_ * constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;

    uint64_t gmOffset = dStartIdx;
    // ape 布局 [cmpRatio, coff*headDim]（行内连续、行间 stride = coff*headDim）。
    // 连续拷贝仅当 dDealSize == headDim（c4 全宽）时等价；c128 切 d 后 dDealSize < headDim
    // 必须按行 stride 取列，否则跨行错位。与 fused CopyInApe 的 DataCopyAlignGmToUb 同语义。
    DataCopyAlignGmToUb(apeUb, apeGm_[gmOffset], copyRowCount, copyColCount, constInfo_.headDim, copyColCount);
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


// CopyInHistoryGm：窗口左半（本 call 前驱行）从用户 mm GM 直接拷到窗口行后半（纯 copy，同步在主流程）
template <typename COMP>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::CopyInHistoryGm(const LocalTensor<T> dstLocal, const GlobalTensor<X_T> &srcGm,
                                                         const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                         uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0 || !sliceInfo.isFirst) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t dstLocalOffset =
        (sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio + constInfo_.cmpRatio - copyRowCount) * dstSingleRowElemNum;
    uint64_t srcRowBase = (uint64_t)tools_.GetTIdxByBatch(sliceInfo.bIdx) + sliceInfo.sIdx - copyRowCount;
    uint64_t srcGmOffset = srcRowBase * coff_ * constInfo_.headDim + dStartIdx;
    // X_T 数据直接拷到每行后半段（2*dDealSize 字节起），计算阶段逐行原地前向 Cast：
    // 写字节 4i < 读字节 2*dDealSize+2i（i < dDealSize），互不覆盖，无需中转 buffer
    constexpr uint32_t X_T_PER_T = sizeof(T) / sizeof(X_T); // 一个 fp32 位宽容纳的 X_T 个数
    LocalTensor<X_T> dstX = dstLocal.template ReinterpretCast<X_T>();
    // 注意 GM 行 stride 是 coff_*headDim（每 token 一行，只取 coff0 半边作为窗口 D_L），与 CopyInMm 的交错读不同
    DataCopyAlignGmToUb(dstX[X_T_PER_T * dstLocalOffset + dDealSize], srcGm[srcGmOffset], copyRowCount,
                        dDealSize, coff_ * constInfo_.headDim, X_T_PER_T * coff_ * dDealSize);
}

// CastHistoryGm：CopyInHistoryGm 的就绪数据逐行原地 Cast（纯计算）
template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::CastHistoryGm(const LocalTensor<T> dstLocal, const Vec1SliceInfo &sliceInfo,
                                                       uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0 || !sliceInfo.isFirst) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t dstLocalOffset =
        (sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio + constInfo_.cmpRatio - copyRowCount) * dstSingleRowElemNum;
    constexpr uint32_t X_T_PER_T = sizeof(T) / sizeof(X_T);
    LocalTensor<X_T> dstX = dstLocal.template ReinterpretCast<X_T>();
    for (uint32_t r = 0; r < copyRowCount; r++) {
        uint32_t rowOffset = dstLocalOffset + r * coff_ * dDealSize;
        Cast(dstLocal[rowOffset], dstX[X_T_PER_T * rowOffset + dDealSize], RoundMode::CAST_NONE, dDealSize);
    }
}

// CopyHistoryUb：窗口左半从本段 UB（前驱行已 cast）装配（纯计算）
template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::CopyHistoryUb(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                                       const Vec1SliceInfo &sliceInfo, uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0 || sliceInfo.isFirst) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t dstLocalOffset =
        (sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio + constInfo_.cmpRatio - copyRowCount) * dstSingleRowElemNum;
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint64_t srcLocalOffset = (sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum;
    DataCopyAlignUbToUb(dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], copyRowCount, dDealSize,
                        coff_ * dDealSize, coff_ * dDealSize);
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
            // V->MTE3 由主流程 SaveState 前的 V_MTE3 flag 保证；MTE3->V 由 SaveState 后的 MTE3_V flag 保证。
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


// CopyInState：历史 state GM->UB（纯 copy，同步在主流程）；batch 首块的初值填充在计算阶段 FillFirstBlock
template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::CopyInState(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &stateGm,
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

// FillFirstBlock：batch 首块无历史，V 侧填充初值（score 填 SOFTMAX_MIN，kv 填 0）。纯计算
template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void
CompressorEpilogueBlockVectorPerf<COMP>::FillFirstBlock(const LocalTensor<T> &dstLocal, const Vec1SliceInfo &sliceInfo,
                                                        uint32_t dDealSize)
{
    if constexpr (COMP::coff == CompressorEpilogue::COFF::OVERLAP) {
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < constInfo_.cmpRatio;
        if (isFirst) {
            uint64_t dstBaseOffset = sliceInfo.compressor_epilogueedScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
            DuplicateFirstBlock<IS_SCORE>(dstLocal[dstBaseOffset], constInfo_.cmpRatio, dDealSize, coff_ * dDealSize);
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

// OverLapScoreKv 主流程：copy in 统一发射 -> 统一等待 -> 计算（cast/add ape/save state/窗口装配）
// 同步 flag 全部在此（helper 内零 flag）：轮末 MTE3_V + V_MTE2 紧贴自同步（Wait 挂在 MTE2/V 队列尾部）
template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const Vec1RunInfo &info,
    const StatisticInfo &statisticInfo,
    const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
    uint32_t needDealTcSize)
{
    CompressorEpilogueVec1SliceIterator iter(tools_);
    iter.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo &slice = iter.GetSlice();

    GlobalTensor<X_T> scoreMmGm = mmScoreGm_;
    GlobalTensor<X_T> kvMmGm = mmKvGm_;
    LocalTensor<T> scoreUb = inBufScore.Get<T>();
    LocalTensor<T> kvUb = inBufKv.Get<T>();

    // ==================== copy in：本轮所有 copy 一次发射 ====================
    // buffer 复用保护由上一轮末尾的紧贴 MTE3_V->V_MTE2 传递链承担（下轮 copy 排在 MTE2 队列的 Wait 后）
    CopyInMm(scoreUb, scoreMmGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    CopyInMm(kvUb, kvMmGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    // 历史 state + 历史 mm 左半（逐 slice 发射，helper 内部按 slice 条件自行跳过）
    iter.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    iter.SetNeedDealTcSize(needDealTcSize);
    while (!iter.IsEnd()) {
        iter.GetSlice();
        CopyInState<true>(scoreLocal, stateCacheGm_, stateBlockTableGm_, slice, dStartIdx, dDealSize, 1U);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            CopyInHistoryGm(scoreLocal, scoreMmGm, slice, dStartIdx, dDealSize);
        }
        iter.IteratorSlice();
    }
    iter.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    iter.SetNeedDealTcSize(needDealTcSize);
    while (!iter.IsEnd()) {
        iter.GetSlice();
        CopyInState<false>(kvLocal, stateCacheGm_, stateBlockTableGm_, slice, dStartIdx, dDealSize, 0U);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            CopyInHistoryGm(kvLocal, kvMmGm, slice, dStartIdx, dDealSize);
        }
        iter.IteratorSlice();
    }
    // 本基本块 sc 行的 rope cos/sin
    if (statisticInfo.compressor_epilogueScCnt > 0) {
        CopyInRopeCosSin(compressedCnt_, statisticInfo.compressor_epilogueScCnt);
    }
    // ==================== 统一等待 ====================
    SetFlag<HardEvent::MTE2_V>(evMte2V_);
    WaitFlag<HardEvent::MTE2_V>(evMte2V_);

    // ==================== 计算 ====================
    // --- score：cast -> add ape -> (PIPE_V) -> save state(MTE3) + 窗口装配 ---
    CastMm(scoreUb, statisticInfo, dDealSize);
    iter.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    iter.SetNeedDealTcSize(needDealTcSize);
    while (!iter.IsEnd()) {
        iter.GetSlice();
        if (slice.dealTcSize > 0) {
            AddApeToScore(scoreUb, apeUb, slice, dDealSize);
        }
        iter.IteratorSlice();
    }
    // SaveState（MTE3 读 scoreUb）的源是 V（Cast/AddApe）写的：V->MTE3 pipe 间同步
    SetFlag<HardEvent::V_MTE3>(evVMte3_);
    WaitFlag<HardEvent::V_MTE3>(evVMte3_);
    iter.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    iter.SetNeedDealTcSize(needDealTcSize);
    while (!iter.IsEnd()) {
        iter.GetSlice();
        if (slice.dealTcSize > 0) {
            SaveState(scoreUb, stateCacheGm_, stateBlockTableGm_, slice, dStartIdx, dDealSize, 1U);
        }
        iter.IteratorSlice();
    }
    // SaveState（MTE3 读 scoreUb）排空后再继续 V（保守，先求对再优化）
    SetFlag<HardEvent::MTE3_V>(evMte3V_);
    WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    iter.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    iter.SetNeedDealTcSize(needDealTcSize);
    while (!iter.IsEnd()) {
        iter.GetSlice();
        if (slice.dealTcSize > 0 && slice.compressTcSize > 0) {
            FillFirstBlock<true>(scoreLocal, slice, dDealSize);
            if constexpr (COMP::coff == COFF::OVERLAP) {
                CastHistoryGm<true>(scoreLocal, slice, dDealSize);
                CopyHistoryUb<true>(scoreLocal, scoreUb, slice, dDealSize);
            }
            PadAlign(scoreLocal, scoreUb, slice, dStartIdx, dDealSize);
        }
        iter.IteratorSlice();
    }
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左半行来自裸 mm GM（不含 ape），CopyInHistoryGm 填过左半就必须补 ape
        if (originSliceInfo.sIdx != 0 && originSliceInfo.compressTcSize > 0) {
            AddSingleApeToScore(scoreLocal, apeUb, originSliceInfo, dDealSize);
        }
    }
    // --- kv：cast -> save state + 窗口装配 ---
    CastMm(kvUb, statisticInfo, dDealSize); // 末尾自带 PIPE_V
    SetFlag<HardEvent::V_MTE3>(evVMte3_);
    WaitFlag<HardEvent::V_MTE3>(evVMte3_);
    iter.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    iter.SetNeedDealTcSize(needDealTcSize);
    while (!iter.IsEnd()) {
        iter.GetSlice();
        if (slice.dealTcSize > 0) {
            SaveState(kvUb, stateCacheGm_, stateBlockTableGm_, slice, dStartIdx, dDealSize, 0U);
        }
        iter.IteratorSlice();
    }
    SetFlag<HardEvent::MTE3_V>(evMte3V_);
    WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    iter.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    iter.SetNeedDealTcSize(needDealTcSize);
    while (!iter.IsEnd()) {
        iter.GetSlice();
        if (slice.dealTcSize > 0 && slice.compressTcSize > 0) {
            FillFirstBlock<false>(kvLocal, slice, dDealSize);
            if constexpr (COMP::coff == COFF::OVERLAP) {
                CastHistoryGm<false>(kvLocal, slice, dDealSize);
                CopyHistoryUb<false>(kvLocal, kvUb, slice, dDealSize);
            }
            PadAlign(kvLocal, kvUb, slice, dStartIdx, dDealSize);
        }
        iter.IteratorSlice();
    }
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
        uint32_t scCnt = statisticInfo.compressor_epilogueScCnt;
        LocalTensor<T> tmpUb = kvLocal[BUFFER_SIZE_BYTE_32K / sizeof(T)];
        SoftmaxDN(scoreLocal, tmpUb, scCnt, dDealSize);
        // 压缩行直接落到 tmpBuff1 区域：scoreLocal 窗口在 KvMulReduceScore 的 Mul 之后即废弃
        LocalTensor<T> compressedUb = tmpBuff1.Get<T>();
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, compressedUb, tmpUb, scCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        // 完全串行化：本核独占完整 headDim，压缩行即完整行，核内完成 rms_norm + rope + cast，
        // 无 vec1Res workspace 中转、无 SyncAll 全局同步。rope cos/sin 已在 copy in 阶段就绪
        RmsNormParam rmsNormParams;
        rmsNormParams.reciprocal = constInfo_.reciprocalD;
        rmsNormParams.epsilon = constInfo_.normEps;
        rmsNormParams.row = scCnt;
        rmsNormParams.col = constInfo_.headDim;
        RmsNorm(compressedUb, compressedUb, normWeightUb, tmpUb, rmsNormParams);
        PipeBarrier<PIPE_V>();
        CalRope(compressedUb, 0, scCnt);
        // 输出 X_T 中转借用 tmpBuff2 前 16K（rope 临时区已用完）
        LocalTensor<X_T> outputUb = tmpBuff2.Get<X_T>();
        Cast(outputUb, compressedUb, RoundMode::CAST_RINT, scCnt * constInfo_.headDim);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evVMte3_);
        WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        CopyFinalResultOut(outputUb, scCnt);
    }
    // 收尾同步（紧贴）：先 V 等 MTE3（SaveState/输出读完 buffer），再 MTE2 等 V——
    // 形成 MTE3->V->MTE2 传递链，下轮 copy（MTE2）间接等齐 MTE3 与 V
    SetFlag<HardEvent::MTE3_V>(evMte3V_);
    WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    SetFlag<HardEvent::V_MTE2>(evVMte2_);
    WaitFlag<HardEvent::V_MTE2>(evVMte2_);
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
        // d 切分路径：ape 也是窗口级整片拷入 apeBuf（16K 固定预算，见 InitBuffers），
        // 必须满足 coff*cmpRatio*dDealSize*sizeof(T) <= 16K。c128（cmpRatio=128, coff=1）
        // 若不加约束 dDealSize=64 -> ape 128*64*4B=32K 溢出 apeBuf，实测崩溃：
        //   VEC instruction error: the ub address out of bounds（blk:10, fixp 0x6000022）。
        // 约束后 c128 dSplitSize=32（ape=16KB 恰好）。
        // 注意（已知遗留）：d 分块使压缩结果只有 dSplitSize 列，而本算子 rms_norm 在 vec1 内
        // 对完整 headDim 行做（col=headDim），d 分块下数值错误——c128 精度待后续修复
        // （详见 reports/compressor_epilogue_c128_analysis.md）。c4 走 else 分支不受影响。
        uint32_t apeMaxDealColNum = BUFFER_SIZE_BYTE_16K / (constInfo_.cmpRatio * coff_ * sizeof(T));
        maxDealColNum = maxDealColNum < apeMaxDealColNum ? maxDealColNum : apeMaxDealColNum;
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

        // ape copy（纯 copy）：复用保护由上一轮末尾的紧贴 V_MTE2 承担；就绪同步紧贴
        CopyInApe(apeUb, dBaseOffset, splitInfo.dSplitSize);
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);

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


// CopyInRopeCosSin：本基本块 sc 行的 rope cos/sin 拷入 ropeBuf（纯 copy，同步在主流程）
template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CopyInRopeCosSin(uint32_t globalScStart,
                                                                                 uint32_t curDealScSize)
{
    uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
    uint64_t sinCosOffset = (uint64_t)globalScStart * constInfo_.ropeHeadDim;
    LocalTensor<ROPE_T> cosUb = ropeBuf.Get<ROPE_T>();
    LocalTensor<ROPE_T> sinUb = cosUb[computeSize];
    DataCopy(cosUb, ropeCosGm_[sinCosOffset], computeSize);
    DataCopy(sinUb, ropeSinGm_[sinCosOffset], computeSize);
}

// CalRope：rope 计算（纯 V）：cos/sin fp32 展开 + RotaryPosEmb；数据须已由 copy in 阶段就绪
template <typename COMP>
__aicore__ inline void CompressorEpilogueBlockVectorPerf<COMP>::CalRope(const LocalTensor<T> &normResUb,
                                                                      uint32_t rowCnt, uint32_t curDealScSize)
{
    uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
    LocalTensor<ROPE_T> cosUb = ropeBuf.Get<ROPE_T>();
    LocalTensor<ROPE_T> sinUb = cosUb[computeSize];
    LocalTensor<T> ropeCosFp32Local = tmpBuff2.Get<T>();
    LocalTensor<T> ropeSinFp32Local = ropeCosFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
    LocalTensor<T> tempLocal = ropeSinFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<ROPE_T, T>::value) {
        DataCopy(ropeCosFp32Local, cosUb, computeSize);
        DataCopy(ropeSinFp32Local, sinUb, computeSize);
    } else {
        Cast(ropeCosFp32Local, cosUb, RoundMode::CAST_NONE, computeSize);
        Cast(ropeSinFp32Local, sinUb, computeSize);
    }
    PipeBarrier<PIPE_V>();
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
