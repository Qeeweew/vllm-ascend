/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_GATING_TOP_K_WITHOUT_GROUP_BATCH_H
#define MOE_GATING_TOP_K_WITHOUT_GROUP_BATCH_H
#include "kernel_operator.h"
#include "common.h"

namespace MoeGatingTopK {
using namespace AscendC;

constexpr int32_t FP32_NEG_INF_BITS = 0xFF800000; // fp32 -inf

template <typename T>
class MoeGatingTopKWithoutGroupBatch {
public:
    __aicore__ inline MoeGatingTopKWithoutGroupBatch(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitConstant();
    __aicore__ inline void CopyInX(int64_t bufIdx, int64_t startRow, int64_t rows);
    __aicore__ inline void ComputeX(int64_t bufIdx, int64_t rows);
    __aicore__ inline void TopK(int64_t rows, LocalTensor<int32_t> idxDst, bool waitAcc);
    __aicore__ inline void TailAndCopyOut(int64_t startRow, int64_t rows, int64_t accSlotRows,
                                          int64_t flushRows);

    __aicore__ inline LocalTensor<float> NormOut()
    {
        return addBias_ ? xNormBuf_.Get<float>() : xNormWithBiasBuf_.Get<float>();
    }

private:
    TPipe *pipe_;

    TBuf<TPosition::VECCALC> xBufPing_;
    TBuf<TPosition::VECCALC> xBufPong_;
    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> xNormBuf_;
    TBuf<TPosition::VECCALC> xNormWithBiasBuf_;
    TBuf<TPosition::VECCALC> biasFp32Buf_;      // P
    TBuf<TPosition::VECCALC> biasStageBuf_;     // P
    TBuf<TPosition::VECCALC> biasBatchBuf_;
    TBuf<TPosition::VECCALC> expertIdBuf_;
    TBuf<TPosition::VECCALC> sortedBuf_;
    TBuf<TPosition::VECCALC> mergeBuf_;
    TBuf<TPosition::VECCALC> finalBuf_;
    TBuf<TPosition::VECCALC> yAccBuf_;
    TBuf<TPosition::VECCALC> idxAccBuf_;
    TBuf<TPosition::VECCALC> byteIdxBuf_;
    TBuf<TPosition::VECCALC> rowBaseBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<float> outGm_;

    int64_t blockIdx_ = 0;
    int64_t perCoreRowCount_ = 0;
    int64_t curCoreRowCount_ = 0;
    int64_t expertCount_ = 0;
    int64_t paddedCount_ = 0;
    int64_t rowBatch_ = 0;
    int64_t accWinRows_ = 0;
    TEventID eventAccReuse_;
    TEventID evMte2V_;
    static constexpr int64_t ACC_COPYOUT_BATCHES = 4;
    int64_t runs_ = 0;          // P / 32
    bool addBias_ = false;
    bool outFlag_ = false;
    int64_t k_ = 0;
    int64_t renorm_ = 0;
    int64_t normType_ = 0;
    float routedScalingFactor_ = 1.0f;
    float eps_ = 1e-20f;

    const MoeGatingTopKTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupBatch<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                               GM_ADDR out, GM_ADDR workspace,
                                                               const MoeGatingTopKTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    paddedCount_ = tilingData_->perGroupExpertCountAlign;
    rowBatch_ = tilingData_->vmsCount;
    runs_ = paddedCount_ / ONE_REPEAT_SORT_NUM;
    addBias_ = tilingData_->addBias == 1;
    outFlag_ = tilingData_->outFlag == 1;
    k_ = tilingData_->k;
    renorm_ = tilingData_->renorm;
    normType_ = tilingData_->normType;
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;


    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ float *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    // init ub buf
    int64_t rp = rowBatch_ * paddedCount_;
    pipe_->InitBuffer(xBufPing_, rp * sizeof(T));
    pipe_->InitBuffer(xBufPong_, rp * sizeof(T));
    if constexpr (!IsSameType<T, float>::value) {
        pipe_->InitBuffer(xFp32Buf_, rp * sizeof(float));
    }
    evMte2V_ = pipe_->FetchEventID(HardEvent::MTE2_V);
    if (addBias_) {
        pipe_->InitBuffer(xNormBuf_, rp * sizeof(float));
    }
    pipe_->InitBuffer(xNormWithBiasBuf_, rp * sizeof(float));
    if (addBias_) {
        pipe_->InitBuffer(biasFp32Buf_, paddedCount_ * sizeof(float));
        if constexpr (!IsSameType<T, float>::value) {
            pipe_->InitBuffer(biasStageBuf_, paddedCount_ * sizeof(T));
        }
        pipe_->InitBuffer(biasBatchBuf_, rp * sizeof(float));
    }
    pipe_->InitBuffer(expertIdBuf_, rp * sizeof(int32_t));
    pipe_->InitBuffer(sortedBuf_, rp * sizeof(float) * CONSTANT_TWO);
    pipe_->InitBuffer(mergeBuf_, rp * sizeof(float) * CONSTANT_TWO);
    pipe_->InitBuffer(finalBuf_, rowBatch_ * REPEAT_BYTES * sizeof(float));
    accWinRows_ = ACC_COPYOUT_BATCHES * rowBatch_;
    pipe_->InitBuffer(yAccBuf_, accWinRows_ * ONE_REPEAT_SORT_NUM * sizeof(T));
    pipe_->InitBuffer(idxAccBuf_, accWinRows_ * ONE_REPEAT_SORT_NUM * sizeof(int32_t));
    eventAccReuse_ = pipe_->FetchEventID(HardEvent::MTE3_V);
    pipe_->InitBuffer(byteIdxBuf_, rowBatch_ * ONE_REPEAT_SORT_NUM * sizeof(int32_t));
    pipe_->InitBuffer(rowBaseBuf_, rowBatch_ * ONE_REPEAT_SORT_NUM * sizeof(int32_t));
    pipe_->InitBuffer(yBuf_, rowBatch_ * ONE_REPEAT_SORT_NUM * sizeof(float));
    pipe_->InitBuffer(calcTmpBuf_, tilingData_->calTmpBufUbSize);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupBatch<T>::InitConstant()
{
    int64_t rp = rowBatch_ * paddedCount_;
    LocalTensor<T> ping = xBufPing_.Get<T>();
    LocalTensor<T> pong = xBufPong_.Get<T>();
    if constexpr (IsSameType<T, float>::value) {
        Duplicate(ping.template ReinterpretCast<int32_t>(), FP32_NEG_INF_BITS, rp);
        Duplicate(pong.template ReinterpretCast<int32_t>(), FP32_NEG_INF_BITS, rp);
    } else if constexpr (IsSameType<T, half>::value) {
        Duplicate(ping.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0xFC00), rp);
        Duplicate(pong.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0xFC00), rp);
    } else {
        Duplicate(ping.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0xFF80), rp);
        Duplicate(pong.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0xFF80), rp);
    }
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

    if (addBias_) {
        LocalTensor<float> biasF = biasFp32Buf_.Get<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(biasF, biasGm_, copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else {
            LocalTensor<T> biasS = biasStageBuf_.Get<T>();
            DataCopyPad(biasS, biasGm_, copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Cast(biasF, biasS, RoundMode::CAST_NONE, expertCount_);
            PipeBarrier<PIPE_V>();
        }
        if (paddedCount_ > expertCount_) {
            Duplicate(biasF[expertCount_].template ReinterpretCast<int32_t>(), FP32_NEG_INF_BITS,
                      paddedCount_ - expertCount_);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> biasBatch = biasBatchBuf_.Get<float>();
        for (int64_t r = 0; r < rowBatch_; r++) {
            DataCopy(biasBatch[r * paddedCount_], biasF, paddedCount_);
        }
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<int32_t> eid = expertIdBuf_.Get<int32_t>();
    ArithProgression(eid, static_cast<int32_t>(0), static_cast<int32_t>(1), paddedCount_);
    PipeBarrier<PIPE_V>();
    for (int64_t r = 1; r < rowBatch_; r++) {
        DataCopy(eid[r * paddedCount_], eid, paddedCount_);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> pos = byteIdxBuf_.Get<int32_t>();
    LocalTensor<int32_t> rowBase = rowBaseBuf_.Get<int32_t>();
    LocalTensor<float> fpos = calcTmpBuf_.Get<float>()[REPEAT_BLOCKS * CONSTANT_EIGHT];
    int64_t totalIdx = rowBatch_ * ONE_REPEAT_SORT_NUM;
    ArithProgression(pos, static_cast<int32_t>(0), static_cast<int32_t>(1), totalIdx);
    PipeBarrier<PIPE_V>();
    Cast(fpos, pos, RoundMode::CAST_ROUND, totalIdx);
    PipeBarrier<PIPE_V>();
    Muls(fpos, fpos, 1.0f / static_cast<float>(ONE_REPEAT_SORT_NUM), totalIdx);
    PipeBarrier<PIPE_V>();
    Cast(rowBase, fpos, RoundMode::CAST_TRUNC, totalIdx);
    PipeBarrier<PIPE_V>();
    Muls(rowBase, rowBase, static_cast<int32_t>(paddedCount_ * sizeof(float)), totalIdx);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupBatch<T>::CopyInX(int64_t bufIdx, int64_t startRow,
                                                                  int64_t rows)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = rows;
    copyParams.blockLen = expertCount_ * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = (paddedCount_ - expertCount_) * sizeof(T) / BLOCK_BYTES;
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    LocalTensor<T> xS = bufIdx == 0 ? xBufPing_.Get<T>() : xBufPong_.Get<T>();
    DataCopyPad(xS, xGm_[startRow * expertCount_], copyParams, padParams);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupBatch<T>::ComputeX(int64_t bufIdx, int64_t rows)
{
    LocalTensor<float> xNormWB = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> norm = NormOut();
    int64_t total = rows * paddedCount_;

    LocalTensor<T> xS = bufIdx == 0 ? xBufPing_.Get<T>() : xBufPong_.Get<T>();
    LocalTensor<float> xF;
    if constexpr (IsSameType<T, float>::value) {
        xF = xS;
    } else {
        xF = xFp32Buf_.Get<float>();
        Cast(xF, xS, RoundMode::CAST_NONE, total);
        PipeBarrier<PIPE_V>();
    }

    if (normType_ == 1) {
        LocalTensor<uint8_t> tmp = calcTmpBuf_.Get<uint8_t>();
        Sigmoid(norm, xF, tmp, total);
        PipeBarrier<PIPE_V>();
    } else if (normType_ == 0) {
        LocalTensor<float> red = calcTmpBuf_.Get<float>();
        LocalTensor<float> wk = calcTmpBuf_.Get<float>()[ONE_REPEAT_SORT_NUM];
        for (int64_t r = 0; r < rows; r++) {
            ReduceMax(red, xF[r * paddedCount_], wk, paddedCount_);
            event_t evtVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtVToS);
            WaitFlag<HardEvent::V_S>(evtVToS);
            float maxValue = red.GetValue(0);
            event_t evtSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evtSToV);
            WaitFlag<HardEvent::S_V>(evtSToV);
            Adds(norm[r * paddedCount_], xF[r * paddedCount_], -maxValue, paddedCount_);
        }
        PipeBarrier<PIPE_V>();
        Exp(norm, norm, total);
        PipeBarrier<PIPE_V>();
        for (int64_t r = 0; r < rows; r++) {
            ReduceSum(red, norm[r * paddedCount_], wk, paddedCount_);
            event_t evtVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtVToS);
            WaitFlag<HardEvent::V_S>(evtVToS);
            float sumValue = red.GetValue(0);
            event_t evtSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evtSToV);
            WaitFlag<HardEvent::S_V>(evtSToV);
            Muls(norm[r * paddedCount_], norm[r * paddedCount_], 1.0f / sumValue, paddedCount_);
        }
        PipeBarrier<PIPE_V>();
    }
    if (addBias_) {
        LocalTensor<float> biasBatch = biasBatchBuf_.Get<float>();
        Add(xNormWB, norm, biasBatch, total);
        PipeBarrier<PIPE_V>();
    } else {
        if (paddedCount_ > expertCount_) {
            for (int64_t r = 0; r < rows; r++) {
                Duplicate(norm[r * paddedCount_ + expertCount_].template ReinterpretCast<int32_t>(),
                          FP32_NEG_INF_BITS, paddedCount_ - expertCount_);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupBatch<T>::TopK(int64_t rows, LocalTensor<int32_t> idxDst,
                                                               bool waitAcc)
{
    LocalTensor<float> sorted = sortedBuf_.Get<float>();
    LocalTensor<float> merged = mergeBuf_.Get<float>();
    LocalTensor<float> compact = sortedBuf_.Get<float>();
    LocalTensor<float> finalTensor = finalBuf_.Get<float>();
    LocalTensor<uint32_t> eid = expertIdBuf_.Get<uint32_t>();
    LocalTensor<float> xNormWB = xNormWithBiasBuf_.Get<float>();

    Sort32(sorted, xNormWB, eid, rows * runs_);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> src = sorted;
    int64_t curRuns = runs_;
    while (curRuns > MERGE_LIST_FOUR) {
        int64_t groups = curRuns / MERGE_LIST_FOUR;
        MrgSort4Info params;
        params.elementLengths[0] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[1] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[2] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[3] = ONE_REPEAT_SORT_NUM;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = groups * rows;
        MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[ONE_REPEAT_SORT_NUM * CONSTANT_TWO];
        srcList.src3 = src[ONE_REPEAT_SORT_NUM * CONSTANT_FOUR];
        srcList.src4 = src[ONE_REPEAT_SORT_NUM * CONSTANT_TWO * CONSTANT_THREE];
        MrgSort(merged, srcList, params);
        PipeBarrier<PIPE_V>();

        curRuns = groups;
        DataCopyParams compactParams{static_cast<uint16_t>(curRuns * rows), REPEAT_BLOCKS,
                                     static_cast<uint16_t>(ONE_REPEAT_SORT_NUM - REPEAT_BLOCKS), 0};
        DataCopy(compact, merged, compactParams);
        PipeBarrier<PIPE_V>();
        src = compact;
    }

    LocalTensor<float> finalSrc = src;
    int64_t finalWidthFloats = ONE_REPEAT_SORT_NUM * CONSTANT_TWO;
    if (curRuns == MERGE_LIST_FOUR) {
        MrgSort4Info params;
        params.elementLengths[0] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[1] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[2] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[3] = ONE_REPEAT_SORT_NUM;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = rows;
        MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[ONE_REPEAT_SORT_NUM * CONSTANT_TWO];
        srcList.src3 = src[ONE_REPEAT_SORT_NUM * CONSTANT_FOUR];
        srcList.src4 = src[ONE_REPEAT_SORT_NUM * CONSTANT_TWO * CONSTANT_THREE];
        MrgSort(finalTensor, srcList, params);
        PipeBarrier<PIPE_V>();
        finalSrc = finalTensor;
        finalWidthFloats = REPEAT_BYTES;
    } else if (curRuns == MERGE_LIST_THREE) {
        MrgSort4Info params;
        params.elementLengths[0] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[1] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[2] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[3] = 0;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b111;
        params.repeatTimes = 1;
        int64_t compactRowStride = ONE_REPEAT_SORT_NUM * CONSTANT_TWO * MERGE_LIST_THREE; // 192
        int64_t finalRowStride = ONE_REPEAT_SORT_NUM * CONSTANT_FOUR;
        for (int64_t r = 0; r < rows; r++) {
            MrgSortSrcList<float> srcList;
            srcList.src1 = src[r * compactRowStride];
            srcList.src2 = src[r * compactRowStride + ONE_REPEAT_SORT_NUM * CONSTANT_TWO];
            srcList.src3 = src[r * compactRowStride + ONE_REPEAT_SORT_NUM * CONSTANT_FOUR];
            srcList.src4 = src[r * compactRowStride];
            MrgSort(finalTensor[r * finalRowStride], srcList, params);
        }
        PipeBarrier<PIPE_V>();
        finalSrc = finalTensor;
        finalWidthFloats = ONE_REPEAT_SORT_NUM * CONSTANT_FOUR;
    } else if (curRuns == MERGE_LIST_TWO) {
        MrgSort4Info params;
        params.elementLengths[0] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[1] = ONE_REPEAT_SORT_NUM;
        params.elementLengths[2] = 0;
        params.elementLengths[3] = 0;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b11;
        params.repeatTimes = 1;
        int64_t rowStride = ONE_REPEAT_SORT_NUM * CONSTANT_FOUR; // 128
        for (int64_t r = 0; r < rows; r++) {
            MrgSortSrcList<float> srcList;
            srcList.src1 = src[r * rowStride];
            srcList.src2 = src[r * rowStride + ONE_REPEAT_SORT_NUM * CONSTANT_TWO];
            srcList.src3 = src[r * rowStride];
            srcList.src4 = src[r * rowStride];
            MrgSort(finalTensor[r * rowStride], srcList, params);
        }
        PipeBarrier<PIPE_V>();
        finalSrc = finalTensor;
        finalWidthFloats = ONE_REPEAT_SORT_NUM * CONSTANT_FOUR; // 128
    }

    LocalTensor<int32_t> pairs = mergeBuf_.Get<int32_t>();
    LocalTensor<int32_t> finalSrcI32 = finalSrc.template ReinterpretCast<int32_t>();
    DataCopyParams pairParams;
    pairParams.blockCount = rows;
    pairParams.blockLen = REPEAT_BLOCKS;
    pairParams.srcStride = finalWidthFloats * sizeof(float) / BLOCK_BYTES - REPEAT_BLOCKS;
    pairParams.dstStride = 0;
    DataCopy(pairs, finalSrcI32, pairParams);
    PipeBarrier<PIPE_V>();

    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = rows;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = REPEAT_BLOCKS;
    gatherMaskParams.src1RepeatStride = 0;
    uint8_t src1Pattern = 2;
    uint64_t rsvdCnt = 0;
    if (waitAcc) {
        WaitFlag<HardEvent::MTE3_V>(eventAccReuse_);
    }
    GatherMask(idxDst, pairs, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupBatch<T>::TailAndCopyOut(int64_t startRow, int64_t rows,
                                                                         int64_t accSlotRows, int64_t flushRows)
{
    LocalTensor<int32_t> topKIdx = idxAccBuf_.Get<int32_t>()[accSlotRows * ONE_REPEAT_SORT_NUM];
    LocalTensor<int32_t> byteIdx = byteIdxBuf_.Get<int32_t>();
    LocalTensor<int32_t> rowBase = rowBaseBuf_.Get<int32_t>();
    LocalTensor<float> yF = yBuf_.Get<float>();
    LocalTensor<float> xNorm = NormOut();
    int64_t total = rows * ONE_REPEAT_SORT_NUM;

    Muls(byteIdx, topKIdx, static_cast<int32_t>(sizeof(float)), total);
    PipeBarrier<PIPE_V>();
    Add(byteIdx, byteIdx, rowBase, total);
    PipeBarrier<PIPE_V>();
    Gather(yF, xNorm, byteIdx.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), total);
    PipeBarrier<PIPE_V>();

    bool needRenorm = (normType_ == 1) ||
                      (normType_ == 0 && renorm_ == 1); // softmax + renorm
    if (needRenorm) {
        LocalTensor<float> sums = calcTmpBuf_.Get<float>();
        LocalTensor<float> brcb = calcTmpBuf_.Get<float>()[ONE_REPEAT_SORT_NUM * CONSTANT_TWO];
        WholeReduceSum<float>(sums, yF, k_, rows, 1, 1, ONE_REPEAT_SORT_NUM * sizeof(float) / BLOCK_BYTES);
        PipeBarrier<PIPE_V>();
        Adds(sums, sums, eps_, rows);
        PipeBarrier<PIPE_V>();
        Brcb(brcb, sums, (rows + CONSTANT_EIGHT - 1) / CONSTANT_EIGHT, {1, CONSTANT_EIGHT});
        PipeBarrier<PIPE_V>();
        Div(yF, yF, brcb, ONE_REPEAT_SORT_NUM, rows,
            {1, 1, 0, ONE_REPEAT_SORT_NUM * sizeof(float) / BLOCK_BYTES,
             ONE_REPEAT_SORT_NUM * sizeof(float) / BLOCK_BYTES, 1});
        PipeBarrier<PIPE_V>();
    }

    int64_t slot = accSlotRows * ONE_REPEAT_SORT_NUM;
    if constexpr (IsSameType<T, float>::value) {
        LocalTensor<float> yAccF = yAccBuf_.Get<float>();
        Muls(yAccF[slot], yF, routedScalingFactor_, total);
    } else {
        LocalTensor<T> yAcc = yAccBuf_.Get<T>();
        Muls(yF, yF, routedScalingFactor_, total);
        PipeBarrier<PIPE_V>();
        Cast(yAcc[slot], yF, RoundMode::CAST_RINT, total);
    }
    if (flushRows > 0) {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint32_t yRowBytes = ONE_REPEAT_SORT_NUM * sizeof(T);
        uint32_t yBlkPad = (k_ * sizeof(T) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
        uint32_t iBlkPad = (k_ * sizeof(int32_t) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
        DataCopyExtParams copyYParams{static_cast<uint16_t>(flushRows), static_cast<uint32_t>(k_ * sizeof(T)),
                                      static_cast<uint32_t>((yRowBytes - yBlkPad) / BLOCK_BYTES), 0, 0};
        DataCopyExtParams copyIParams{static_cast<uint16_t>(flushRows),
                                      static_cast<uint32_t>(k_ * sizeof(int32_t)),
                                      static_cast<uint32_t>((ONE_REPEAT_SORT_NUM * sizeof(int32_t) - iBlkPad) /
                                                            BLOCK_BYTES),
                                      0, 0};
        int64_t accStartRow = startRow - accSlotRows;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(yGm_[accStartRow * k_], yAccBuf_.Get<float>(), copyYParams);
        } else {
            DataCopyPad(yGm_[accStartRow * k_], yAccBuf_.Get<T>(), copyYParams);
        }
        DataCopyPad(expertIdxGm_[accStartRow * k_], idxAccBuf_.Get<int32_t>(), copyIParams);
        SetFlag<HardEvent::MTE3_V>(eventAccReuse_);
    }
    if (outFlag_) {
        if (flushRows == 0) {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        }
        DataCopyExtParams copyOParams{static_cast<uint16_t>(rows),
                                      static_cast<uint32_t>(expertCount_ * sizeof(float)),
                                      static_cast<uint32_t>((paddedCount_ - expertCount_) * sizeof(float) /
                                                            BLOCK_BYTES),
                                      0, 0};
        DataCopyPad(outGm_[startRow * expertCount_], xNorm, copyOParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupBatch<T>::Process()
{
    InitConstant();
    LocalTensor<int32_t> idxAcc = idxAccBuf_.Get<int32_t>();
    int64_t accRows = 0;
    int64_t windows = 0;
    CopyInX(0, 0, Min(rowBatch_, curCoreRowCount_));
    SetFlag<HardEvent::MTE2_V>(evMte2V_);
    int64_t batchIdx = 0;
    for (int64_t startRow = 0; startRow < curCoreRowCount_; startRow += rowBatch_) {
        int64_t rows = Min(rowBatch_, curCoreRowCount_ - startRow);
        bool last = (startRow + rows == curCoreRowCount_);
        if (!last) {
            CopyInX(1 - (batchIdx & 1), startRow + rows,
                    Min(rowBatch_, curCoreRowCount_ - startRow - rows));
        }
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        ComputeX(batchIdx & 1, rows);
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        TopK(rows, idxAcc[accRows * ONE_REPEAT_SORT_NUM], accRows == 0 && windows > 0);
        int64_t flushRows = (last || accRows + rows == accWinRows_) ? accRows + rows : 0;
        TailAndCopyOut(startRow, rows, accRows, flushRows);
        if (flushRows > 0) {
            accRows = 0;
            windows++;
        } else {
            accRows += rows;
        }
        batchIdx++;
    }
    WaitFlag<HardEvent::MTE2_V>(evMte2V_);
    if (windows > 0) {
        WaitFlag<HardEvent::MTE3_V>(eventAccReuse_);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_WITHOUT_GROUP_BATCH_H
