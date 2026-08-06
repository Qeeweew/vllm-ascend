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
 * \file compressor_epilogue_tools.h
 * \brief 放算子都需要、与算子联系紧密、但是又不方便单独独立出来的公共工具
 */

#ifndef COMPRESSOR_EPILOGUE_TOOLS_H
#define COMPRESSOR_EPILOGUE_TOOLS_H

#include "compressor_epilogue_comm.h"

using namespace AscendC;

namespace CompressorEpilogue {

struct ToolsParams {
    uint32_t seqSize = 0U;
    uint32_t cmpRatio = 0U;
};

template <typename COMP>
class CompressorEpilogueTools {
public:
    __aicore__ inline CompressorEpilogueTools()
    {
    }

    __aicore__ inline void Init(__gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos);

    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint32_t GetTIdxByBatch(uint32_t bIdx);

public:
    ToolsParams toolParams_{};
    bool isExistSeqUsed_ = false;

private:
    bool isExistStartPos_ = false;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
};

template <typename COMP>
__aicore__ inline void CompressorEpilogueTools<COMP>::Init(__gm__ uint8_t *startPos, __gm__ uint8_t *seqUsed,
                                                   __gm__ uint8_t *cuSeqlens)
{
    isExistStartPos_ = (startPos != nullptr);
    if (isExistStartPos_) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }

    isExistSeqUsed_ = (seqUsed != nullptr);
    if (isExistSeqUsed_) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }

    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueTools<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed_) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return toolParams_.seqSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueTools<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return (uint32_t)startPosGm_.GetValue(bIdx);
    } else {
        return 0;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueTools<COMP>::GetSeqLength(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return toolParams_.seqSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueTools<COMP>::GetTIdxByBatch(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return (uint32_t)(cuSeqlensGm_.GetValue(bIdx));
    } else {
        return toolParams_.seqSize * bIdx;
    }
}

// iterator
struct SliceInfo {
    __aicore__ inline SliceInfo(){};
    __aicore__ inline SliceInfo(uint32_t bIdx, uint32_t sIdx) : bIdx(bIdx), sIdx(sIdx){};

    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t bSeqUsed = 0U;
    uint32_t bStartPos = 0U;

    uint32_t headHolderSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailHolderSeqCnt = 0U;

    uint32_t dealSeqCnt = 0;
    uint32_t dealTcSize = 0U;
    uint32_t compressTcSize = 0U;
};

struct Vec1SliceInfo : public SliceInfo {
    __aicore__ inline Vec1SliceInfo(){};
    __aicore__ inline Vec1SliceInfo(uint32_t bIdx, uint32_t sIdx) : SliceInfo(bIdx, sIdx){};
    __aicore__ inline Vec1SliceInfo(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt)
        : SliceInfo(bIdx, sIdx), dealedSeqCnt(dealedSeqCnt){};

    uint32_t dealedSeqCnt = 0U;
    uint32_t dealedTcCnt = 0U;
    uint32_t bSeqLength = 0U;
    uint32_t compressor_epilogueedScCnt = 0U;
    bool isFirst = false;
    bool isLast = false;
};

struct StatisticInfo {
    __aicore__ inline StatisticInfo(){};
    __aicore__ inline StatisticInfo(uint32_t actualTcCnt, uint32_t dealSeqCnt, uint32_t compressor_epilogueScCnt)
        : actualTcCnt(actualTcCnt), dealSeqCnt(dealSeqCnt), compressor_epilogueScCnt(compressor_epilogueScCnt){};

    uint32_t actualTcCnt = 0U;
    uint32_t dealSeqCnt = 0U;
    uint32_t compressor_epilogueScCnt = 0U;
};

template <typename COMP>
class CompressorEpilogueVec1SliceIterator {
public:
    __aicore__ inline CompressorEpilogueVec1SliceIterator(CompressorEpilogueTools<COMP> &tools) : tools_(tools)
    {
    }

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt, uint32_t compressor_epilogueedScCnt);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetDealedSeqCnt(uint32_t dealedSeqCnt);
    __aicore__ inline void SetDealedTcCnt(uint32_t dealedTcCnt);
    __aicore__ inline void SetCompressorEpilogueedScCnt(uint32_t compressor_epilogueedScCnt);
    __aicore__ inline void SetNeedDealTcSize(uint32_t needDealTcSize);
    __aicore__ inline void SetNeedDealTcSize(uint32_t needDealTcSize, uint32_t canDealTcSize);
    __aicore__ inline uint32_t GetNeedDealTcSize();
    __aicore__ inline bool IsEnd();
    template <bool IS_STATISTIC = false>
    __aicore__ inline void IteratorSlice();
    __aicore__ inline Vec1SliceInfo &GetSlice();
    template <bool IS_STATISTIC = false>
    __aicore__ inline StatisticInfo &FullIteratorSlice();

private:
    CompressorEpilogueTools<COMP> &tools_;

    bool isFirst_ = true;
    Vec1SliceInfo sliceInfo_{};
    StatisticInfo statisticInfo_{};
    uint32_t needDealTcSize_ = 0U;
    uint32_t batch_size_ = 0U;
};

template <typename COMP>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
{
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    while (tools_.GetSeqLength(sliceInfo_.bIdx) == 0) {
        sliceInfo_.bIdx++;
        if (sliceInfo_.bIdx == batch_size_) {
            sliceInfo_.bIdx = 0;
        }
    }
    sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
    sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
    sliceInfo_.bSeqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
    isFirst_ = true;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt,
                                                                uint32_t compressor_epilogueedScCnt)
{
    Reset(bIdx, sIdx);
    SetDealedSeqCnt(dealedSeqCnt);
    SetCompressorEpilogueedScCnt(compressor_epilogueedScCnt);
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::SetDealedSeqCnt(uint32_t dealedSeqCnt)
{
    this->sliceInfo_.dealedSeqCnt = dealedSeqCnt;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::SetCompressorEpilogueedScCnt(uint32_t compressor_epilogueedScCnt)
{
    this->sliceInfo_.compressor_epilogueedScCnt = compressor_epilogueedScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::SetDealedTcCnt(uint32_t dealedTcCnt)
{
    this->sliceInfo_.dealedTcCnt = dealedTcCnt;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::SetNeedDealTcSize(uint32_t needDealTcSize)
{
    this->needDealTcSize_ = needDealTcSize;
}

template <typename COMP>
template <bool IS_STATISTIC>
__aicore__ inline void CompressorEpilogueVec1SliceIterator<COMP>::IteratorSlice()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if constexpr (IS_STATISTIC) {
        statisticInfo_.actualTcCnt += sliceInfo_.dealTcSize;
        statisticInfo_.compressor_epilogueScCnt += sliceInfo_.compressTcSize;
    }
    needDealTcSize_ -= sliceInfo_.dealTcSize;
    sliceInfo_.dealedSeqCnt += sliceInfo_.validSeqCnt;
    sliceInfo_.compressor_epilogueedScCnt += sliceInfo_.compressTcSize;
    sliceInfo_.sIdx += sliceInfo_.validSeqCnt;
    if (sliceInfo_.sIdx >= sliceInfo_.bSeqUsed) {
        do {
            uint32_t seqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
            if (sliceInfo_.bSeqUsed < seqLength) {
                uint32_t nextAlignSIdx = Align(sliceInfo_.bStartPos + sliceInfo_.sIdx, cmpRatio) - sliceInfo_.bStartPos;
                sliceInfo_.dealedSeqCnt += nextAlignSIdx - sliceInfo_.sIdx;
                uint32_t tcGap = CeilDivT(static_cast<int32_t>(seqLength - nextAlignSIdx),
                                    static_cast<int32_t>(cmpRatio));
                if (sliceInfo_.bSeqUsed == 0 && nextAlignSIdx > sliceInfo_.sIdx) {
                    // 此时bseqused所在压缩块未被纳入计算
                    tcGap++;
                }
                sliceInfo_.sIdx = nextAlignSIdx;
                if (needDealTcSize_ < tcGap) {
                    sliceInfo_.dealedSeqCnt += needDealTcSize_ * cmpRatio;
                    sliceInfo_.sIdx += needDealTcSize_ * cmpRatio;
                    needDealTcSize_ = 0;
                    break;
                }
                sliceInfo_.dealedSeqCnt += seqLength - sliceInfo_.sIdx;
                needDealTcSize_ -= tcGap;
            }
            sliceInfo_.bIdx++;
            if (sliceInfo_.bIdx == batch_size_) {
                sliceInfo_.bIdx = 0;
            }
            sliceInfo_.sIdx = 0;
            sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
        } while (sliceInfo_.bSeqUsed == 0);
        sliceInfo_.bSeqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
        sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
    }
    if (isFirst_) {
        isFirst_ = false;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueVec1SliceIterator<COMP>::GetNeedDealTcSize()
{
    return needDealTcSize_;
}


template <typename COMP>
__aicore__ inline bool CompressorEpilogueVec1SliceIterator<COMP>::IsEnd()
{
    return (needDealTcSize_ == 0);
}

template <typename COMP>
__aicore__ inline Vec1SliceInfo &CompressorEpilogueVec1SliceIterator<COMP>::GetSlice()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (sliceInfo_.bSeqUsed < sliceInfo_.sIdx) {
        sliceInfo_.headHolderSeqCnt = 0;
        sliceInfo_.validSeqCnt = 0;
        sliceInfo_.tailHolderSeqCnt = 0;
        sliceInfo_.dealTcSize = 0;
        sliceInfo_.compressTcSize = 0;
    } else {
        // 计算头部占位行数、有效数据行数、尾部占位行数
        sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % cmpRatio;
        sliceInfo_.validSeqCnt = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
        if (CeilDivT(sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt, cmpRatio) > needDealTcSize_) {
            sliceInfo_.validSeqCnt = needDealTcSize_ * cmpRatio - sliceInfo_.headHolderSeqCnt;
        }
        uint32_t globalTotalSeqCnt = sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt;
        sliceInfo_.tailHolderSeqCnt = Align(globalTotalSeqCnt, cmpRatio) - globalTotalSeqCnt;

        // 计算本次可以处理的Tc个数
        sliceInfo_.dealTcSize =
            (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt + sliceInfo_.tailHolderSeqCnt) / cmpRatio;

        sliceInfo_.compressTcSize =
            (sliceInfo_.headHolderSeqCnt + min(sliceInfo_.validSeqCnt, sliceInfo_.bSeqUsed - sliceInfo_.sIdx)) /
            cmpRatio;
    }

    sliceInfo_.isFirst = isFirst_;
    sliceInfo_.isLast =
        sliceInfo_.bSeqUsed > sliceInfo_.sIdx &&
        CeilDivT(sliceInfo_.headHolderSeqCnt + sliceInfo_.bSeqUsed - sliceInfo_.sIdx, cmpRatio) >= needDealTcSize_;

    return sliceInfo_;
}

template <typename COMP>
template <bool IS_STATISTIC>
__aicore__ inline StatisticInfo &CompressorEpilogueVec1SliceIterator<COMP>::FullIteratorSlice()
{
    if constexpr (IS_STATISTIC) {
        statisticInfo_ = {0U, 0U, 0U};
        Vec1SliceInfo tempSliceInfo = GetSlice();
        while (!IsEnd()) {
            GetSlice();
            IteratorSlice<IS_STATISTIC>();
        }
        Vec1SliceInfo sliceInfo = GetSlice();
        statisticInfo_.dealSeqCnt = sliceInfo.dealedSeqCnt - tempSliceInfo.dealedSeqCnt;
    } else {
        while (!IsEnd()) {
            GetSlice();
            IteratorSlice<IS_STATISTIC>();
        }
    }
    return statisticInfo_;
}
} // namespace CompressorEpilogue

#endif
