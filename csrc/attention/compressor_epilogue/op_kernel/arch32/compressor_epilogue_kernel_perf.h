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
 * \file compressor_epilogue_kernel_perf.h
 * \brief
 */

#ifndef COMPRESSOR_EPILOGUE_KERNEL_PERF_H
#define COMPRESSOR_EPILOGUE_KERNEL_PERF_H

#include "compressor_epilogue_comm.h"
#include "compressor_epilogue_template_tiling_key.h"
#include "compressor_epilogue_tiling_data.h"
#include "compressor_epilogue_tools.h"
#include "compressor_epilogue_block_vec_perf.h"


using namespace AscendC;

namespace CompressorEpilogue {



struct BasicBlockInfo {
    uint32_t bIdx = 0;
    uint32_t sIdx = 0;
    uint32_t compressedTcNum = 0;
    uint32_t dealSeqCnt = 0;
    uint32_t dealTcNum = 0;
};

struct BatchInfo {
    uint32_t tcNum = 0;
    uint32_t compressedTcNum = 0;
    uint32_t remSeqCnt = 0;
    uint32_t seqCnt = 0;
    uint32_t seqUsedCnt = 0;
    uint32_t headHolderSeq = 0;
    uint32_t bStartPos = 0;
    uint32_t bIdx = 0;
    uint32_t sIdx = 0;
};

template <typename COMP>
class CompressorEpilogueKernelPerf {
public:
    __aicore__ inline CompressorEpilogueKernelPerf(TPipe* pipe, const optiling::CompressorEpilogueTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *mmKv,
        __gm__ uint8_t *mmScore,
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
    __aicore__ inline void Process();

private:
    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void SetBaseSize();
    // 获取基本块数量
    __aicore__ inline uint32_t GetLoopTimes();
    __aicore__ inline void SkipInvalidBatch(BatchInfo &batchInfo);
    __aicore__ inline void UpdateCurGroup(BasicBlockInfo &basicBlockInfo, BatchInfo batchInfo, uint32_t &curGroupQuota, uint32_t curDealSeq);
    __aicore__ inline BasicBlockInfo SkipOneLoop(BatchInfo &batchInfo);
    // 计算分核基本信息
    __aicore__ inline void CalcSplitCoreInfo();

    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);

    __aicore__ inline void CalcVec1Params(Vec1RunInfo &vec1Info, BatchInfo &batchInfo, uint32_t loopIdx);

    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
    using T = float;
    using MM1_OUT_T = T;
    using VEC1_OUT_T = T;


    // ==============================TilingData&TPipe==============================
    TPipe* pipe_;
    const optiling::CompressorEpilogueTilingData* __restrict tilingData_;
    // ===========================Workspace Global Tensor===========================
    GlobalTensor<X_T> mmKvGm_;
    GlobalTensor<X_T> mmScoreGm_;
    // ================================Task Info====================================
    CompressorEpilogueTools<COMP> tools_;
    ConstInfo constInfo{};
    uint32_t aiCoreIdx = 0;

    // ==============================Service Define==============================
    CompressorEpilogueBlockVectorPerf<COMP> blockVec_;

    uint32_t loopTimes = 0;
    bool isFirstUpdateCurGroup = true;
};

template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::Init(
        __gm__ uint8_t *mmKv,
        __gm__ uint8_t *mmScore,
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
    constInfo.aiCoreIdx = GetBlockIdx();  // AIV-only：核号即逻辑核号
    InitTilingData();
    // init tools
    tools_.toolParams_.seqSize = tilingData_->baseParams.seqSize;
    tools_.toolParams_.cmpRatio = tilingData_->baseParams.cmpRatio;
    tools_.Init(startPos, seqUsed, cuSeqlens);

    // 剔除尾部的无效batch
    for (; constInfo.batchSize > 0; --constInfo.batchSize) {
        uint32_t bSeqUsed = tools_.GetSeqLength(constInfo.batchSize - 1);
        if (bSeqUsed > 0) {
            break;
        }
    }

    // 所有batch的有效序列都为0时, 直接退出
    if (constInfo.batchSize == 0) {
        return;
    }

    // 1. 计算head_dim的切分大小, 构建ConstInfo的其他信息
    SetBaseSize(); // 设置基本块大小
    CalcSplitCoreInfo();
    // 2. 计算循环次数
    loopTimes = GetLoopTimes();
    // 3. 初始化block层（纯 AIV kernel，无 cube、无 workspace）
    mmKvGm_.SetGlobalBuffer((__gm__ X_T *)mmKv);
    mmScoreGm_.SetGlobalBuffer((__gm__ X_T *)mmScore);
    blockVec_.InitParams(constInfo, tools_);
    blockVec_.Init(stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,
                    cuSeqlens, seqUsed, startPos, cmpKvOut);
    blockVec_.InitBuffers(pipe_);
    blockVec_.InitVec1GlobalTensor(mmKvGm_, mmScoreGm_);
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::InitTilingData() {
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.sSize = tilingData_->baseParams.seqSize;
    constInfo.headDim = tilingData_->baseParams.headDim;
    constInfo.ropeHeadDim = tilingData_->baseParams.ropeHeadDim;
    constInfo.normEps = tilingData_->baseParams.normEps;
    constInfo.reciprocalD = tilingData_->baseParams.reciprocalD;
    constInfo.usedCoreNum = tilingData_->baseParams.usedCoreNum;

    constInfo.blockSize = tilingData_->pageAttentionParams.blockSize;
    constInfo.maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;
    constInfo.stateCacheStrideDim0 = tilingData_->baseParams.stateCacheStrideDim0;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::SetBaseSize()
{
    uint32_t mSize = 0;
    uint32_t minMBaseSize = 0;
    bool sameSeqUsed = true;
    uint32_t firstBatchSeqUsed = tools_.GetSeqLength(0);
    for (uint32_t i = 0; i < constInfo.batchSize; i++) {
        uint32_t bSeqUsed = tools_.GetSeqLength(i);
        uint32_t bStartPos = tools_.GetStartPos(i);
        // 获取m大小
        mSize += bSeqUsed;
        // 获取是否等长
        if (sameSeqUsed && (bSeqUsed != firstBatchSeqUsed)) {
            sameSeqUsed = false;
        }
        // 获取m轴最小切分大小
        if (minMBaseSize != constInfo.cmpRatio) {
            uint32_t startCmpIdx = bStartPos / constInfo.cmpRatio;
            uint32_t endCmpIdx = (bStartPos + bSeqUsed) / constInfo.cmpRatio;
            if (startCmpIdx == endCmpIdx) {
                if (bSeqUsed > minMBaseSize) {
                    minMBaseSize = bSeqUsed;
                }
            } else if (startCmpIdx + 1 == endCmpIdx) {
                uint32_t startCmpValidSeqCnt = constInfo.cmpRatio - (bStartPos % constInfo.cmpRatio);
                uint32_t endCmpValidSeqCnt = (bStartPos + bSeqUsed) % constInfo.cmpRatio;
                if (startCmpValidSeqCnt > minMBaseSize) {
                    minMBaseSize = startCmpValidSeqCnt;
                }
                if (endCmpValidSeqCnt > minMBaseSize) {
                    minMBaseSize = endCmpValidSeqCnt;
                }
            } else {
                minMBaseSize = constInfo.cmpRatio;
            }
        }
    }

    uint32_t aiCoreNum = constInfo.usedCoreNum;
    // mBaseSize 负载均衡启发式（沿用原 fused 调好的数值）：小 token 时按参与核数缩小基本块。
    // dBaseBlockNum 仅用于推导参与核数（行并行下 D 不切分，但保留原数值保证 mBaseSize 调整行为不变）
    uint32_t dBaseBlockNum = constInfo.headDim / 64;
    if (sameSeqUsed && mSize <= (constInfo.mBaseSize * (aiCoreNum / dBaseBlockNum))) {
        if (constInfo.headDim == 128) {
            dBaseBlockNum = 8;
        } else if (constInfo.headDim == 512) {
            dBaseBlockNum = 16;
        }
        // 核数足够时, 修改才生效
        if (aiCoreNum >= dBaseBlockNum) {
            uint32_t coreGroupNum = aiCoreNum / dBaseBlockNum;
            uint32_t newMBaseSize = (constInfo.batchSize + coreGroupNum - 1) / coreGroupNum * firstBatchSeqUsed;
            if (newMBaseSize > minMBaseSize && newMBaseSize < constInfo.mBaseSize) {
                constInfo.mBaseSize = newMBaseSize;
            }
        }
    }
}


template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::SkipInvalidBatch(BatchInfo &batchInfo)
{
    for (; batchInfo.bIdx < constInfo.batchSize; ++batchInfo.bIdx) {
        batchInfo.seqCnt = tools_.GetSeqLength(batchInfo.bIdx);
        if (batchInfo.seqCnt > 0) {
            break;
        }
    }
    batchInfo.remSeqCnt = batchInfo.seqCnt;
    if (tools_.isExistSeqUsed_) {
        batchInfo.seqUsedCnt = tools_.GetSeqUsed(batchInfo.bIdx);
    } else {
        batchInfo.seqUsedCnt = batchInfo.seqCnt;
    }
    if (batchInfo.bIdx < constInfo.batchSize) {
        batchInfo.bStartPos = tools_.GetStartPos(batchInfo.bIdx);
        batchInfo.sIdx = 0;
        batchInfo.headHolderSeq = batchInfo.bStartPos & (constInfo.cmpRatio - 1);
        batchInfo.tcNum = (batchInfo.bStartPos + batchInfo.seqCnt + constInfo.cmpRatio - 1) / constInfo.cmpRatio - batchInfo.bStartPos /  constInfo.cmpRatio;
        batchInfo.compressedTcNum = (batchInfo.bStartPos + batchInfo.seqUsedCnt) / constInfo.cmpRatio - batchInfo.bStartPos /  constInfo.cmpRatio;
    }
}


template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::UpdateCurGroup(BasicBlockInfo &basicBlockInfo,
                                BatchInfo batchInfo, uint32_t &curGroupQuota, uint32_t curDealSeq)
{
    // 更新当前组的信息
    if (curGroupQuota == 0 && !isFirstUpdateCurGroup) {
        return;
    }
    isFirstUpdateCurGroup = false;
    basicBlockInfo.bIdx = batchInfo.bIdx;
    uint32_t curGroupDealSeq = curGroupQuota < curDealSeq ? curGroupQuota : curDealSeq;
    basicBlockInfo.sIdx = batchInfo.sIdx + curGroupDealSeq;
    basicBlockInfo.dealSeqCnt += curGroupDealSeq;
    curGroupQuota -= curGroupDealSeq;
    // 结尾需要跳batch，需要考虑在当前组起始为末尾，或者当前组起始大于整个M轴
    if ((curGroupQuota == 0 || basicBlockInfo.bIdx == constInfo.batchSize - 1) && basicBlockInfo.sIdx == batchInfo.seqCnt) {
        basicBlockInfo.sIdx = 0;
        for (basicBlockInfo.bIdx++; basicBlockInfo.bIdx < constInfo.batchSize; ++basicBlockInfo.bIdx) {
            uint32_t seqCnt = tools_.GetSeqLength(basicBlockInfo.bIdx);
            if (seqCnt > 0) {
                break;
            }
        }
    }
}

template <typename COMP>
__aicore__ inline BasicBlockInfo CompressorEpilogueKernelPerf<COMP>::SkipOneLoop(BatchInfo &batchInfo)
{
    BasicBlockInfo basicBlockInfo{};
    isFirstUpdateCurGroup = true;
    uint32_t curGroupQuota = constInfo.mBaseSize * constInfo.curGroupIdx;       // m轴当前组起始
    bool curGroupStartFlag = false;
    uint32_t quota = constInfo.coreGroupNum * constInfo.mBaseSize;

    for (; batchInfo.bIdx < constInfo.batchSize;) {
        uint32_t curDealSeq = 0;
        uint32_t curDealTcNum = 0;
        uint32_t curDealCompressedTcNum = 0;
        // 无法处理完当前整个batch
        if (quota < batchInfo.remSeqCnt) {
            // 向下对齐r，
            if (quota > constInfo.cmpRatio - batchInfo.headHolderSeq) {
                uint32_t delta = (batchInfo.bStartPos + batchInfo.sIdx + quota) & (constInfo.cmpRatio - 1);  // 超出对齐的部分
                curDealSeq = quota - delta;
                quota -= curDealSeq;
                curDealTcNum = (curDealSeq + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
                curDealCompressedTcNum = min(curDealTcNum, batchInfo.compressedTcNum);
                // 更新当前组所需信息
                UpdateCurGroup(basicBlockInfo, batchInfo, curGroupQuota, curDealSeq);
                // 更新batch信息
                batchInfo.remSeqCnt = batchInfo.remSeqCnt - curDealSeq;
                batchInfo.sIdx = batchInfo.sIdx + curDealSeq;
                batchInfo.compressedTcNum -= curDealCompressedTcNum;
                batchInfo.tcNum -= curDealTcNum;
                // 更新loop信息
                basicBlockInfo.dealTcNum += curDealTcNum;
                basicBlockInfo.compressedTcNum += curDealCompressedTcNum;
            }
            break;
        } else {
            // 处理整个batch
            quota -= batchInfo.remSeqCnt;
            curDealSeq = batchInfo.remSeqCnt;
            curDealTcNum = batchInfo.tcNum;
            // 更新当前组所需信息
            UpdateCurGroup(basicBlockInfo, batchInfo, curGroupQuota, curDealSeq);
            // 更新batch和loop信息
            batchInfo.remSeqCnt = 0;
            basicBlockInfo.dealTcNum += batchInfo.tcNum;
            basicBlockInfo.compressedTcNum += batchInfo.compressedTcNum;
            batchInfo.bIdx++;
            SkipInvalidBatch(batchInfo);
        }
    }
    uint32_t totalDataSize = constInfo.coreGroupNum * constInfo.mBaseSize - quota;
    // 2. 当前组的起始偏移
    uint32_t currentGroupStart = constInfo.curGroupIdx * constInfo.mBaseSize;

    // 3. 安全判断
    if (currentGroupStart >= totalDataSize) {
        // 超出尾块
        basicBlockInfo.dealSeqCnt = 0;
    } else {
        // 还在有效范围内，计算剩余量
        uint32_t remaining = totalDataSize - currentGroupStart;
        basicBlockInfo.dealSeqCnt = (remaining < constInfo.mBaseSize) ? remaining : constInfo.mBaseSize;
    }

    return basicBlockInfo;
}


template <typename COMP>
__aicore__ inline uint32_t CompressorEpilogueKernelPerf<COMP>::GetLoopTimes()
{
    // 计算主循环次数
    uint32_t loopTimes = 0;
    BatchInfo batchInfo{};
    SkipInvalidBatch(batchInfo);
    for (;batchInfo.bIdx < constInfo.batchSize; ++loopTimes) {
        SkipOneLoop(batchInfo);
    }
    return loopTimes;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::CalcSplitCoreInfo()
{
    // 每核独占完整 D 维（行并行）：dBasicBlockNum=1，coreGroupNum=usedCoreNum，每核一组
    constInfo.dBasicBlockNum = 1;
    constInfo.coreGroupNum = constInfo.usedCoreNum;
    constInfo.curGroupIdx = constInfo.aiCoreIdx;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::ComputeVec1(const Vec1RunInfo &info) {
    blockVec_.ComputeVec1(info);
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::CalcVec1Params(Vec1RunInfo &vec1Info, BatchInfo &batchInfo, uint32_t loopIdx)
{
    vec1Info.bStart = batchInfo.bIdx;
    vec1Info.sStart = batchInfo.sIdx;
    BasicBlockInfo basicBlockInfo = SkipOneLoop(batchInfo);
    vec1Info.dealTcNum = basicBlockInfo.dealTcNum;
    vec1Info.dealScSize = basicBlockInfo.compressedTcNum;
}

template <typename COMP>
__aicore__ inline void CompressorEpilogueKernelPerf<COMP>::Process()
{
    // 所有batch的有效序列都为0时, 直接退出
    if (constInfo.batchSize == 0) {
        return;
    }

    BatchInfo batchInfo{};
    Vec1RunInfo vec1Info{};
    SkipInvalidBatch(batchInfo);
    // 完全串行：每核独立完成窗口装配->softmax->加权和->rms_norm->rope->输出，
    // 无跨核数据依赖，无 SyncAll / vec1Res workspace / vec2 阶段
    for (uint32_t i = 0; i < loopTimes; ++i) {
        CalcVec1Params(vec1Info, batchInfo, i);
        ComputeVec1(vec1Info);
    }
}

} // namespace CompressorEpilogue

#endif // COMPRESSOR_EPILOGUE_KERNEL_PERF_H
