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
 * \file compressor_epilogue_kernel_c128.h
 * \brief Kernel B：C128（cmpRatio=128, coff=1）——d 分块压缩 + workspace 中转两阶段 norm/rope
 *
 * 窗口 128 行 = 当前组（无 overlap）。窗口 128×512 fp32 = 256KB > UB，必须 d 分块：
 *   阶段一：任务 = (组, dChunk 64 列)，窗口装配→列 softmax→加权和得 cmpRow[64] fp32，
 *           写 GM workspace[scIdx*512 + dStart]（未 norm/rope）；state 写不变；
 *   SyncAll：全核屏障（tiling SetScheduleMode(1) 保证同调度），workspace 跨核可见；
 *   阶段二：压缩行按核均分（≤ maxScNum 行，C128 压缩比 128:1 行数很小），每行读回
 *           完整 512 维 → RmsNorm → RoPE（cos/sin 行号=行序号）→ CAST_RINT → cmp_kv。
 * 事件：阶段一同前（固定 id 自配对）；阶段二复用 ID0~ID3 相邻自配对，全局 1:1。
 */

#ifndef COMPRESSOR_EPILOGUE_KERNEL_C128_H
#define COMPRESSOR_EPILOGUE_KERNEL_C128_H

#include "compressor_epilogue_comm.h"
#include "compressor_epilogue_tiling_data.h"
#include "compressor_epilogue_tools.h"
#include "compressor_epilogue_vector_comm.h"

namespace CompressorEpilogue {

using optiling::CompressorEpilogueTilingData;

template <typename T, typename T_NORM, typename T_ROPE>
class CompressorEpilogueKernelC128 {
public:
    __aicore__ inline void Init(TPipe *pipe, const CompressorEpilogueTilingData *tilingData,
                                __gm__ uint8_t *mmKv, __gm__ uint8_t *mmScore, __gm__ uint8_t *stateCache,
                                __gm__ uint8_t *ape, __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin,
                                __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                __gm__ uint8_t *cmpKvOut, __gm__ uint8_t *workspace)
    {
        pipe_ = pipe;
        batchSize_ = tilingData->batchSize;
        headDim_ = tilingData->headDim;
        cmpRatio_ = tilingData->cmpRatio;
        blockSize_ = tilingData->blockSize;
        maxBlockNumPerBatch_ = tilingData->maxBlockNumPerBatch;
        stateStride0_ = tilingData->stateCacheStrideDim0;
        dChunkSize_ = tilingData->dChunkSize;
        maxGroupTaskNum_ = tilingData->maxGroupTaskNum;
        usedCoreNum_ = tilingData->usedCoreNum;
        ropeHeadDim_ = tilingData->ropeHeadDim;
        rotaryMode_ = tilingData->rotaryMode;
        normEps_ = tilingData->normEps;
        maxScNum_ = tilingData->maxScNum;
        rowLen_ = headDim_;          // coff=1：mm 行元素数
        stateRowLen_ = 2 * headDim_; // state 行元素数（kv|score）

        uint32_t coreIdx = GetBlockIdx();
        taskStart_ = coreIdx * tilingData->taskPerCore +
                     (coreIdx < tilingData->taskRem ? coreIdx : tilingData->taskRem);
        taskEnd_ = taskStart_ + tilingData->taskPerCore + (coreIdx < tilingData->taskRem ? 1 : 0);

        mmKvGm_.SetGlobalBuffer((__gm__ T *)mmKv);
        mmScoreGm_.SetGlobalBuffer((__gm__ T *)mmScore);
        stateGm_.SetGlobalBuffer((__gm__ float *)stateCache);
        apeGm_.SetGlobalBuffer((__gm__ float *)ape);
        normWeightGm_.SetGlobalBuffer((__gm__ T_NORM *)normWeight);
        ropeSinGm_.SetGlobalBuffer((__gm__ T_ROPE *)ropeSin);
        ropeCosGm_.SetGlobalBuffer((__gm__ T_ROPE *)ropeCos);
        sbtGm_.SetGlobalBuffer((__gm__ int32_t *)stateBlockTable);
        cmpKvOutGm_.SetGlobalBuffer((__gm__ T *)cmpKvOut);
        wsGm_.SetGlobalBuffer((__gm__ float *)workspace);

        iter_.Init(cuSeqlens, seqUsed, startPos, batchSize_, cmpRatio_);

        pipe_->InitBuffer(apeChunkBuf_, cmpRatio_ * dChunkSize_ * sizeof(float));
        pipe_->InitBuffer(kvStageBuf0_, cmpRatio_ * dChunkSize_ * sizeof(T));
        pipe_->InitBuffer(kvStageBuf1_, cmpRatio_ * dChunkSize_ * sizeof(T));
        pipe_->InitBuffer(scoreStageBuf0_, cmpRatio_ * dChunkSize_ * sizeof(T));
        pipe_->InitBuffer(scoreStageBuf1_, cmpRatio_ * dChunkSize_ * sizeof(T));
        pipe_->InitBuffer(kvRowsBuf_, cmpRatio_ * dChunkSize_ * sizeof(float));
        pipe_->InitBuffer(scoreRowsBuf_, cmpRatio_ * dChunkSize_ * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, (cmpRatio_ / 2) * dChunkSize_ * sizeof(float));
        pipe_->InitBuffer(cmpBuf_, dChunkSize_ * sizeof(float));
        pipe_->InitBuffer(gammaBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(ropeStageBuf_, 3 * ropeHeadDim_ * sizeof(float));
        pipe_->InitBuffer(gatherOffsetBuf_, ropeHeadDim_ * sizeof(int32_t));
        pipe_->InitBuffer(outRowBuf_, headDim_ * sizeof(T));

        // 事件 ID：EVENT_ID0 load→cast 自配对；EVENT_ID1 V→save/out 自配对；
        // EVENT_ID2 rows 复用自配对；EVENT_ID3 ape 加载自配对；EVENT_ID4/5 stage ping-pong
        eventMTE2V_ = static_cast<event_t>(EVENT_ID0);
        eventVMTE3_ = static_cast<event_t>(EVENT_ID1);
        eventMTE3V_ = static_cast<event_t>(EVENT_ID2);
        eventVMTE2_ = static_cast<event_t>(EVENT_ID3);
        idVToMTE2_[0] = static_cast<event_t>(EVENT_ID4);
        idVToMTE2_[1] = static_cast<event_t>(EVENT_ID5);
        evtSToV_ = static_cast<event_t>(EVENT_ID0); // S_V 方向与 MTE2_V 物理 flag 独立，仅 Init 用一次
        eventMTE3MTE2_ = static_cast<event_t>(EVENT_ID6); // MTE3_MTE2 方向独立 id 空间
    }

    // gamma 常驻 + INTERLEAVE 偏移表（启动一次性；复用 eventMTE2V_ 顺序自配对）
    __aicore__ inline void InitNormRope()
    {
        LocalTensor<float> gammaUb = gammaBuf_.Get<float>();
        if constexpr (std::is_same_v<T_NORM, float>) {
            DataCopy(gammaUb, normWeightGm_, headDim_);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
        } else {
            // bf16/fp16 输入：原始数据 copy 到 gamma fp32 buffer 后半段（T_NORM view 偏移
            // headDim_，字节 1024 起 = 后半段），一次 Cast 到前半段 fp32（源/目标不重叠）
            LocalTensor<T_NORM> gammaRawUb = gammaBuf_.Get<T_NORM>();
            DataCopy(gammaRawUb[headDim_], normWeightGm_, headDim_);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
            Cast(gammaUb, gammaRawUb[headDim_], RoundMode::CAST_NONE, headDim_);
        }
        if (rotaryMode_ == (uint32_t)ROTARY_MODE::INTERLEAVE) {
            LocalTensor<int32_t> gatherOffsetUb = gatherOffsetBuf_.Get<int32_t>();
            SetGatherSrcOffset<float>(gatherOffsetUb, ropeHeadDim_, evtSToV_);
        }
    }

    __aicore__ inline void Process()
    {
        InitNormRope();
        // ── 阶段一：d 分块压缩（无任务的核也必须走到 SyncAll 判断点）──
        if (taskStart_ < taskEnd_) {
            Phase1Compress();
        }
        // totalSc 是纯标量推算（每核结果一致）：0 行时阶段二无工作，全核一致跳过屏障
        // （decode 非 boundary 步省 ~1-2us SyncAll 开销）
        uint32_t totalSc = iter_.Locate(0xFFFFFFFF);
        if (totalSc == 0) {
            return;
        }
        // 全核屏障：workspace 中所有压缩行 chunk 跨核可见（tiling 已 SetScheduleMode(1)）
        SyncAll<true>();
        // ── 阶段二：压缩行 RmsNorm + RoPE → cmp_kv ──
        Phase2NormRope(totalSc);
    }

private:
    __aicore__ inline void Phase1Compress()
    {
        // prime：stage ping-pong id 各 Set 一次（与前两次迭代的 wait 配对，配平末尾 drain）
        SetFlag<HardEvent::V_MTE2>(idVToMTE2_[0]);
        SetFlag<HardEvent::V_MTE2>(idVToMTE2_[1]);
        LocalTensor<float> apeChunk = apeChunkBuf_.Get<float>();
        LocalTensor<T> kvStage[2] = {kvStageBuf0_.Get<T>(), kvStageBuf1_.Get<T>()};
        LocalTensor<T> scoreStage[2] = {scoreStageBuf0_.Get<T>(), scoreStageBuf1_.Get<T>()};
        uint32_t curDc = 0xFFFFFFFF;
        uint32_t curHh = 0;
        uint32_t curNTok = 0;
        uint32_t buf = 0;
        for (uint32_t task = taskStart_; task < taskEnd_; task++) {
            // dChunk-major：task = dc * maxGroupTaskNum + groupIdx
            uint32_t dc = task / maxGroupTaskNum_;
            uint32_t gIdx = task % maxGroupTaskNum_;
            uint32_t compressedCnt = iter_.Locate(gIdx);
            if (iter_.IsEnd()) {
                continue; // 上界 padding 任务（实际组数 < 上界）
            }
            GroupInfo g;
            iter_.GetGroupInfo(g);
            if (g.nTok == 0) {
                continue;
            }
            if (dc != curDc || g.headHolder != curHh || g.nTok != curNTok) {
                // ape 只载本组需要的行 [hh, hh+nTok)（连续）——decode 非 boundary 步
                // 从整 chunk 32KB 降到 nTok×256B；prefill 全组 (0,128) 缓存键稳定，
                // 同核同 dc 连续任务仍只载一次
                SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
                WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
                DataCopyAlignGmToUb(apeChunk, apeGm_[(uint64_t)g.headHolder * headDim_ + dc * dChunkSize_],
                                    g.nTok, dChunkSize_, headDim_, dChunkSize_);
                SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
                curDc = dc;
                curHh = g.headHolder;
                curNTok = g.nTok;
            }
            // 等 stage[buf] 可写（cast(任务 i-2) 完成；前两次迭代由 prime 配对）
            WaitFlag<HardEvent::V_MTE2>(idVToMTE2_[buf]);
            uint64_t srcOff = (uint64_t)g.xRow * rowLen_ + dc * dChunkSize_;
            DataCopyAlignGmToUb(kvStage[buf], mmKvGm_[srcOff], g.nTok, dChunkSize_, rowLen_, dChunkSize_);
            DataCopyAlignGmToUb(scoreStage[buf], mmScoreGm_[srcOff], g.nTok, dChunkSize_, rowLen_, dChunkSize_);
            ProcessTask(g, dc, compressedCnt, apeChunk, kvStage[buf], scoreStage[buf]);
            SetFlag<HardEvent::V_MTE2>(idVToMTE2_[buf]); // stage[buf] 可再写
            buf = buf ^ 1;
        }
        // drain：与 prime 配对（全局 Set:Wait 严格 1:1）
        WaitFlag<HardEvent::V_MTE2>(idVToMTE2_[0]);
        WaitFlag<HardEvent::V_MTE2>(idVToMTE2_[1]);
    }

    // 阶段二：压缩行按核均分，逐行 workspace→RmsNorm→RoPE→CAST_RINT→cmp_kv
    // 行数 ≤ maxScNum（128:1 压缩后很小），用相邻自配对串行即可，无需流水
    __aicore__ inline void Phase2NormRope(uint32_t totalSc)
    {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t rowStart = coreIdx * (totalSc / usedCoreNum_) +
                            (coreIdx < totalSc % usedCoreNum_ ? coreIdx : totalSc % usedCoreNum_);
        uint32_t rowEnd = rowStart + totalSc / usedCoreNum_ + (coreIdx < totalSc % usedCoreNum_ ? 1 : 0);
        LocalTensor<float> wsRowUb = kvRowsBuf_.Get<float>();      // 复用阶段一窗口 buffer（32KB ≥ 2KB）
        LocalTensor<float> tmpUb = scoreRowsBuf_.Get<float>();     // RmsNorm 需 (512+1)*4B
        LocalTensor<float> gammaUb = gammaBuf_.Get<float>();
        LocalTensor<float> cosUb = ropeStageBuf_.Get<float>();
        LocalTensor<float> sinUb = cosUb[ropeHeadDim_];
        LocalTensor<uint32_t> gatherOff = gatherOffsetBuf_.Get<uint32_t>();
        LocalTensor<T> outRow = outRowBuf_.Get<T>();
        RmsNormParam normParams{1.0f / (float)(int32_t)headDim_, normEps_, 1, headDim_};
        uint64_t baseAddr = headDim_ - ropeHeadDim_;
        for (uint32_t row = rowStart; row < rowEnd; row++) {
            // 等上一轮 V 完成（wsRowUb/ropeStage 可被 MTE2 重写）
            SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
            DataCopy(wsRowUb, wsGm_[(uint64_t)row * headDim_], headDim_);
            uint64_t ropeOff = (uint64_t)row * ropeHeadDim_;
            if constexpr (std::is_same_v<T_ROPE, float>) {
                DataCopy(cosUb, ropeCosGm_[ropeOff], ropeHeadDim_);
                DataCopy(sinUb, ropeSinGm_[ropeOff], ropeHeadDim_);
            } else {
                // bf16/fp16 输入：cos/sin 原始数据 copy 到 buffer 后两块（T_ROPE view 偏移
                // 4/5 * ropeHeadDim_），各一次 Cast 到前两块 fp32（源/目标不重叠）
                LocalTensor<T_ROPE> ropeRawUb = ropeStageBuf_.Get<T_ROPE>();
                DataCopy(ropeRawUb[4 * ropeHeadDim_], ropeCosGm_[ropeOff], ropeHeadDim_);
                DataCopy(ropeRawUb[5 * ropeHeadDim_], ropeSinGm_[ropeOff], ropeHeadDim_);
            }
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
            if constexpr (!std::is_same_v<T_ROPE, float>) {
                LocalTensor<T_ROPE> ropeRawUb = ropeStageBuf_.Get<T_ROPE>();
                Cast(cosUb, ropeRawUb[4 * ropeHeadDim_], RoundMode::CAST_NONE, ropeHeadDim_);
                Cast(sinUb, ropeRawUb[5 * ropeHeadDim_], RoundMode::CAST_NONE, ropeHeadDim_);
            }
            RmsNorm(wsRowUb, wsRowUb, gammaUb, tmpUb, normParams);
            PipeBarrier<PIPE_V>();
            if (rotaryMode_ == (uint32_t)ROTARY_MODE::INTERLEAVE) {
                RotaryPosEmb<ROTARY_MODE::INTERLEAVE>(wsRowUb, wsRowUb, cosUb, sinUb, tmpUb, gatherOff, 1,
                                                      ropeHeadDim_, headDim_, baseAddr);
            } else {
                RotaryPosEmb<ROTARY_MODE::HALF>(wsRowUb, wsRowUb, cosUb, sinUb, tmpUb, gatherOff, 1,
                                                ropeHeadDim_, headDim_, baseAddr);
            }
            PipeBarrier<PIPE_V>();
            // 等上一轮输出写（MTE3 读 outRow）完成再 cast 重写
            SetFlag<HardEvent::MTE3_V>(eventMTE3V_);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V_);
            Cast(outRow, wsRowUb, RoundMode::CAST_RINT, headDim_);
            SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
            DataCopy(cmpKvOutGm_[(uint64_t)row * headDim_], outRow, headDim_);
        }
    }

    static constexpr uint32_t NORM_BATCH = 8; // 阶段二批量行数（摊销 barrier；受 tmp/outRow buffer 限）

    __aicore__ inline void ProcessTask(const GroupInfo &g, uint32_t dc, uint32_t compressedCnt,
                                       const LocalTensor<float> &apeChunk, const LocalTensor<T> &kvStage,
                                       const LocalTensor<T> &scoreStage)
    {
        LocalTensor<float> kvRows = kvRowsBuf_.Get<float>();
        LocalTensor<float> scoreRows = scoreRowsBuf_.Get<float>();
        uint32_t dStart = dc * dChunkSize_;
        uint32_t nTok = g.nTok;
        uint32_t hh = g.headHolder;

        // ── 自同步：上一任务 state 写（MTE3 读 rows）完成（rows 单缓冲复用）──
        SetFlag<HardEvent::MTE3_V>(eventMTE3V_);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V_);

        // ── 1. 等本组 mm load（MTE2）完成 ──
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

        // ── 2. cast fp32 → 窗口行 [hh, hh+nTok)；score += ape ──
        // （ape 按行切片缓存在 apeChunk[0..nTok)，与本组窗口行对齐）
        Cast(kvRows[hh * dChunkSize_], kvStage, RoundMode::CAST_NONE, nTok * dChunkSize_);
        Cast(scoreRows[hh * dChunkSize_], scoreStage, RoundMode::CAST_NONE, nTok * dChunkSize_);
        PipeBarrier<PIPE_V>();
        Add(scoreRows[hh * dChunkSize_], scoreRows[hh * dChunkSize_], apeChunk, nTok * dChunkSize_);
        PipeBarrier<PIPE_V>();

        // ── 3. 写 state（该 d 段）──
        SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
        SaveStateChunk(kvRows[hh * dChunkSize_], g, 0, dStart, nTok);
        SaveStateChunk(scoreRows[hh * dChunkSize_], g, 1, dStart, nTok);
        // state 写（MTE3 读 rows）完成后才能原地 softmax/Mul（V 写 rows）
        SetFlag<HardEvent::MTE3_V>(eventMTE3V_);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V_);

        // ── 4. 压缩：首组 headHolder 历史行从 state 读入窗口前部 ──
        if (g.produce) {
            // 跨 pipe 守卫：此前任务的 SaveState（MTE3 读 rows[0..nTok)）可能未完成，
            // 本任务 ReadStateChunk（MTE2 写 rows[0..hh)）与之区域可重叠（如前任务 hh=0、
            // 本任务 hh=120 时 rows[0..2)∩rows[0..120)）——id4/5 只 gate MTE2 等 V，
            // MTE3→MTE2 必须显式守卫（潜伏竞态，S=8 跨组尾 case 实测抓到）
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            if (hh > 0) {
                ReadStateChunk(kvRows, hh, g.gStart, 0, dStart, g.bIdx);
                ReadStateChunk(scoreRows, hh, g.gStart, 1, dStart, g.bIdx);
            }
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
            LocalTensor<float> tmpUb = tmpBuf_.Get<float>();
            LocalTensor<float> cmpRow = cmpBuf_.Get<float>();
            ColumnSoftMax(scoreRows, scoreRows, tmpUb, cmpRatio_, dChunkSize_);
            PipeBarrier<PIPE_V>();
            Mul(kvRows, kvRows, scoreRows, cmpRatio_ * dChunkSize_);
            PipeBarrier<PIPE_V>();
            ColumnSum(cmpRow, kvRows, tmpUb, cmpRatio_, dChunkSize_);
            PipeBarrier<PIPE_V>();
            // 压缩 chunk fp32 写 workspace（未 norm/rope）；cast 到 bf16 在阶段二整行完成后做
            SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
            DataCopy(wsGm_[(uint64_t)compressedCnt * headDim_ + dStart], cmpRow, dChunkSize_);
        }
    }

    // 从分页 state 读历史行段（fp32，dChunk 列）
    __aicore__ inline void ReadStateChunk(const LocalTensor<float> &dst, uint32_t rowCnt, uint32_t posStart,
                                          uint32_t stateIdx, uint32_t dStart, uint32_t bIdx)
    {
        uint32_t p = posStart;
        uint32_t done = 0;
        while (done < rowCnt) {
            uint32_t blk = p / blockSize_;
            uint32_t rowInBlk = p % blockSize_;
            int32_t blockId = sbtGm_.GetValue(bIdx * maxBlockNumPerBatch_ + blk);
            uint32_t seg = blockSize_ - rowInBlk;
            if (done + seg > rowCnt) {
                seg = rowCnt - done;
            }
            uint64_t srcOff =
                (uint64_t)blockId * stateStride0_ + rowInBlk * stateRowLen_ + stateIdx * rowLen_ + dStart;
            DataCopyAlignGmToUb(dst[done * dChunkSize_], stateGm_[srcOff], seg, dChunkSize_, stateRowLen_,
                                dChunkSize_);
            done += seg;
            p += seg;
        }
    }

    // 写本组 token 的 d 段到分页 state（fp32；blockId==0 跳过）
    __aicore__ inline void SaveStateChunk(const LocalTensor<float> &src, const GroupInfo &g, uint32_t stateIdx,
                                          uint32_t dStart, uint32_t nTok)
    {
        uint32_t p = g.tokStart;
        uint32_t done = 0;
        while (done < nTok) {
            uint32_t blk = p / blockSize_;
            uint32_t rowInBlk = p % blockSize_;
            int32_t blockId = sbtGm_.GetValue(g.bIdx * maxBlockNumPerBatch_ + blk);
            uint32_t seg = blockSize_ - rowInBlk;
            if (done + seg > nTok) {
                seg = nTok - done;
            }
            if (blockId != 0) {
                uint64_t dstOff =
                    (uint64_t)blockId * stateStride0_ + rowInBlk * stateRowLen_ + stateIdx * rowLen_ + dStart;
                DataCopyAlignUbToGm(stateGm_[dstOff], src[done * dChunkSize_], seg, dChunkSize_, dChunkSize_,
                                    stateRowLen_);
            }
            done += seg;
            p += seg;
        }
    }

    TPipe *pipe_ = nullptr;
    GroupIterator iter_;

    GlobalTensor<T> mmKvGm_;
    GlobalTensor<T> mmScoreGm_;
    GlobalTensor<float> stateGm_;
    GlobalTensor<float> apeGm_;
    GlobalTensor<T_NORM> normWeightGm_;
    GlobalTensor<T_ROPE> ropeSinGm_;
    GlobalTensor<T_ROPE> ropeCosGm_;
    GlobalTensor<int32_t> sbtGm_;
    GlobalTensor<T> cmpKvOutGm_;
    GlobalTensor<float> wsGm_;

    TBuf<TPosition::VECCALC> apeChunkBuf_;
    TBuf<TPosition::VECCALC> kvStageBuf0_;
    TBuf<TPosition::VECCALC> kvStageBuf1_;
    TBuf<TPosition::VECCALC> scoreStageBuf0_;
    TBuf<TPosition::VECCALC> scoreStageBuf1_;
    TBuf<TPosition::VECCALC> kvRowsBuf_;
    TBuf<TPosition::VECCALC> scoreRowsBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> cmpBuf_;
    TBuf<TPosition::VECCALC> gammaBuf_;
    TBuf<TPosition::VECCALC> ropeStageBuf_;
    TBuf<TPosition::VECCALC> gatherOffsetBuf_;
    TBuf<TPosition::VECCALC> outRowBuf_;

    event_t eventMTE2V_;
    event_t eventVMTE3_;
    event_t eventMTE3V_;
    event_t eventVMTE2_;
    event_t idVToMTE2_[2];
    event_t evtSToV_;
    event_t eventMTE3MTE2_;

    uint32_t batchSize_ = 0;
    uint32_t headDim_ = 0;
    uint32_t cmpRatio_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlockNumPerBatch_ = 0;
    uint64_t stateStride0_ = 0;
    uint32_t dChunkSize_ = 0;
    uint32_t maxGroupTaskNum_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t rowLen_ = 0;
    uint32_t stateRowLen_ = 0;
    uint32_t taskStart_ = 0;
    uint32_t taskEnd_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t rotaryMode_ = 0;
    float normEps_ = 1e-6f;
    uint32_t maxScNum_ = 0;
};

} // namespace CompressorEpilogue

#endif // COMPRESSOR_EPILOGUE_KERNEL_C128_H
