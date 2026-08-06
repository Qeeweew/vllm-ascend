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
 * \file compressor_epilogue_comm.h
 * \brief
 */

#ifndef COMPRESSOR_EPILOGUE_COMM_H
#define COMPRESSOR_EPILOGUE_COMM_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

namespace CompressorEpilogue {
template <typename T>
__aicore__ inline T CeilDivT(T num1, T num2)
{
    if (num2 == 0) {
        return static_cast<T>(0);
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T>
__aicore__ inline T Trunc(T num, T rnd)
{
    return ((rnd) == 0) ? 0 : (((num) / (rnd) * (rnd)));
}



enum class X_LAYOUT : std::uint8_t {
    BSH = static_cast<std::uint8_t>(0),
    TH = static_cast<std::uint8_t>(1)
};

enum class X_DTYPE : std::uint8_t {
    BF16 = static_cast<std::uint8_t>(0),
    FP16 = static_cast<std::uint8_t>(1)
};

enum class ROPE_DTYPE : std::uint8_t {
    SAME_AS_X = static_cast<std::uint8_t>(0),
    FP32 = static_cast<std::uint8_t>(1)
};

enum class COFF : std::uint8_t {
    DISABLE = static_cast<std::uint8_t>(1),
    OVERLAP = static_cast<std::uint8_t>(2)
};

enum class ROTARY_MODE : std::uint8_t {
    HALF = static_cast<std::uint8_t>(1),
    INTERLEAVE = static_cast<std::uint8_t>(2)
};

enum class CACHE_MODE : std::uint8_t {
    CONTINUOUS = static_cast<std::uint8_t>(1),
    CYCLE = static_cast<std::uint8_t>(2)
};

enum class TEMPLATE_ID : uint8_t {
    NORMAL = 0,
    EMPTY_X = 1,
    PERF = 2
};

template <X_LAYOUT X_L, X_DTYPE X_T, ROPE_DTYPE R_T, COFF C, ROTARY_MODE Rotary_Mode, typename... Args>
struct COMPType {
    static constexpr X_LAYOUT xLayout = X_L;
    static constexpr X_DTYPE xDtype = X_T;
    static constexpr ROPE_DTYPE ropeDtype = R_T;
    static constexpr COFF coff = C;
    static constexpr ROTARY_MODE rotaryMode = Rotary_Mode;
};

struct ConstInfo {
    // 分核相关（行并行：每核独占完整 D 维）
    uint32_t usedCoreNum = 0;
    uint32_t mBaseSize = 0;
    uint32_t dBasicBlockNum = 1;
    uint32_t coreGroupNum = 0;
    uint32_t curGroupIdx = 0;
    uint32_t aiCoreIdx = 0;

    // shape及参数
    uint32_t batchSize = 0;
    uint32_t sSize = 0;   // BSH 布局下 batch 内序列长度（GetSeqLength/GetScSize 使用）
    uint32_t headDim = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t cmpRatio = 0;
    float normEps = 1e-6;
    float reciprocalD = 0;

    // pageAttention
    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t stateCacheStrideDim0 = 0;
};
struct Vec1RunInfo {
    // 当前基本块要处理的 token 区间（batch 起始 + 起点 sIdx）
    uint32_t bStart = 0;
    uint32_t sStart = 0;
    uint32_t dealTcNum = 0;
    uint32_t dealScSize = 0;
};



// BUFFER的字节数
inline constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
inline constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
inline constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
inline constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
inline constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
inline constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
inline constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
inline constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
inline constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
inline constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
inline constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;

// BLOCK和REPEAT的字节数
inline constexpr uint64_t BYTE_BLOCK = 32UL;
inline constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// BLOCK和REPEAT的FP32元素数
inline constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float); // 8
inline constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float); // 64
inline constexpr uint32_t REPEAT_STRIDE_NUM = REPEAT_BLOCK_BYTE / BYTE_BLOCK; // 8
inline constexpr uint32_t REPEAT_MAX_NUM = 255;
inline constexpr uint32_t BRCB_NUM = 8;
inline constexpr uint32_t MAX_R = 256;

template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = nValue; // nd矩阵的行数
    if constexpr (IsSameType<T, int4b_t>::value) {
        constexpr uint32_t HALF_SIZE_DIVISOR = 2;
        nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
        nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
    } else {
        nd2nzPara.dValue = dValue; // nd矩阵的列数
        nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
    }
    nd2nzPara.dstNzC0Stride = dstNzC0Stride;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}
template <typename T>
__aicore__ inline void DumpTensorForDim2(GlobalTensor<T> tensor, uint32_t desc, uint32_t dumpSize, uint32_t row, uint32_t col)
{
    uint32_t array2[] = {static_cast<uint32_t>(row), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

template <typename T>
__aicore__ inline void DumpTensorForDim2(LocalTensor<T> tensor, uint32_t desc, uint32_t dumpSize, uint32_t row, uint32_t col)
{
    uint32_t array2[] = {static_cast<uint32_t>(row), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

template <typename T>
__aicore__ inline void DumpTensorForDim2(LocalTensor<T> tensor, uint32_t desc, uint32_t dumpSize)
{
    uint32_t col = 32 / sizeof(T);
    uint32_t array2[] = {static_cast<uint32_t>(dumpSize / col), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

template <typename T>
__aicore__ inline void DumpTensorForDim2(GlobalTensor<T> tensor, uint32_t desc, uint32_t dumpSize)
{
    uint32_t col = 32 / sizeof(T);
    uint32_t array2[] = {static_cast<uint32_t>(dumpSize / col), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

} // namespace CompressorEpilogue
#endif
