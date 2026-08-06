/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"); you may not use this file except in compliance with the License.
 * Please refer to the License for details. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file compressor_epilogue.cpp
 * \brief Compressor 拆分版 epilogue：GEMM（x @ wkv / x @ wgate）由外部 MatMulV3 完成，
 *        本算子只做 ape + softmax gate + 加权压缩 + state 递归 + rms_norm + rope。
 *        输入 mm_kv / mm_score 为 [tokenSize, coff*headDim] 的 bf16/fp16 GEMM 结果。
 */

#if (__CCE_AICORE__ == 220)
#include "arch32/compressor_epilogue_kernel_perf.h"
#else
#error "compressor_epilogue currently only supports arch32 (Ascend910B)"
#endif

using namespace CompressorEpilogue;

#define INVOKE_COMPRESSOR_EPILOGUE_OP_IMPL(templateClass, ...)                                                     \
    do {                                                                                                         \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData);                                              \
        op.Init(mmKv, mmScore, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,                   \
                cuSeqlens, seqUsed, startPos, cmpKvOut, workspace);                                              \
        op.Process();                                                                                            \
    } while (0)

template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId, uint8_t RopeDType>
__global__ __aicore__ void compressor_epilogue(
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
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *stateCacheOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    REGISTER_TILING_DEFAULT(optiling::CompressorEpilogueTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);  // AIV-only 但保留 SyncAll 能力（vec1->vec2 需要）
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorEpilogueTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }
    const optiling::CompressorEpilogueTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    constexpr auto xLayout = static_cast<X_LAYOUT>(XLayout);
    constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
    constexpr auto ropeDtype = static_cast<ROPE_DTYPE>(RopeDType);
    constexpr auto coff = static_cast<COFF>(Coff);
    constexpr auto rotaryMode = static_cast<ROTARY_MODE>(RotaryMode);
    INVOKE_COMPRESSOR_EPILOGUE_OP_IMPL(CompressorEpilogueKernelPerf, xLayout, xDtype, ropeDtype, coff, rotaryMode);
}
