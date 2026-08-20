/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
 *
 * arch22（910B）两阶段重构（源自 ops-transformer compressor-epilogue 分支）：
 *   - C4（coff=2, cmpRatio=4）：组流式 + 双缓冲流水，UB 内融合 RmsNorm/RoPE/cast（无 SyncAll）
 *   - C128（coff=1, cmpRatio=128）：d 分块压缩 → GM workspace → SyncAll → 完整行 RmsNorm/RoPE/cast
 * 分发用 TILING_KEY_IS 运行时分发（rms_norm_dynamic_quant 风格）：key 1=C4, 2=C128；
 * dtype 由编译变体宏 DTYPE_MM_KV（编译器按 mm_kv 输入 dtype 自动生成）决定。
 */

#if (__CCE_AICORE__ == 220)
#include "arch32/compressor_epilogue_kernel_c4.h"
#include "arch32/compressor_epilogue_kernel_c128.h"
#else
#error "compressor_epilogue currently only supports arch22 (Ascend910B)"
#endif

using namespace CompressorEpilogue;

extern "C" __global__ __aicore__ void compressor_epilogue(
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
    // ref（ops-transformer compressor-epilogue 分支）语义：C4 无 SyncAll 用 AIV_ONLY，C128 两阶段
    // SyncAll 需 MIX_AIV_1_0（全核同调度）。本 CANN 下 KERNEL_TASK_TYPE(key,..) 无法按 ASCENDC_TPL
    // 编码 key 区分，统一用 MIX_AIV_1_0 默认（与 v2 旧代码一致，AIV 执行，taskRation 0:1）。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorEpilogueTilingData, tilingDataIn, tiling);
    const optiling::CompressorEpilogueTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    // 编译变体宏：输入 dtype（bf16 → bfloat16_t，fp16 → half，fp32 → float），由编译器自动注入
    using X_T = DTYPE_MM_KV;
    using NORM_T = DTYPE_NORM_WEIGHT;
    using ROPE_T = DTYPE_ROPE_SIN;
    if (TILING_KEY_IS(1)) {
        // C4：coff=2（overlap），无需 workspace
        CompressorEpilogueKernelC4<X_T, NORM_T, ROPE_T> op;
        op.Init(&pipe, tilingData, mmKv, mmScore, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,
                cuSeqlens, seqUsed, startPos, cmpKvOut);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        // C128：coff=1，c128 压缩行（未 norm/rope fp32）经用户 workspace 中转
        __gm__ uint8_t *userWs = GetUserWorkspace(workspace);
        CompressorEpilogueKernelC128<X_T, NORM_T, ROPE_T> op;
        op.Init(&pipe, tilingData, mmKv, mmScore, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,
                cuSeqlens, seqUsed, startPos, cmpKvOut, userWs);
        op.Process();
    }
}
