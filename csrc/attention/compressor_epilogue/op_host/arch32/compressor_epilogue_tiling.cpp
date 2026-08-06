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
* \file compressor_epilogue_tiling.cpp
* \file compressor_epilogue_tiling.cpp
* \brief
*/

#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "compressor_epilogue_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {



void CompressorEpilogueTiling::ConvertRequiredParams(gert::TilingContext &context, CompressorEpilogueContext &compressor_epilogueContext)
{
    compressor_epilogueContext.mmKv.desc = context.GetRequiredInputDesc(MM_KV_INPUT_INDEX);
    compressor_epilogueContext.mmKv.shape = context.GetRequiredInputShape(MM_KV_INPUT_INDEX);
    compressor_epilogueContext.mmScore.desc = context.GetRequiredInputDesc(MM_SCORE_INPUT_INDEX);
    compressor_epilogueContext.mmScore.shape = context.GetRequiredInputShape(MM_SCORE_INPUT_INDEX);
    compressor_epilogueContext.stateCache.desc = context.GetRequiredInputDesc(STATE_CACHE_INPUT_INDEX);
    compressor_epilogueContext.stateCache.shape = context.GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    compressor_epilogueContext.ape.desc = context.GetRequiredInputDesc(APE_INPUT_INDEX);
    compressor_epilogueContext.ape.shape = context.GetRequiredInputShape(APE_INPUT_INDEX);
    compressor_epilogueContext.normWeight.desc = context.GetRequiredInputDesc(NORM_WEIGHT_INPUT_INDEX);
    compressor_epilogueContext.normWeight.shape = context.GetRequiredInputShape(NORM_WEIGHT_INPUT_INDEX);
    compressor_epilogueContext.ropeSin.desc = context.GetRequiredInputDesc(ROPE_SIN_INPUT_INDEX);
    compressor_epilogueContext.ropeSin.shape = context.GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);
    compressor_epilogueContext.ropeCos.desc = context.GetRequiredInputDesc(ROPE_COS_INPUT_INDEX);
    compressor_epilogueContext.ropeCos.shape = context.GetRequiredInputShape(ROPE_COS_INPUT_INDEX);

    compressor_epilogueContext.cmpKv.desc = context.GetOutputDesc(CMP_KV_OUTPUT_INDEX);
    compressor_epilogueContext.cmpKv.shape = context.GetOutputShape(CMP_KV_OUTPUT_INDEX);

    compressor_epilogueContext.dtype = compressor_epilogueContext.mmKv.desc->GetDataType();
    auto xDimNum = compressor_epilogueContext.mmKv.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_EPILOGUE_DIM_NUM_3) {
        compressor_epilogueContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_EPILOGUE_DIM_NUM_2) {
        compressor_epilogueContext.layout = LayoutType::LAYOUT_TH;
    }
}

void CompressorEpilogueTiling::ConvertOptionalParams(gert::TilingContext &context, CompressorEpilogueContext &compressor_epilogueContext)
{
    compressor_epilogueContext.stateBlockTable.desc = context.GetOptionalInputDesc(STATE_BLOCK_TABLE_INPUT_INDEX);
    compressor_epilogueContext.stateBlockTable.shape = context.GetOptionalInputShape(STATE_BLOCK_TABLE_INPUT_INDEX);
    compressor_epilogueContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressor_epilogueContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressor_epilogueContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressor_epilogueContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressor_epilogueContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressor_epilogueContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorEpilogueTiling::ConvertContext(gert::TilingContext &context, CompressorEpilogueContext &compressor_epilogueContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("CompressorEpilogue", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    compressor_epilogueContext.opName = context.GetNodeName();
    compressor_epilogueContext.opType = context.GetNodeType();
    compressor_epilogueContext.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, compressor_epilogueContext);
    ConvertOptionalParams(context, compressor_epilogueContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    compressor_epilogueContext.ropeHeadDim = attrs->GetAttrPointer<int>(ROPE_HEAD_DIM_ATTR_INDEX);
    compressor_epilogueContext.coff = attrs->GetAttrPointer<int>(COFF_ATTR_INDEX);
    compressor_epilogueContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    compressor_epilogueContext.normEps = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_INDEX);
    compressor_epilogueContext.rotaryMode = attrs->GetAttrPointer<int>(ROTARY_MODE_ATTR_INDEX);
    compressor_epilogueContext.cacheMode = attrs->GetAttrPointer<int>(CACHE_MODE_ATTR_INDEX);
    compressor_epilogueContext.stateCacheStrideDim0 = attrs->GetAttrPointer<int>(STATE_CACHE_STRIDE_DIM0_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    compressor_epilogueContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::SetBaseInfo()
{
    if (context_->mmKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_EPILOGUE_DIM_NUM_3) {
        baseParams_->batchSize = context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0);
        baseParams_->seqSize = context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
        baseParams_->cgSize = context_->ropeSin.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_1);
    } else {
        baseParams_->batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_1);
        baseParams_->cgSize = context_->ropeSin.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0);
    }

    baseParams_->headDim = context_->normWeight.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0);
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->csSize = baseParams_->seqSize - (baseParams_->seqSize % baseParams_->cmpRatio);
    baseParams_->ropeHeadDim = static_cast<uint32_t>(*context_->ropeHeadDim);
    baseParams_->normEps = static_cast<float>(*context_->normEps);
    baseParams_->reciprocalD = 1.0 / baseParams_->headDim;
    baseParams_->cgSize =
        (baseParams_->seqSize + baseParams_->cmpRatio - 1) / baseParams_->cmpRatio; // number of token after compress
    baseParams_->stateCacheStrideDim0 = static_cast<uint64_t>(*context_->stateCacheStrideDim0);
    coff = static_cast<uint8_t>(*context_->coff);

    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u", baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_1);
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::CONTINUOUS)) {
        pageAttentionParams_->maxBlockNumPerBatch =
            context_->stateBlockTable.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::SetWorkSpaceInfo()
{
    // mm 结果由外部 MatMulV3 产出；完全串行化后无 vec1Res workspace / SyncAll，workspace 全部置 0
    workspaceParams_->dbWorkspaceRatio = 1;
    workspaceParams_->mm1KvResSize = 0;
    workspaceParams_->mm1ScoreResSize = 0;
    workspaceParams_->vec1TailCacheSize = 0;
    workspaceParams_->vec1ResSize = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::SetScenarioInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::SetTemplateId()
{
    if (context_->templateId == TemplateId::EMPTY_X) {
        return ge::GRAPH_SUCCESS;
    }
    // 设置高性能模板
    context_->templateId = TemplateId::PERF;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::SetInnerSplitInfo()
{
    // 行并行：每核独占完整 D 维，kernel 不再读 innerSplitParams.dBaseSize（仅保留 tiling 布局）
    if (context_->templateId == TemplateId::PERF) {
        innerSplitParams_->mBaseSize = (coff == 2) ? 128 : 256;
    } else {
        innerSplitParams_->mBaseSize = 256; // 256:核间切分，M轴基本块大小
    }
    // a5 由于loc更大, mBaseSize x 2
    // if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
    //      innerSplitParams_->mBaseSize *= 2;
    //  }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CalcWorkSpace()
{
    // 完全串行化后 kernel 不需要 workspace（无 vec1Res 中转），仅保留 libapi 基础大小
    workspaceSize_ = libapiSize_;

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckEmptyTensor() const
{
    if (context_->layout == LayoutType::LAYOUT_BSH && context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0) == 0 ||
        context_->layout == LayoutType::LAYOUT_BSH && context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_1) == 0 ||
        context_->layout == LayoutType::LAYOUT_TH && context_->mmKv.shape->GetStorageShape().GetDim(COMPRESSOR_EPILOGUE_DIM_INDEX_0) == 0) {
        context_->templateId = TemplateId::EMPTY_X;
    } else {
        if (context_->mmKv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->mmScore.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateCache.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ape.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->normWeight.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ropeSin.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ropeCos.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateBlockTable.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE(context_->opName, "Only input tensor x dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::RunBigKernelTiling(CompressorEpilogueTilingData* tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs {
        std::bind(&CompressorEpilogueTiling::GetNpuInfo, this),
        std::bind(&CompressorEpilogueTiling::CheckRequiredParaExistence, this),
        std::bind(&CompressorEpilogueTiling::CheckEmptyTensor, this),
        std::bind(&CompressorEpilogueTiling::CheckSinglePara, this),
        std::bind(&CompressorEpilogueTiling::SetBaseInfo, this),
        std::bind(&CompressorEpilogueTiling::SetPageAttentionInfo, this),
        std::bind(&CompressorEpilogueTiling::CheckFeature, this),
        std::bind(&CompressorEpilogueTiling::CheckMultiParaConsistency, this),
        std::bind(&CompressorEpilogueTiling::CheckBlockDimConstrain, this),
        std::bind(&CompressorEpilogueTiling::SetTemplateId, this),
        std::bind(&CompressorEpilogueTiling::SetInnerSplitInfo, this),
        std::bind(&CompressorEpilogueTiling::SetWorkSpaceInfo, this),
        std::bind(&CompressorEpilogueTiling::SetScenarioInfo, this)
    };
    for (const auto &func: requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->templateId == TemplateId::EMPTY_X) {
        workspaceSize_ = libapiSize_;
        if (context_->workSpaces) {
            context_->workSpaces[0] = workspaceSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }
    std::vector<StatusFunction> optionalTilingFuncs {
        std::bind(&CompressorEpilogueTiling::CalcWorkSpace, this),
        std::bind(&CompressorEpilogueTiling::GenTilingKey, this)
    };
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // KERNEL_TYPE_MIX_AIV_1_0：blockDim = AIV 核数（40）；usedCoreNum 保持原 MIX 语义（AIC 核数），
    // vec 代码内部 aiCoreNum = usedCoreNum * 2 = 40 与物理 AIV 数一致
    baseParams_->usedCoreNum = aivNum_ / 2;

    context_->blockDim = aivNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::GenTilingKey() const
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;
    uint8_t ropeDtype = 0;
    uint8_t rotaryMode = static_cast<uint8_t>(*context_->rotaryMode);
    uint8_t templateId = static_cast<uint8_t>(context_->templateId);
    uint8_t cacheMode = static_cast<uint8_t>(*context_->cacheMode);

    auto xDtype = context_->mmKv.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto ropeSinDtype = context_->ropeSin.desc->GetDataType();
    auto ropeCosDtype = context_->ropeCos.desc->GetDataType();
    bool supportFp32Rope = socVersion_ == platform_ascendc::SocVersion::ASCEND910B ||
                           socVersion_ == platform_ascendc::SocVersion::ASCEND910_93;
    if (ropeSinDtype == ge::DT_FLOAT && ropeCosDtype == ge::DT_FLOAT && supportFp32Rope) {
        ropeDtype = 1;
    }
    auto xDimNum = context_->mmKv.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_EPILOGUE_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    context_->tilingKey = GET_TPL_TILING_KEY(
        layout,
        dtype,
        coff,
        rotaryMode,
        cacheMode,
        templateId,
        ropeDtype
    );
    OP_LOGI(context_->opName,
            "CompressorEpilogue dtype:%hhu layout:%hhu  coff:%hhu rotary_mode:%hhu, cacheMode: %u, template_id:%hhu, rope_dtype:%hhu",
            dtype, layout, coff, rotaryMode, cacheMode, templateId, ropeDtype);
    OP_LOGI(context_->opName, "CompressorEpilogue tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaMmKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaMmScore() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStateCache() ||
        ge::GRAPH_SUCCESS != CheckSingleParaApe() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNormWeight() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeSin() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeCos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStateBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeHeadDim() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNormEps() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRotaryMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCacheMode()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorEpilogueTiling::CheckFeatureValueSupport(const T *featureValue,
    const std::vector<T> &expectFeatureValList, const std::string &name) const
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) == expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorEpilogueTiling::CheckAttrValueSupport(const T *attrValue,
    const std::vector<T> &expectAttrValList, const std::string &name) const
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
std::string to_string(const T &value) {
    if (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

template <typename T>
void CompressorEpilogueTiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE(context_->opName, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), to_string(actualValue).c_str());
}

std::string LayoutTypeToStrEpilogue(LayoutType layout)
{
    switch (layout) {
        case LayoutType::LAYOUT_BSH:
            return "BSH";
        case LayoutType::LAYOUT_TH:
            return "TH";
        default:
            return "UNKNOWN_LAYOUT";
    }
}

ge::graphStatus CompressorEpilogueTiling::CheckDimNumInLayoutSupport(const std::string &layout, const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    const auto& dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE(context_->opName, "When layout is %s, %s dimension should be %zu, but it's %zu",
            layout.c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                   const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE(context_->opName, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) ==
                        expectDtypeList.end(),
                    LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorEpilogueTiling::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                            const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(context_->opName, "Tensor %s only supports dtype %s, but got %s", name.c_str(), oss.str().c_str(),
            DataTypeToSerialString(actualDtype).c_str());
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("CompressorEpilogue", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

ge::graphStatus CompressorEpilogueTiling::CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE(context_->opName, "%s dim number support list should be specify in DIM_NUM_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaMmKv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->mmKv.desc, MM_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->mmKv.shape, MM_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStrEpilogue(context_->layout), context_->mmKv.shape, MM_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaMmScore() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->mmScore.desc, MM_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->mmScore.shape, MM_SCORE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaStateCache() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateCache.desc, STATE_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateCache.shape, STATE_CACHE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaApe() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ape.desc, APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ape.shape, APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaNormWeight() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->normWeight.desc, NORM_WEIGHT_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->normWeight.shape, NORM_WEIGHT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaRopeSin() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ropeSin.desc, ROPE_SIN_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ropeSin.shape, ROPE_SIN_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStrEpilogue(context_->layout), context_->ropeSin.shape, ROPE_SIN_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaRopeCos() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ropeCos.desc, ROPE_COS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ropeCos.shape, ROPE_COS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStrEpilogue(context_->layout), context_->ropeCos.shape, ROPE_COS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaStateBlockTable() const
{
    if (context_->stateBlockTable.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateBlockTable.desc, STATE_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateBlockTable.shape, STATE_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cuSeqlens.shape, CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaSeqused() const
{
    if (context_->seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqUsed.shape, SEQUSED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaStartPos() const
{
    if (context_->startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->startPos.shape, START_POS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaCmpKv() const
{
    if (context_->cmpKv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cmpKv.desc, CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cmpKv.shape, CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaRopeHeadDim()const
{
    if (CheckAttrValueSupport(context_->ropeHeadDim, ROPE_HEAD_DIM, ROPE_HEAD_DIM_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaCmpRatio()const
{
    if (CheckAttrValueSupport(context_->cmpRatio, CMP_RATIO, CMP_RATIO_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaCoff()const
{
    if (CheckAttrValueSupport(context_->coff, COFF, COFF_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaNormEps()const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaRotaryMode()const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->rotaryMode, ROTARY_MODE, ROTARY_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckSingleParaCacheMode() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->cacheMode, CACHE_MODE, CACHE_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(context_->mmKv.shape == nullptr, OP_LOGE(context_->opName, "tensor x is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->mmKv.desc == nullptr, OP_LOGE(context_->opName, "tensor x is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->mmScore.shape == nullptr, OP_LOGE(context_->opName, "tensor mmScore is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->mmScore.desc == nullptr, OP_LOGE(context_->opName, "tensor mmScore is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.shape == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.desc == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.shape == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.desc == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->normWeight.shape == nullptr, OP_LOGE(context_->opName, "tensor normWeight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->normWeight.desc == nullptr, OP_LOGE(context_->opName, "tensor normWeight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeSin.shape == nullptr, OP_LOGE(context_->opName, "tensor ropeSin is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeSin.desc == nullptr, OP_LOGE(context_->opName, "tensor ropeSin is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeCos.shape == nullptr, OP_LOGE(context_->opName, "tensor ropeCos is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeCos.desc == nullptr, OP_LOGE(context_->opName, "tensor ropeCos is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.shape == nullptr,
                OP_LOGE(context_->opName, "tensor stateBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.desc == nullptr,
                OP_LOGE(context_->opName, "tensor stateBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.shape == nullptr, OP_LOGE(context_->opName, "tensor cmpKv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.desc == nullptr, OP_LOGE(context_->opName, "tensor cmpKv is nullptr"),
                return ge::GRAPH_FAILED);
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr,
        OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens should not be nullptr"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr,
        OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens should not be nullptr"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->cuSeqlens.desc != nullptr,
        OP_LOGE(context_->opName, "In BSH layout, tensor cuSeqlens must be nullptr"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape != nullptr,
        OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens must be nullptr"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->ropeHeadDim == nullptr, OP_LOGE(context_->opName, "attr ropeHeadDim is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->cmpRatio == nullptr, OP_LOGE(context_->opName, "attr cmpRatio is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_->headDim, HEAD_DIM, "headDim")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(pageAttentionParams_->blockSize < MIN_BLOCK_SIZE,
                OP_LOGE(context_->opName, "blockSize should not be less than 1, but got %u",
                        pageAttentionParams_->blockSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::LogErrorShapeConsistency(const std::string &name,
    const gert::StorageShape *shape, const uint32_t &dimNum, const std::string &subName, const uint32_t &expectNum) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE(context_->opName,
                        "%s shape dim %u, should be equal to %s: %u, but got %u",
                        name.c_str(), dimNum, subName.c_str(), expectNum, actualNum),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckShapeConsistency() const
{
    if (CheckShapeConsistencyRope() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto coffD = coff * baseParams_->headDim;
    uint32_t stateNum = 2;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateBlockTable", context_->stateBlockTable.shape,
                                                      COMPRESSOR_EPILOGUE_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("cuSeqlens", context_->cuSeqlens.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0,
                                                      "batchSize+1", baseParams_->batchSize + 1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqUsed", context_->seqUsed.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("startPos", context_->startPos.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("mmKv", context_->mmKv.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("mmScore", context_->mmScore.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_2,
                                                      "2*coff*headDim", stateNum * static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0, "cmpRatio",
                                                      baseParams_->cmpRatio)) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::CONTINUOUS) &&
        (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0,
                                                       "blockNum", pageAttentionParams_->blockNum) ||
         ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1,
                                                       "blockSize", pageAttentionParams_->blockSize))) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckShapeConsistencyRope() const
{
    auto cmpT = std::min(baseParams_->tokenSize, baseParams_->tokenSize / baseParams_->cmpRatio + baseParams_->batchSize);
    if (context_->layout == LayoutType::LAYOUT_BSH) {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1, "ceil(seqSize/cmpRatio)", baseParams_->cgSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1, "ceil(seqSize/cmpRatio)", baseParams_->cgSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_2, "ropeHeadDim", baseParams_->ropeHeadDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_2, "ropeHeadDim", baseParams_->ropeHeadDim)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0, "min(tokenSize, tokenSize/cmpRatio+batchSize)", static_cast<uint32_t>(cmpT)) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_0, "min(tokenSize, tokenSize/cmpRatio+batchSize)", static_cast<uint32_t>(cmpT)) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1, "ropeHeadDim", baseParams_->ropeHeadDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_EPILOGUE_DIM_INDEX_1, "ropeHeadDim", baseParams_->ropeHeadDim)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckDtypeConsistencyMm(const gert::CompileTimeTensorDesc *desc,
                                                         const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(
        actualDtype != context_->dtype,
        OP_LOGE(context_->opName, "%s datatype should be same with x: %s, but got %s", name.c_str(),
                DataTypeToSerialString(actualDtype).c_str(), DataTypeToSerialString(context_->dtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckDtypeConsistencyRope() const
{
    auto sinDtype = context_->ropeSin.desc->GetDataType();
    auto cosDtype = context_->ropeCos.desc->GetDataType();
    OP_CHECK_IF(
        sinDtype != cosDtype,
        OP_LOGE(context_->opName, "%s datatype should be same with %s: %s, but got %s", ROPE_COS_NAME.c_str(),
                ROPE_SIN_NAME.c_str(), DataTypeToSerialString(sinDtype).c_str(),
                DataTypeToSerialString(cosDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sinDtype != context_->dtype && sinDtype != ge::DT_FLOAT,
        OP_LOGE(context_->opName, "rope datatype should be same with x or DT_FLOAT, x is %s, but got %s",
                DataTypeToSerialString(context_->dtype).c_str(), DataTypeToSerialString(sinDtype).c_str()),
        return ge::GRAPH_FAILED);
    bool supportFp32Rope = socVersion_ == platform_ascendc::SocVersion::ASCEND910B ||
                           socVersion_ == platform_ascendc::SocVersion::ASCEND910_93;
    OP_CHECK_IF(
        sinDtype == ge::DT_FLOAT && !supportFp32Rope,
        OP_LOGE(context_->opName, "float32 rope is only enabled on ascend910b and ascend910_93."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyMm(context_->mmScore.desc, MM_SCORE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyMm(context_->normWeight.desc, NORM_WEIGHT_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyRope() != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyMm(context_->cmpKv.desc, CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckDimNumConsistency() const
{
    auto xDimNum = context_->mmKv.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != context_->ropeSin.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "ropeSin dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->ropeSin.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum != context_->ropeCos.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "ropeCos dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->ropeCos.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum != context_->cmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "cmpKv dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->cmpKv.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckScenarioConsistency() const
{
    auto curCmpratio = baseParams_->cmpRatio;
    auto curHeaddim = baseParams_->headDim;
    auto curCoff = static_cast<uint8_t>(*context_->coff);
    std::vector<uint32_t> curScenario{curCmpratio, curCoff, curHeaddim};
    const std::vector<std::vector<uint32_t>> allowdScenarios = {{4, 2, 512}, {4, 2, 128}, {128, 1, 512}};

    OP_CHECK_IF(std::find(allowdScenarios.begin(), allowdScenarios.end(), curScenario) == allowdScenarios.end(),
                OP_LOGE(context_->opName, "Cmpratio Coff Headdim should be equal to {4, 2, 512}, {4, 2, 128}, {128, 1, 512},\
 but now cmpratio=%u, coff=%u, headdim=%u", curCmpratio, curCoff, curHeaddim), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64;  // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum, OP_LOGE(context_->opName, "aicNum is %d, which should not be less than %d",
    aicNum_, minBlockNum), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorEpilogueTiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#ifdef DAY0_SCOPE
    if (CheckScenarioConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#endif
    return ge::GRAPH_SUCCESS;
}

CMP_EXTERN_C ge::graphStatus TilingCompressorEpilogue(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CompressorEpilogue", "Context is nullptr."),
               return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    CompressorEpilogueContext compressor_epilogueContext{};
    if (CompressorEpilogueTiling::ConvertContext(*context, compressor_epilogueContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to CompressorEpilogue context");
        return ge::GRAPH_FAILED;
    }
    CompressorEpilogueTiling compressor_epilogueTiling(&compressor_epilogueContext);
    CompressorEpilogueTilingData* tilingData = context->GetTilingData<CompressorEpilogueTilingData>();
    OP_CHECK_IF(tilingData == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(compressor_epilogueContext.opName, "TilingData is nullptr."),
            return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (compressor_epilogueTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(compressor_epilogueContext.tilingKey);
    context->SetBlockDim(compressor_epilogueContext.blockDim);
    OP_LOGI(compressor_epilogueContext.opName, "block dim: %u.", compressor_epilogueContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForCompressorEpilogue(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressorEpilogue)
    .Tiling(TilingCompressorEpilogue)
    .TilingParse<CompressorEpilogueCompileInfo>(TilingPrepareForCompressorEpilogue);
} // namespace optiling
