// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <limits>
#include "v41_moe_router_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingRouter(gert::TilingContext *context)
{
    if (!context->GetPlatformInfo() || !context->GetAttrs()) { return ge::GRAPH_FAILED; }
    const ge::DataType types[] = {ge::DT_FLOAT, ge::DT_INT64, ge::DT_BOOL,
        ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32};
    for (uint32_t i = 0; i < 8; ++i) {
        if (i == 3 || i == 4) { continue; }
        if (!context->GetInputShape(i) || !context->GetInputDesc(i) ||
            context->GetInputDesc(i)->GetDataType() != types[i]) { return ge::GRAPH_FAILED; }
    }
    const auto &x = context->GetInputShape(0)->GetStorageShape();
    const auto *topK = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    const auto *renormalize = context->GetAttrs()->GetAttrPointer<bool>(1);
    const auto *scaling = context->GetAttrs()->GetAttrPointer<float>(2);
    if (x.GetDimNum() != 2 || x.GetDim(0) < 1 ||
        x.GetDim(0) > std::numeric_limits<uint32_t>::max() || !topK || !renormalize || !scaling ||
        !std::isfinite(*scaling)) { return ge::GRAPH_FAILED; }
    const int64_t rows = x.GetDim(0), experts = x.GetDim(1);
    if (!((experts == 384 && *topK == 6) || (experts == 128 && *topK == 3))) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i : {1, 2}) {
        const auto &s = context->GetInputShape(i)->GetStorageShape();
        if (s.GetDimNum() != 1 || s.GetDim(0) != rows) { return ge::GRAPH_FAILED; }
    }
    for (uint32_t i : {6, 7}) {
        const auto &s = context->GetInputShape(i)->GetStorageShape();
        if (s.GetDimNum() != 2 || s.GetDim(0) != rows || s.GetDim(1) != *topK) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto &imageBias = context->GetInputShape(5)->GetStorageShape();
    if (imageBias.GetDimNum() != 1 || imageBias.GetDim(0) != experts) { return ge::GRAPH_FAILED; }
    uint32_t vocabulary = 0, hasTextBias = 0;
    const auto *table = context->GetOptionalInputShape(3);
    if (table && table->GetStorageShape().GetShapeSize() != 0) {
        const auto &s = table->GetStorageShape();
        if (!context->GetOptionalInputDesc(3) || context->GetOptionalInputDesc(3)->GetDataType() != ge::DT_INT32 ||
            s.GetDimNum() != 2 || s.GetDim(0) < 1 || s.GetDim(0) > std::numeric_limits<uint32_t>::max() ||
            s.GetDim(1) != *topK) { return ge::GRAPH_FAILED; }
        vocabulary = s.GetDim(0);
    }
    const auto *bias = context->GetOptionalInputShape(4);
    if (bias && bias->GetStorageShape().GetShapeSize() != 0) {
        const auto &s = bias->GetStorageShape();
        if (!context->GetOptionalInputDesc(4) || context->GetOptionalInputDesc(4)->GetDataType() != ge::DT_FLOAT ||
            s.GetDimNum() != 1 || s.GetDim(0) != experts) { return ge::GRAPH_FAILED; }
        hasTextBias = 1;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = std::min(platform.GetCoreNumAiv(), static_cast<uint32_t>(rows));
    if (!cores) { return ge::GRAPH_FAILED; }
    V41MoeRouterTilingData data;
    data.set_rows(rows); data.set_experts(experts); data.set_topK(*topK); data.set_cores(cores);
    data.set_vocabulary(vocabulary); data.set_hasTextBias(hasTextBias);
    data.set_renormalize(*renormalize); data.set_scaling(*scaling);
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(V41MoeRouter).Tiling(TilingRouter);
}  // namespace optiling
