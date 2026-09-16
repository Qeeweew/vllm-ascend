// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <limits>
#include "v41_rope_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingV41Rope(gert::TilingContext *context)
{
    if (context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType types[5] = {ge::DT_BF16, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16};
    for (uint32_t i = 0; i < 5; ++i) {
        if (context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr ||
            context->GetInputDesc(i)->GetDataType() != types[i]) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto &x = context->GetInputShape(0)->GetStorageShape();
    const auto &positions = context->GetInputShape(1)->GetStorageShape();
    const auto &cos = context->GetInputShape(2)->GetStorageShape();
    const auto &sin = context->GetInputShape(3)->GetStorageShape();
    const auto &output = context->GetInputShape(4)->GetStorageShape();
    if ((x.GetDimNum() != 2 && x.GetDimNum() != 3) || x.GetDim(0) < 0 ||
        x.GetDim(0) > std::numeric_limits<int32_t>::max() || output.GetDimNum() != x.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    const int64_t heads = x.GetDimNum() == 3 ? x.GetDim(1) : 1;
    const int64_t width = x.GetDim(x.GetDimNum() - 1);
    if ((heads != 1 && heads != 8 && heads != 32) || (width != 128 && width != 512) ||
        x.GetDim(0) * heads > std::numeric_limits<int32_t>::max() ||
        positions.GetDimNum() != 1 || positions.GetDim(0) != x.GetDim(0) ||
        cos.GetDimNum() != 2 || sin.GetDimNum() != 2 || cos.GetDim(1) != 32 || sin.GetDim(1) != 32 ||
        cos.GetDim(0) != sin.GetDim(0) || cos.GetDim(0) < 1 ||
        cos.GetDim(0) > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t axis = 0; axis < x.GetDimNum(); ++axis) {
        if (x.GetDim(axis) != output.GetDim(axis)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *inverse = context->GetAttrs()->GetBool(0);
    if (inverse == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetCoreNumAiv() == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t rows = x.GetDim(0) * heads;
    const uint32_t cores = std::max(1U, std::min(platform.GetCoreNumAiv(), rows));
    V41RopeTilingData data;
    data.set_rows(rows);
    data.set_heads(heads);
    data.set_width(width);
    data.set_tableRows(cos.GetDim(0));
    data.set_cores(cores);
    context->SetBlockDim(cores);
    context->SetTilingKey(*inverse ? 1 : 0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(V41Rope).Tiling(TilingV41Rope);
}  // namespace optiling
