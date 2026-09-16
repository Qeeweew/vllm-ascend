// SPDX-License-Identifier: Apache-2.0
#include "v41_cache_metadata_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *context)
{
    if (!context->GetPlatformInfo() || !context->GetAttrs()) return ge::GRAPH_FAILED;
    for (uint32_t i = 0; i < 12; ++i) {
        if (!context->GetInputShape(i) || !context->GetInputDesc(i)) return ge::GRAPH_FAILED;
        const auto &shape = context->GetInputShape(i)->GetStorageShape();
        if (shape.GetDimNum() != (i == 3 || i == 7 ? 2 : 1) ||
            context->GetInputDesc(i)->GetDataType() != (i == 0 || i == 4 || i == 9 ? ge::DT_INT64 : ge::DT_INT32)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto dim = [context](int i, int j = 0) { return context->GetInputShape(i)->GetStorageShape().GetDim(j); };
    const int64_t batch = dim(6), tokens = dim(4), columns = dim(7, 1);
    const auto *logical = context->GetAttrs()->GetInt(0);
    const auto *physical = context->GetAttrs()->GetInt(1);
    const auto *ratio = context->GetAttrs()->GetInt(2);
    const auto *compressed = context->GetAttrs()->GetBool(3);
    if (!logical || !physical || !ratio || !compressed || batch < 0 || batch > 4096 || tokens < 0 ||
        tokens > 32768 || columns < 1 || columns > 1048576 || dim(3, 1) > columns ||
        dim(0) < tokens || dim(1) < batch + 1 || dim(2) < batch || dim(3) < batch ||
        dim(5) != batch + 1 || dim(7) != batch || dim(8) != tokens || dim(9) != tokens ||
        dim(10) != batch || dim(11) != batch || *logical <= 0 || *logical > INT32_MAX ||
        *physical < 16 || *physical > 1024 || *physical % 16 || (*ratio != 1 && *ratio != 2) ||
        *logical != *physical * *ratio) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = std::min<uint32_t>(platform.GetCoreNumAiv(), std::max<int64_t>({1, batch, tokens}));
    V41CacheMetadataTilingData data;
    data.set_batch(batch); data.set_tokens(tokens); data.set_inputColumns(dim(3, 1));
    data.set_outputColumns(columns); data.set_logicalBlock(*logical); data.set_physicalBlock(*physical);
    data.set_ratio(*ratio); data.set_compressed(*compressed); data.set_cores(cores);
    context->SetBlockDim(cores); context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(V41CacheMetadata).Tiling(Tiling);
}
