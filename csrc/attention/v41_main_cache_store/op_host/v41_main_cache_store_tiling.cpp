// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <limits>
#include "v41_main_cache_store_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingV41MainCacheStore(gert::TilingContext *context)
{
    if (context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) return ge::GRAPH_FAILED;
    const auto *ratio = context->GetAttrs()->GetInt(0);
    const auto *stride = context->GetAttrs()->GetInt(1);
    if (ratio == nullptr || stride == nullptr || (*ratio != 1 && *ratio != 2)) return ge::GRAPH_FAILED;
    const ge::DataType types[6] = {ge::DT_BF16, ge::DT_INT64, ge::DT_INT64,
                                  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16};
    for (uint32_t i = 0; i < 6; ++i) {
        if (context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr ||
            context->GetInputDesc(i)->GetDataType() != types[i]) return ge::GRAPH_FAILED;
    }
    const auto &x = context->GetInputShape(0)->GetOriginShape();
    const auto &positions = context->GetInputShape(1)->GetOriginShape();
    const auto &slots = context->GetInputShape(2)->GetOriginShape();
    const auto &cos = context->GetInputShape(3)->GetOriginShape();
    const auto &sin = context->GetInputShape(4)->GetOriginShape();
    const auto &cache = context->GetInputShape(5)->GetOriginShape();
    if (x.GetDimNum() != 2 || x.GetDim(1) != 512 || x.GetDim(0) < 0 ||
        x.GetDim(0) > std::numeric_limits<int32_t>::max() ||
        positions.GetDimNum() != 1 || positions.GetDim(0) != x.GetDim(0) ||
        slots.GetDimNum() != 1 || slots.GetDim(0) != x.GetDim(0) ||
        cos.GetDimNum() != 2 || sin.GetDimNum() != 2 || cos.GetDim(1) != 32 || sin.GetDim(1) != 32 ||
        cos.GetDim(0) != sin.GetDim(0) || cos.GetDim(0) < 1 ||
        cos.GetDim(0) > std::numeric_limits<int32_t>::max() ||
        (cache.GetDimNum() != 3 && cache.GetDimNum() != 4) ||
        (cache.GetDimNum() == 4 && cache.GetDim(2) != 1) || cache.GetDim(cache.GetDimNum() - 1) != 512 ||
        cache.GetDim(0) < 1 || cache.GetDim(0) > std::numeric_limits<int32_t>::max() ||
        cache.GetDim(1) < 1 || cache.GetDim(1) > std::numeric_limits<int32_t>::max() ||
        *stride < cache.GetDim(1) * 512) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetCoreNumAiv() == 0) return ge::GRAPH_FAILED;
    const uint32_t cores = std::max(1U, std::min(platform.GetCoreNumAiv(), uint32_t(x.GetDim(0))));
    V41MainCacheStoreTilingData data;
    data.set_tokens(x.GetDim(0));
    data.set_page(cache.GetDim(1));
    data.set_blocks(cache.GetDim(0));
    data.set_tableRows(cos.GetDim(0));
    data.set_cores(cores);
    data.set_ratio(*ratio);
    data.set_cacheStride(*stride);
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(V41MainCacheStore).Tiling(TilingV41MainCacheStore);
}  // namespace optiling
