// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <limits>
#include "indexer_v41_candidate_gather_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingGather(gert::TilingContext *context)
{
    const ge::DataType types[] = {ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32};
    for (uint32_t i = 0; i < 9; ++i) {
        if (!context->GetInputShape(i) || !context->GetInputDesc(i) ||
            context->GetInputDesc(i)->GetDataType() != types[i]) { return ge::GRAPH_FAILED; }
    }
    if (!context->GetAttrs() || !context->GetPlatformInfo()) { return ge::GRAPH_FAILED; }
    const auto *keyStride = context->GetAttrs()->GetInt(0);
    const auto *scaleStride = context->GetAttrs()->GetInt(1);
    const auto &key = context->GetInputShape(0)->GetOriginShape();
    const auto &scale = context->GetInputShape(1)->GetOriginShape();
    const auto &candidates = context->GetInputShape(2)->GetStorageShape();
    const auto &table = context->GetInputShape(3)->GetStorageShape();
    const auto &length = context->GetInputShape(4)->GetStorageShape();
    const auto &boundaries = context->GetInputShape(5)->GetStorageShape();
    const auto &output = context->GetInputShape(6)->GetStorageShape();
    const auto &outScale = context->GetInputShape(7)->GetStorageShape();
    const auto &positions = context->GetInputShape(8)->GetStorageShape();
    if (key.GetDimNum() != 4 || key.GetDim(0) < 1 || key.GetDim(0) > INT32_MAX ||
        key.GetDim(1) < 8 || key.GetDim(1) % 8 || key.GetDim(1) > INT32_MAX / 128 ||
        key.GetDim(2) != 1 || key.GetDim(3) != 128 || scale.GetDimNum() != 3 ||
        scale.GetDim(0) != key.GetDim(0) || scale.GetDim(1) != key.GetDim(1) || scale.GetDim(2) != 1 ||
        !keyStride || !scaleStride || *keyStride < key.GetDim(1) * 128 || *scaleStride < key.GetDim(1) ||
        candidates.GetDimNum() != 1 || candidates.GetDim(0) != 2048 ||
        table.GetDimNum() != 2 || table.GetDim(0) != 1 || table.GetDim(1) < 1 ||
        table.GetDim(1) * key.GetDim(1) > (1LL << 27) ||
        length.GetDimNum() != 1 || length.GetDim(0) != 1 ||
        boundaries.GetDimNum() != 1 || boundaries.GetDim(0) != 2 ||
        output.GetDimNum() != 3 || output.GetDim(0) != 1 || output.GetDim(2) != 128 ||
        output.GetDim(1) < 8 || output.GetDim(1) > 16384 || output.GetDim(1) % 8 ||
        outScale.GetDimNum() != 1 || outScale.GetDim(0) != output.GetDim(1) ||
        positions.GetDimNum() != 1 || positions.GetDim(0) != output.GetDim(1)) { return ge::GRAPH_FAILED; }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t available = platform.GetCoreNumAiv();
    if (!available) { return ge::GRAPH_FAILED; }
    const uint32_t cores = std::min(available, static_cast<uint32_t>((output.GetDim(1) + 63) / 64));
    IndexerV41CandidateGatherTilingData data;
    data.set_positions(output.GetDim(1));
    data.set_cores(cores);
    data.set_pageSize(key.GetDim(1));
    data.set_pages(key.GetDim(0));
    data.set_tablePages(table.GetDim(1));
    data.set_reserved(0);
    data.set_keyStride(*keyStride);
    data.set_scaleStride(*scaleStride);
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(IndexerV41CandidateGather).Tiling(TilingGather);
}  // namespace optiling
