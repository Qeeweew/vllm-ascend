// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include "indexer_v41_candidate_score_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingScore(gert::TilingContext *context)
{
    const ge::DataType types[] = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    for (uint32_t i = 0; i < 6; ++i) {
        if (!context->GetInputShape(i) || !context->GetInputDesc(i) ||
            context->GetInputDesc(i)->GetDataType() != types[i]) { return ge::GRAPH_FAILED; }
    }
    if (!context->GetPlatformInfo()) { return ge::GRAPH_FAILED; }
    const auto &qk = context->GetInputShape(0)->GetStorageShape();
    if (qk.GetDimNum() != 3 || qk.GetDim(0) != 1 || qk.GetDim(1) != 32 ||
        qk.GetDim(2) < 8 || qk.GetDim(2) > 16384 || qk.GetDim(2) % 8) { return ge::GRAPH_FAILED; }
    for (uint32_t i : {1, 2}) {
        const auto &shape = context->GetInputShape(i)->GetStorageShape();
        if (shape.GetDimNum() != 2 || shape.GetDim(0) != 1 || shape.GetDim(1) != 32) { return ge::GRAPH_FAILED; }
    }
    for (uint32_t i : {3, 4}) {
        const auto &shape = context->GetInputShape(i)->GetStorageShape();
        if (shape.GetDimNum() != 1 || shape.GetDim(0) != qk.GetDim(2)) { return ge::GRAPH_FAILED; }
    }
    const auto &output = context->GetInputShape(5)->GetStorageShape();
    if (output.GetDimNum() != 2 || output.GetDim(0) != 1 || output.GetDim(1) != qk.GetDim(2)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t available = platform.GetCoreNumAiv();
    if (!available) { return ge::GRAPH_FAILED; }
    const uint32_t cores = std::min(available, static_cast<uint32_t>((qk.GetDim(2) + 255) / 256));
    IndexerV41CandidateScoreTilingData data;
    data.set_positions(qk.GetDim(2));
    data.set_cores(cores);
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(IndexerV41CandidateScore).Tiling(TilingScore);
}  // namespace optiling
