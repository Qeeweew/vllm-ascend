// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <limits>
#include "engram_gate_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingEngramGate(gert::TilingContext *context)
{
    if (!context->GetAttrs() || !context->GetPlatformInfo()) { return ge::GRAPH_FAILED; }
    const auto *eps = context->GetAttrs()->GetFloat(0);
    if (!eps || !std::isfinite(*eps) || *eps <= 0) { return ge::GRAPH_FAILED; }
    for (uint32_t i = 0; i < 6; ++i) {
        if (!context->GetInputShape(i) || !context->GetInputDesc(i) ||
            context->GetInputDesc(i)->GetDataType() != (i == 4 ? ge::DT_BOOL : ge::DT_BF16)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto &hidden = context->GetInputShape(0)->GetStorageShape();
    if (hidden.GetDimNum() != 3 || hidden.GetDim(1) != 4 || hidden.GetDim(2) != 5120 ||
        hidden.GetDim(0) < 0 || hidden.GetDim(0) > std::numeric_limits<int32_t>::max() / 4) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tokens = hidden.GetDim(0);
    const auto &kv = context->GetInputShape(1)->GetStorageShape();
    const auto &mask = context->GetInputShape(4)->GetStorageShape();
    const auto &output = context->GetInputShape(5)->GetStorageShape();
    if (kv.GetDimNum() != 2 || kv.GetDim(0) != tokens || kv.GetDim(1) != 25600 ||
        mask.GetDimNum() != 1 || mask.GetDim(0) != tokens || output.GetDimNum() != 3 ||
        output.GetDim(0) != tokens || output.GetDim(1) != 4 || output.GetDim(2) != 5120) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i : {2, 3}) {
        const auto &w = context->GetInputShape(i)->GetStorageShape();
        if (w.GetDimNum() != 2 || w.GetDim(0) != 4 || w.GetDim(1) != 5120) {
            return ge::GRAPH_FAILED;
        }
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t available = platform.GetCoreNumAiv();
    if (!available) { return ge::GRAPH_FAILED; }
    const uint32_t cores = std::max(1U, std::min(available, tokens * 4));
    EngramGateTilingData data;
    data.set_tokens(tokens);
    data.set_cores(cores);
    data.set_eps(*eps);
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(EngramGate).Tiling(TilingEngramGate);
}  // namespace optiling
