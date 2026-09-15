// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <limits>

#include "compressor_v41_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingCompressorV41(gert::TilingContext *context)
{
    if (context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *ratioPtr = context->GetAttrs()->GetInt(0);
    const auto *epsPtr = context->GetAttrs()->GetFloat(1);
    if (ratioPtr == nullptr || epsPtr == nullptr || (*ratioPtr != 1 && *ratioPtr != 2) ||
        !std::isfinite(*epsPtr) || *epsPtr <= 0) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < 8; ++i) {
        if (context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t ratio = *ratioPtr;
    const auto &raw = context->GetInputShape(0)->GetStorageShape();
    if (raw.GetDimNum() != 2 || raw.GetDim(1) != 512 * ratio || raw.GetDim(0) < 0 ||
        raw.GetDim(0) > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tokens = raw.GetDim(0);
    const ge::DataType types[8] = {
        ratio == 1 ? ge::DT_BF16 : ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT32, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16};
    for (uint32_t i = 0; i < 8; ++i) {
        if (context->GetInputDesc(i)->GetDataType() != types[i]) {
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i : {1, 2}) {
        const auto &shape = context->GetInputShape(i)->GetStorageShape();
        if (shape.GetDimNum() != 1 || shape.GetDim(0) != tokens) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto &weight = context->GetInputShape(5)->GetStorageShape();
    const auto &output = context->GetInputShape(7)->GetStorageShape();
    const auto &starts = context->GetInputShape(3)->GetStorageShape();
    const auto &ids = context->GetInputShape(4)->GetStorageShape();
    if (weight.GetDimNum() != 1 || weight.GetDim(0) != 512 || output.GetDimNum() != 2 ||
        output.GetDim(0) != tokens || output.GetDim(1) != 512 ||
        starts.GetDimNum() != 1 || ids.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    uint32_t requests = 0;
    uint32_t capacity = 0;
    uint32_t stateBlocks = 0;
    if (ratio == 2) {
        const auto &state = context->GetInputShape(6)->GetStorageShape();
        if (state.GetDimNum() != 3 || state.GetDim(2) != 1024 ||
            state.GetDim(0) < 0 || state.GetDim(0) > std::numeric_limits<int32_t>::max() ||
            state.GetDim(1) < 8 || state.GetDim(1) > std::numeric_limits<int32_t>::max() ||
            starts.GetDim(0) < 1 || starts.GetDim(0) > std::numeric_limits<int32_t>::max() ||
            ids.GetDim(0) != tokens) {
            return ge::GRAPH_FAILED;
        }
        capacity = state.GetDim(1);
        if ((capacity & (capacity - 1)) != 0) {
            return ge::GRAPH_FAILED;
        }
        stateBlocks = state.GetDim(0);
        requests = starts.GetDim(0) - 1;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t available = platform.GetCoreNumAiv();
    if (available == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t cores = std::max(1U, std::min(available, tokens + requests));
    CompressorV41TilingData data;
    data.set_tokens(tokens);
    data.set_requests(requests);
    data.set_capacity(capacity);
    data.set_stateBlocks(stateBlocks);
    data.set_cores(cores);
    data.set_eps(*epsPtr);
    context->SetBlockDim(cores);
    context->SetTilingKey(ratio);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(CompressorV41).Tiling(TilingCompressorV41);
}  // namespace optiling
