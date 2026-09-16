// SPDX-License-Identifier: Apache-2.0
#include "v41_dspark_metadata_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *context)
{
    if (context->GetPlatformInfo() == nullptr) { return ge::GRAPH_FAILED; }
    for (uint32_t i = 0; i < 4; ++i) {
        if (context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr ||
            context->GetInputDesc(i)->GetDataType() != ge::DT_INT32) { return ge::GRAPH_FAILED; }
    }
    const auto &cu = context->GetInputShape(0)->GetStorageShape();
    const auto &kv = context->GetInputShape(1)->GetStorageShape();
    const auto &lengths = context->GetInputShape(2)->GetStorageShape();
    const auto &output = context->GetInputShape(3)->GetStorageShape();
    // Only query offsets are staged into UB (at most 16 KiB + 32 bytes).
    // Length tensors are consumed by SMLA; they do not affect this partition.
    if (cu.GetDimNum() != 1 || kv.GetDimNum() != 1 || lengths.GetDimNum() != 2 ||
        output.GetDimNum() != 1 || output.GetDim(0) != 1024 || lengths.GetDim(1) != 1 ||
        kv.GetDim(0) < 0 || kv.GetDim(0) > 4096 || cu.GetDim(0) != kv.GetDim(0) + 1 ||
        lengths.GetDim(0) < 0 || lengths.GetDim(0) > 32768) { return ge::GRAPH_FAILED; }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetCoreNumAic() != 20 || platform.GetCoreNumAiv() < 1) { return ge::GRAPH_FAILED; }
    V41DsparkMetadataTilingData data;
    data.set_batch(kv.GetDim(0));
    data.set_tokens(lengths.GetDim(0));
    context->SetBlockDim(1);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(V41DsparkMetadata).Tiling(Tiling);
}  // namespace optiling
