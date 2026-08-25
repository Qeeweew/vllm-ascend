#include "w4a16_moe_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024;

static ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto x = context->GetInputShape(0)->GetStorageShape();
    const auto w13 = context->GetInputShape(1)->GetStorageShape();
    const auto w2 = context->GetInputShape(3)->GetStorageShape();
    const auto ids = context->GetInputShape(5)->GetStorageShape();
    OP_CHECK_IF(x.GetDimNum() != 2 || w13.GetDimNum() != 3 || w2.GetDimNum() != 3 || ids.GetDimNum() != 2,
                OP_LOGE(context, "W4A16Moe expects x/ids 2-D and weights 3-D"), return ge::GRAPH_FAILED);

    const uint32_t batch = x.GetDim(0);
    const uint32_t hidden = x.GetDim(1);
    const uint32_t experts = w13.GetDim(0);
    const uint32_t inter = w2.GetDim(1);
    const uint32_t topk = ids.GetDim(1);
    OP_CHECK_IF(hidden % 128 != 0 || inter % 128 != 0,
                OP_LOGE(context, "hidden and intermediate dimensions must be divisible by 128"),
                return ge::GRAPH_FAILED);

    W4A16MoeTilingData tiling;
    tiling.set_batch_size(batch);
    tiling.set_hidden_size(hidden);
    tiling.set_inter_size(inter);
    tiling.set_num_experts(experts);
    tiling.set_top_k(topk);
    const auto attrs = context->GetAttrs();
    tiling.set_swiglu_limit(attrs == nullptr ? 0.0f : *attrs->GetFloat(0));

    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "x must be FP16 or BF16"), return ge::GRAPH_FAILED);
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ? 1 : 2);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(platform.GetCoreNumAiv());

    const size_t elem_bytes = 2;
    const size_t routes = static_cast<size_t>(batch) * topk;
    const size_t user_workspace = routes * inter * 2 * sizeof(float) +
                                  routes * inter * elem_bytes +
                                  static_cast<size_t>(batch) * hidden * sizeof(float);
    context->GetWorkspaceSizes(1)[0] = SYSTEM_WORKSPACE_BYTES + user_workspace;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(W4a16Moe).Tiling(Tiling);
} // namespace optiling
