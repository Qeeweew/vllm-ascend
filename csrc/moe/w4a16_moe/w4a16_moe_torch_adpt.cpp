#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_fused_moe_small_bs_w4a16_fp16.h"
#include "aclrtlaunch_fused_moe_small_bs_w4a16_bf16.h"
#include "w4a16_moe_torch_adpt.h"

namespace vllm_ascend {

at::Tensor npu_w4a16_moe(
    const at::Tensor& x, const at::Tensor& w13, const at::Tensor& w13_scale,
    const at::Tensor& w2, const at::Tensor& w2_scale,
    const at::Tensor& expert_ids, const at::Tensor& topk_weights,
    double swiglu_limit)
{
    TORCH_CHECK(x.dim() == 2, "x must be [batch, hidden]");
    TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3, "packed weights must be 3-D");
    TORCH_CHECK(w13.scalar_type() == at::kInt && w2.scalar_type() == at::kInt,
                "packed INT4 weights must use int32 storage");
    TORCH_CHECK(w13_scale.scalar_type() == x.scalar_type() &&
                w2_scale.scalar_type() == x.scalar_type(),
                "signed scales must have the activation dtype");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 2,
                "expert_ids must be int32 [batch, top_k]");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                topk_weights.sizes() == expert_ids.sizes(),
                "topk_weights must be FP32 and match expert_ids");

    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be FP16 or BF16");
    for (const auto& tensor : {x, w13, w13_scale, w2, w2_scale, expert_ids, topk_weights}) {
        TORCH_CHECK(tensor.device() == x.device() && tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "all inputs must be on the same NPU");
        TORCH_CHECK(tensor.is_contiguous(), "all inputs must be contiguous ND tensors");
        for (auto dim : tensor.sizes()) {
            TORCH_CHECK(dim <= std::numeric_limits<int32_t>::max(), "dimension exceeds int32 range");
        }
    }
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0,
                "swiglu_limit must be finite and nonnegative");
    const int32_t batch_size = static_cast<int32_t>(x.size(0));
    const int32_t hidden_size = static_cast<int32_t>(x.size(1));
    const int32_t top_k = static_cast<int32_t>(expert_ids.size(1));
    const int32_t num_experts = static_cast<int32_t>(w13.size(0));
    const int32_t inter_size = static_cast<int32_t>(w2.size(1));
    TORCH_CHECK(w13.size(1) == hidden_size,
                "w13 K dimension must equal hidden size");
    TORCH_CHECK(w13.size(2) * 8 == inter_size * 2,
                "w13 packed output must equal 2 * intermediate size");
    TORCH_CHECK(w2.size(2) * 8 == hidden_size,
                "w2 packed output must equal hidden size");

    TORCH_CHECK(hidden_size > 0 && hidden_size % 64 == 0 && inter_size > 0 && inter_size % 32 == 0,
                "hidden must be a positive multiple of 64, intermediate a positive multiple of 32");
    TORCH_CHECK(expert_ids.size(0) == batch_size && top_k > 0 && num_experts > 0,
                "routing batch must match x, and experts/top_k must be positive");
    TORCH_CHECK(w2.size(0) == num_experts, "expert counts must match");
    TORCH_CHECK(w13_scale.sizes() == at::IntArrayRef({num_experts, hidden_size / 32, inter_size * 2}),
                "w13 scales must be [experts, hidden/32, 2*intermediate]");
    TORCH_CHECK(w2_scale.sizes() == at::IntArrayRef({num_experts, inter_size / 32, hidden_size}),
                "w2 scales must be [experts, intermediate/32, hidden]");
    if (batch_size == 0) return at::empty_like(x);
    const int64_t routes = static_cast<int64_t>(batch_size) * top_k;
    const int64_t workspace_bytes =
        routes * inter_size * 2 * static_cast<int64_t>(sizeof(float)) +
        routes * inter_size * static_cast<int64_t>(x.element_size()) +
        static_cast<int64_t>(batch_size) * hidden_size *
            static_cast<int64_t>(sizeof(float));

    at::Tensor y = at::empty_like(x);
    at::Tensor workspace = at::empty(
        {workspace_bytes}, x.options().dtype(at::kByte));


    auto stream = c10_npu::getCurrentNPUStream();
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int32_t block_dim = static_cast<int32_t>(platform->GetCoreNumAiv());
    const float limit = static_cast<float>(swiglu_limit);

    if (x.scalar_type() == at::kHalf) {
        ACLRT_LAUNCH_KERNEL(fused_moe_small_bs_w4a16_fp16)(
            block_dim, stream,
            const_cast<void*>(x.data_ptr()),
            const_cast<void*>(w13.data_ptr()),
            const_cast<void*>(w13_scale.data_ptr()),
            const_cast<void*>(w2.data_ptr()),
            const_cast<void*>(w2_scale.data_ptr()),
            const_cast<void*>(expert_ids.data_ptr()), const_cast<void*>(topk_weights.data_ptr()),
            y.data_ptr(), workspace.data_ptr(),
            batch_size, hidden_size, inter_size, num_experts, top_k, limit);
    } else if (x.scalar_type() == at::kBFloat16) {
        ACLRT_LAUNCH_KERNEL(fused_moe_small_bs_w4a16_bf16)(
            block_dim, stream,
            const_cast<void*>(x.data_ptr()),
            const_cast<void*>(w13.data_ptr()),
            const_cast<void*>(w13_scale.data_ptr()),
            const_cast<void*>(w2.data_ptr()),
            const_cast<void*>(w2_scale.data_ptr()),
            const_cast<void*>(expert_ids.data_ptr()), const_cast<void*>(topk_weights.data_ptr()),
            y.data_ptr(), workspace.data_ptr(),
            batch_size, hidden_size, inter_size, num_experts, top_k, limit);
    } else {
        TORCH_CHECK(false, "x must be FP16 or BF16");
    }
    return y;
}

} // namespace vllm_ascend
