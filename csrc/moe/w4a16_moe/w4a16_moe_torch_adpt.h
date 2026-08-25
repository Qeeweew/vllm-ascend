#ifndef W4A16_MOE_TORCH_ADPT_H
#define W4A16_MOE_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor npu_w4a16_moe(
    const at::Tensor& x, const at::Tensor& w13, const at::Tensor& w13_scale,
    const at::Tensor& w2, const at::Tensor& w2_scale,
    const at::Tensor& expert_ids, const at::Tensor& topk_weights,
    double swiglu_limit);
} // namespace vllm_ascend
#endif
