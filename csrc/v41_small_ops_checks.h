// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>
#include <cmath>
#include <initializer_list>

namespace vllm_ascend::v41 {
inline void tensor(const at::Tensor &value, const at::Tensor &anchor,
                   at::ScalarType dtype, bool contiguous = true)
{
    TORCH_CHECK(value.device() == anchor.device() && value.scalar_type() == dtype,
                "V4.1 tensor dtype/device mismatch");
    TORCH_CHECK(!contiguous || value.is_contiguous(), "V4.1 tensor must be contiguous");
}

inline void no_alias(const at::Tensor &output,
                     std::initializer_list<const at::Tensor *> inputs)
{
    for (const auto *input : inputs) {
        TORCH_CHECK(!output.is_alias_of(*input), "V4.1 output must not share input storage");
    }
}

inline void dspark_metadata(const at::Tensor &cu_q, const at::Tensor &lengths,
                            const at::Tensor &topk_lengths, const at::Tensor &schedule)
{
    tensor(cu_q, cu_q, at::kInt);
    tensor(lengths, cu_q, at::kInt);
    tensor(topk_lengths, cu_q, at::kInt);
    tensor(schedule, cu_q, at::kInt);
    TORCH_CHECK(cu_q.dim() == 1 && cu_q.size(0) >= 1 && lengths.dim() == 1 &&
                cu_q.size(0) == lengths.size(0) + 1, "DSpark metadata requires [B+1] boundaries and [B] lengths");
    TORCH_CHECK(topk_lengths.dim() == 2 && topk_lengths.size(1) == 1,
                "DSpark metadata requires [T,1] top-k lengths");
    TORCH_CHECK(lengths.size(0) <= 4096 && topk_lengths.size(0) <= 32768,
                "DSpark metadata supports at most 4096 requests and 32768 query tokens");
    TORCH_CHECK(schedule.dim() == 1 && schedule.size(0) == 1024,
                "DSpark metadata requires 1024 INT32 schedule words");
    no_alias(schedule, {&cu_q, &lengths, &topk_lengths});
}

inline void rope_inputs(const at::Tensor &x, const at::Tensor &positions,
                        const at::Tensor &cos, const at::Tensor &sin)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "V4.1 RoPE expects [T,D] or [T,H,D]");
    TORCH_CHECK(x.size(-1) == 128 || x.size(-1) == 512, "V4.1 RoPE requires D128/512");
    TORCH_CHECK(x.dim() == 2 || x.size(1) == 1 || x.size(1) == 8 || x.size(1) == 32,
                "V4.1 RoPE requires H1/8/32");
    tensor(x, x, at::kBFloat16);
    tensor(positions, x, at::kLong);
    tensor(cos, x, at::kFloat);
    tensor(sin, x, at::kFloat);
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == x.size(0), "Invalid V4.1 positions");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) > 0 && cos.size(1) == 32 && sin.sizes() == cos.sizes(),
                "V4.1 RoPE tables must be matching [P,32]");
}

inline void rope(const at::Tensor &x, const at::Tensor &positions,
                 const at::Tensor &cos, const at::Tensor &sin, const at::Tensor &output)
{
    rope_inputs(x, positions, cos, sin);
    tensor(output, x, at::kBFloat16);
    TORCH_CHECK(output.sizes() == x.sizes(), "Invalid V4.1 RoPE output shape");
    no_alias(output, {&x, &positions, &cos, &sin});
}

inline void cache(const at::Tensor &value, const at::Tensor &x, at::ScalarType dtype, int64_t width)
{
    tensor(value, x, dtype, false);
    TORCH_CHECK(value.dim() == 3 || value.dim() == 4, "V4.1 cache must have 3 or 4 axes");
    TORCH_CHECK(value.size(0) > 0 && value.size(1) > 0 && value.size(-1) == width &&
                (value.dim() == 3 || value.size(2) == 1), "Invalid V4.1 cache shape");
    int64_t stride = 1;
    for (int64_t axis = value.dim() - 1; axis > 0; --axis) {
        TORCH_CHECK(value.size(axis) == 1 || value.stride(axis) == stride,
                    "Only the V4.1 cache page axis may have gaps");
        stride *= value.size(axis);
    }
    TORCH_CHECK(value.stride(0) >= stride, "V4.1 cache pages must not overlap");
}

inline void store_inputs(const at::Tensor &x, const at::Tensor &positions,
                         const at::Tensor &slots, const at::Tensor &cos,
                         const at::Tensor &sin, int64_t width, int64_t ratio)
{
    rope_inputs(x, positions, cos, sin);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == width, "Invalid V4.1 cache input width");
    TORCH_CHECK(ratio == 1 || ratio == 2, "V4.1 cache ratio must be 1 or 2");
    tensor(slots, x, at::kLong);
    TORCH_CHECK(slots.sizes() == positions.sizes(), "Invalid V4.1 physical slots");
}

inline void main_store(const at::Tensor &x, const at::Tensor &positions,
                       const at::Tensor &slots, const at::Tensor &cos,
                       const at::Tensor &sin, const at::Tensor &destination, int64_t ratio)
{
    store_inputs(x, positions, slots, cos, sin, 512, ratio);
    cache(destination, x, at::kBFloat16, 512);
    no_alias(destination, {&x, &positions, &slots, &cos, &sin});
}

inline void index_store(const at::Tensor &x, const at::Tensor &positions,
                        const at::Tensor &slots, const at::Tensor &cos, const at::Tensor &sin,
                        const at::Tensor &keys, const at::Tensor &scales, int64_t ratio)
{
    store_inputs(x, positions, slots, cos, sin, 128, ratio);
    cache(keys, x, at::kChar, 128);
    cache(scales, x, at::kHalf, 1);
    TORCH_CHECK(keys.size(0) == scales.size(0) && keys.size(1) == scales.size(1),
                "V4.1 key and scale caches must share page dimensions");
    // The runner packs keys and scales into disjoint regions of each raw
    // page. Storage identity alone does not imply overlapping cache writes.
    if (keys.is_alias_of(scales)) {
        const int64_t pitch = keys.stride(0) * keys.element_size();
        const int64_t scale_pitch = scales.stride(0) * scales.element_size();
        const int64_t key_bytes = keys.size(1) * 128 * keys.element_size();
        const int64_t scale_bytes = scales.size(1) * scales.element_size();
        int64_t delta = (scales.storage_offset() * scales.element_size() -
                         keys.storage_offset() * keys.element_size()) % pitch;
        if (delta < 0) delta += pitch;
        TORCH_CHECK(pitch == scale_pitch && delta >= key_bytes && delta + scale_bytes <= pitch,
                    "V4.1 packed key/scale regions must not overlap within or across pages");
    }
    no_alias(keys, {&x, &positions, &slots, &cos, &sin});
    no_alias(scales, {&x, &positions, &slots, &cos, &sin});
}

inline void router(const at::Tensor &logits, const at::Tensor &token_ids,
                   const at::Tensor &image_mask, const c10::optional<at::Tensor> &table,
                   const c10::optional<at::Tensor> &text_bias, const at::Tensor &image_bias,
                   const at::Tensor &weights, const at::Tensor &ids, int64_t k, double scale)
{
    TORCH_CHECK(logits.dim() == 2, "V4.1 router logits must be [T,E]");
    const int64_t t = logits.size(0), e = logits.size(1);
    TORCH_CHECK((e == 384 && k == 6) || (e == 128 && k == 3), "Unsupported V4.1 router E/K");
    TORCH_CHECK(std::isfinite(scale), "V4.1 router scaling must be finite");
    tensor(logits, logits, at::kFloat);
    tensor(token_ids, logits, at::kLong);
    tensor(image_mask, logits, at::kBool);
    tensor(image_bias, logits, at::kFloat);
    tensor(weights, logits, at::kFloat);
    tensor(ids, logits, at::kInt);
    TORCH_CHECK(token_ids.dim() == 1 && token_ids.size(0) == t && image_mask.sizes() == token_ids.sizes(),
                "V4.1 router IDs/mask must be [T]");
    TORCH_CHECK(image_bias.dim() == 1 && image_bias.size(0) == e, "Invalid image bias");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == t && weights.size(1) == k &&
                ids.sizes() == weights.sizes(), "Invalid V4.1 router outputs");
    no_alias(weights, {&logits, &token_ids, &image_mask, &image_bias, &ids});
    no_alias(ids, {&logits, &token_ids, &image_mask, &image_bias});
    if (table.has_value()) {
        tensor(*table, logits, at::kInt);
        TORCH_CHECK(table->dim() == 2 && table->size(0) > 0 && table->size(1) == k, "Invalid routing table");
        no_alias(weights, {&*table});
        no_alias(ids, {&*table});
    }
    if (text_bias.has_value()) {
        tensor(*text_bias, logits, at::kFloat);
        TORCH_CHECK(text_bias->sizes() == image_bias.sizes(), "Invalid text bias");
        no_alias(weights, {&*text_bias});
        no_alias(ids, {&*text_bias});
    }
}
}  // namespace vllm_ascend::v41
