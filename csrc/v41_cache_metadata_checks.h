// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "v41_small_ops_checks.h"
namespace vllm_ascend::v41 {
inline void cache_metadata(const at::Tensor &pi, const at::Tensor &ci, const at::Tensor &li, const at::Tensor &ti,
    const at::Tensor &po, const at::Tensor &co, const at::Tensor &lo, const at::Tensor &to,
    const at::Tensor &ro, const at::Tensor &so, const at::Tensor &cm, const at::Tensor &re,
    int64_t logical, int64_t physical, int64_t ratio, bool compressed)
{
    for (auto *x : {&pi, &po, &so}) tensor(*x, pi, at::kLong);
    for (auto *x : {&ci, &li, &ti, &co, &lo, &to, &ro, &cm, &re}) tensor(*x, pi, at::kInt);
    TORCH_CHECK(pi.dim() == 1 && ci.dim() == 1 && li.dim() == 1 && ti.dim() == 2 && po.dim() == 1 &&
                co.dim() == 1 && lo.dim() == 1 && to.dim() == 2 && ro.dim() == 1 && so.dim() == 1 &&
                cm.dim() == 1 && re.dim() == 1, "Invalid V4.1 cache metadata tensor rank");
    const int64_t batch = lo.numel(), tokens = po.numel();
    TORCH_CHECK(batch <= 4096 && tokens <= 32768 && to.size(1) > 0 && to.size(1) <= 1048576 &&
                pi.numel() >= tokens && ci.numel() >= batch + 1 && li.numel() >= batch &&
                ti.size(0) >= batch && ti.size(1) <= to.size(1) && co.numel() == batch + 1 &&
                to.size(0) == batch && ro.numel() == tokens && so.numel() == tokens &&
                cm.numel() == batch && re.numel() == batch, "Invalid V4.1 cache metadata tensor shape");
    TORCH_CHECK((ratio == 1 || ratio == 2) && physical >= 16 && physical <= 1024 && physical % 16 == 0 &&
                logical == physical * ratio, "Invalid V4.1 cache metadata block geometry");
    // Caller-owned output buffers must remain independent, including during capture.
    std::initializer_list<const at::Tensor *> outputs{&po, &co, &lo, &to, &ro, &so, &cm, &re};
    for (auto *out : outputs) {
        no_alias(*out, {&pi, &ci, &li, &ti});
        for (auto *other : outputs) if (out != other) no_alias(*out, {other});
    }
}
}
