// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace vllm_ascend {
namespace {
void CheckCPU(const at::Tensor& t, at::ScalarType dtype, int64_t dims, const char* name) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == dtype && t.dim() == dims && t.is_contiguous(),
                name, " must be contiguous CPU tensor with the required dtype and rank");
}

void EngramHashGatherCPU(
    const at::Tensor& input_ids, const at::Tensor& query_start_loc,
    const at::Tensor& start_positions, const at::Tensor& lookback_ids,
    const at::Tensor& token_map, const at::Tensor& multipliers,
    const at::Tensor& primes, const at::Tensor& head_indices,
    const at::Tensor& local_starts, at::TensorList tables, at::TensorList outputs,
    int64_t pad_id, int64_t bucket_tokens,
    const std::optional<at::Tensor>& token_mask,
    const std::optional<at::Tensor>& lookback_mask) {
    CheckCPU(input_ids, at::kLong, 1, "input_ids");
    CheckCPU(query_start_loc, at::kLong, 1, "query_start_loc");
    CheckCPU(start_positions, at::kLong, 1, "start_positions");
    CheckCPU(lookback_ids, at::kLong, 2, "lookback_ids");
    CheckCPU(token_map, at::kLong, 1, "token_map");
    CheckCPU(multipliers, at::kLong, 2, "multipliers");
    CheckCPU(primes, at::kLong, 3, "primes");
    CheckCPU(head_indices, at::kLong, 2, "head_indices");
    CheckCPU(local_starts, at::kLong, 2, "local_starts");
    const int64_t tokens = input_ids.numel(), requests = start_positions.numel();
    const int64_t layers = multipliers.size(0), ngram = multipliers.size(1);
    TORCH_CHECK(layers > 0 && ngram >= 2 && primes.size(0) == layers &&
                primes.size(1) == ngram - 1 && primes.size(2) > 0, "Invalid Engram layer/ngram shape");
    const int64_t heads_per_order = primes.size(2), heads = head_indices.size(1);
    TORCH_CHECK(heads > 0 && head_indices.size(0) == layers && local_starts.sizes() == head_indices.sizes(),
                "Invalid local head metadata");
    TORCH_CHECK(static_cast<int64_t>(tables.size()) == layers && outputs.size() == tables.size(),
                "Invalid Engram table/output count");
    TORCH_CHECK(lookback_ids.size(0) == requests && lookback_ids.size(1) == ngram - 1 &&
                query_start_loc.numel() == requests + 1, "Invalid query/lookback shape");
    TORCH_CHECK(pad_id >= 0 && token_map.numel() > 0 && bucket_tokens >= tokens,
                "Invalid padding, vocabulary or token bucket");
    auto CheckMask = [](const std::optional<at::Tensor>& mask, const at::Tensor& ids) {
        if (mask.has_value()) {
            CheckCPU(*mask, at::kBool, ids.dim(), "token mask");
            TORCH_CHECK(mask->sizes() == ids.sizes(), "Token mask shape mismatch");
        }
    };
    CheckMask(token_mask, input_ids);
    CheckMask(lookback_mask, lookback_ids);
    const auto* ids = input_ids.const_data_ptr<int64_t>();
    const auto* cu = query_start_loc.const_data_ptr<int64_t>();
    const auto* starts = start_positions.const_data_ptr<int64_t>();
    const auto* prior = lookback_ids.const_data_ptr<int64_t>();
    const auto* mapping = token_map.const_data_ptr<int64_t>();
    const auto* mult = multipliers.const_data_ptr<int64_t>();
    const auto* moduli = primes.const_data_ptr<int64_t>();
    const auto* selected = head_indices.const_data_ptr<int64_t>();
    const auto* offsets = local_starts.const_data_ptr<int64_t>();
    const bool* mask = token_mask.has_value() ? token_mask->const_data_ptr<bool>() : nullptr;
    const bool* prior_mask = lookback_mask.has_value() ? lookback_mask->const_data_ptr<bool>() : nullptr;
    TORCH_CHECK(cu[0] == 0 && cu[requests] == tokens, "Invalid query boundaries");
    for (int64_t r = 0; r < requests; ++r) {
        TORCH_CHECK(cu[r] <= cu[r + 1] && starts[r] >= 0, "Invalid query boundaries or positions");
        if (cu[r] == cu[r + 1]) continue;
        for (int64_t j = 0; j < std::min(starts[r], ngram - 1); ++j) {
            const int64_t id = prior[r * (ngram - 1) + j];
            TORCH_CHECK(id >= 0 && id < token_map.numel(), "Actual lookback token IDs are required");
            TORCH_CHECK(mapping[id] >= 0, "Invalid compressed token ID");
        }
    }
    for (int64_t t = 0; t < tokens; ++t) {
        TORCH_CHECK(ids[t] >= 0 && ids[t] < token_map.numel(), "Invalid input token ID");
        TORCH_CHECK(mapping[ids[t]] >= 0, "Invalid compressed token ID");
    }
    std::vector<const at::BFloat16*> table_data(layers);
    std::vector<at::BFloat16*> output_data(layers);
    std::vector<int64_t> widths(layers);
    for (int64_t l = 0; l < layers; ++l) {
        CheckCPU(tables[l], at::kBFloat16, 2, "table");
        CheckCPU(outputs[l], at::kBFloat16, 3, "output");
        widths[l] = tables[l].size(1);
        TORCH_CHECK(widths[l] > 0 && outputs[l].size(0) >= bucket_tokens &&
                    outputs[l].size(1) == heads && outputs[l].size(2) == widths[l], "Invalid output shape/capacity");
        int64_t cursor = 0;
        for (int64_t h = 0; h < heads; ++h) {
            int64_t index = selected[l * heads + h];
            TORCH_CHECK(index >= 0 && index < (ngram - 1) * heads_per_order, "Invalid local head index");
            const int64_t prime = moduli[l * (ngram - 1) * heads_per_order + index];
            TORCH_CHECK(prime > 0 && prime < (int64_t{1} << 32) && offsets[l * heads + h] == cursor &&
                        prime <= tables[l].size(0) - cursor, "Invalid local bucket layout");
            for (int64_t j = 0; j < h; ++j) {
                TORCH_CHECK(selected[l * heads + j] != index, "Duplicate local head index");
            }
            cursor += prime;
        }
        TORCH_CHECK(cursor == tables[l].size(0), "Table rows do not match local buckets");
        table_data[l] = tables[l].const_data_ptr<at::BFloat16>();
        output_data[l] = outputs[l].data_ptr<at::BFloat16>();
    }
    // Small decode batches stay serial; each worker owns disjoint output rows.
    at::parallel_for(0, tokens, 64, [&](int64_t begin, int64_t end) {
        std::vector<uint64_t> window(ngram), rolling(ngram);
        int64_t r = std::upper_bound(cu + 1, cu + requests + 1, begin) - (cu + 1);
        for (int64_t t = begin; t < end; ++t) {
            while (t >= cu[r + 1]) ++r;
            bool blocked = false;
            for (int64_t shift = 0; shift < ngram; ++shift) {
                const int64_t position = t - shift;
                int64_t mapped = pad_id;
                if (position >= cu[r]) {
                    blocked = blocked || (mask && !mask[position]);
                    if (!blocked) mapped = mapping[ids[position]];
                } else {
                    const int64_t back = cu[r] - position - 1;
                    if (back < starts[r]) {
                        const int64_t index = r * (ngram - 1) + back;
                        blocked = blocked || (prior_mask && !prior_mask[index]);
                        if (!blocked) mapped = mapping[prior[index]];
                    }
                }
                window[shift] = static_cast<uint64_t>(mapped);
            }
            for (int64_t l = 0; l < layers; ++l) {
                uint64_t hash = 0;
                for (int64_t shift = 0; shift < ngram; ++shift) {
                    hash ^= window[shift] * static_cast<uint64_t>(mult[l * ngram + shift]);
                    rolling[shift] = hash;
                }
                for (int64_t h = 0; h < heads; ++h) {
                    const int64_t index = selected[l * heads + h];
                    const int64_t order = index / heads_per_order + 1;
                    int64_t signed_hash;
                    std::memcpy(&signed_hash, &rolling[order], sizeof(signed_hash));
                    const int64_t prime = moduli[l * (ngram - 1) * heads_per_order + index];
                    int64_t bucket = signed_hash % prime;
                    if (bucket < 0) bucket += prime;
                    const int64_t row = offsets[l * heads + h] + bucket;
                    std::memcpy(output_data[l] + (t * heads + h) * widths[l],
                                table_data[l] + row * widths[l], widths[l] * sizeof(at::BFloat16));
                }
            }
        }
    });
    for (int64_t l = 0; l < layers; ++l) {
        if (bucket_tokens > tokens) {
            std::memset(output_data[l] + tokens * heads * widths[l], 0,
                        (bucket_tokens - tokens) * heads * widths[l] * sizeof(at::BFloat16));
        }
    }
}
} // namespace
} // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("engram_hash_gather_cpu(Tensor input_ids, Tensor query_start_loc, Tensor start_positions, "
          "Tensor lookback_ids, Tensor token_map, Tensor multipliers, Tensor primes, Tensor head_indices, "
          "Tensor local_starts, Tensor[] tables, Tensor(a!)[] outputs, int pad_id, int bucket_tokens, "
          "Tensor? token_mask=None, Tensor? lookback_mask=None) -> ()");
    m.impl("engram_hash_gather_cpu", c10::DispatchKey::CPU, &vllm_ascend::EngramHashGatherCPU);
}
