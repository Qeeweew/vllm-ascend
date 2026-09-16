# V4.1 router integration handoff

Status: CPU contract tests and complete isolated package build r3 pass; native
accuracy and graph checks passed (84 tests) with the new workspace-safe binding.
Performance is not yet accepted. See `package_r3.json` and `RESULT.md`.
Do not enable production dispatch before the baseline probe, exact ID/weight,
changed-input graph and frozen whole-chain performance gates all pass.

## Build and native API

Add `v41_moe_router` to the root-owned complete CANN build list. Native autogen
API is `aclnnV41MoeRouter`; the OpDef has eight tensor inputs (two optional) and
three attributes, with caller-owned outputs represented as mutating inputs.
There is zero explicit user scratch; report CANN fixed workspace separately.

Torch binding signature and launch, inside the same custom-op build guard as
EngramGate:

```cpp
void v41_moe_router(const at::Tensor &logits, const at::Tensor &token_ids,
                   const at::Tensor &image_mask,
                   const c10::optional<at::Tensor> &tid2eid,
                   const c10::optional<at::Tensor> &text_bias,
                   const at::Tensor &image_bias, at::Tensor &weights,
                   at::Tensor &expert_ids, int64_t top_k,
                   bool renormalize, double routed_scaling_factor)
{
    // Repeat the metadata checks in validate_v41_moe_router in this binding:
    // dtype/shape/device/contiguity and no output overlap, including optional
    // inputs. Use ATen/MemoryOverlap.h assert_no_overlap for outputs vs inputs.
    // Require PrivateUse1; top_k=6/E384 or top_k=3/E128; finite scale.
    if (logits.size(0) == 0) { return; }
    EXEC_NPU_CMD(aclnnV41MoeRouter, logits, token_ids, image_mask, tid2eid,
                 text_bias, image_bias, weights, expert_ids, top_k,
                 renormalize, routed_scaling_factor);
}
```

Use the adapter's established optional-tensor conversion. A None optional
must reach the generated optional-input API, not an arbitrary tensor shape.

```cpp
ops.def("v41_moe_router(Tensor logits, Tensor token_ids, Tensor image_mask, "
        "Tensor? tid2eid, Tensor? text_bias, Tensor image_bias, "
        "Tensor(a!) weights, Tensor(b!) expert_ids, int top_k=6, "
        "bool renormalize=True, float routed_scaling_factor=1.0) -> ()");
ops.impl("v41_moe_router", torch::kPrivateUse1, &vllm_ascend::v41_moe_router);
```

Meta implementation returns `None` and repeats the shape/dtype checks without
reading values or device memory. Mutation annotations describe both outputs.

## Model integration

`select_deepseek_v4_vision_experts` remains the independent baseline. The fused
caller must pass the actual explicit BOOL image mask, never reconstruct one
from token IDs; this preserves literal image sentinel text and padding token 0.
Keep the caller's FP32 logits and FP32 biases. Output IDs are INT32, with hash
rows in exact table order. Allocate outputs outside graph capture where the
caller uses persistent buffers. No `.item()` or device-side eligibility check.

Valid native inputs have finite logits/bias and legal text lookup token/expert
IDs. For memory safety only, invalid text token or any invalid selected table
expert writes the entire row as weights=0 / expert_ids=-1. Such rows must never
be passed to MoE execution. Image rows do not load their token or table row.

## Numerical contract after baseline probe

`baseline_contract_r1.json` freezes installed NPU topk ties as ascending expert
index for all 24 tested shape/pattern combinations, repeated three times each.
The independent scalar oracle uses FP32-rounded log1p(exp(x)) and threshold 20.
The actual NPU baseline preserves negative tails, whereas the old generic
AscendC Exp/Adds/Ln does not. The new candidate uses compensated log1p with a
tiny-input branch; native validation is still pending. All IDs must be exact;
weight gate is rtol=2e-6 / atol=2e-7, with adversarial cutoff IDs checked
independently. No recall-only selection gate is allowed.
