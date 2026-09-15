# SPDX-License-Identifier: Apache-2.0
"""Worker-only auxiliary-state observation; import after Ascend plugin setup."""

import torch
from smoke_mm_runner_worker import V41MMSmokeWorker


class V41AuxProbeWorker(V41MMSmokeWorker):
    def load_model(self):
        super().load_model()
        target = self.model_runner.get_model()
        backbone = target.get_language_model().model
        self.aux_router_types = [type(layer.mlp.experts.routed_experts.router).__name__ for layer in backbone.layers]
        if any(name != "AscendFusedTopKRouter" for name in self.aux_router_types):
            raise RuntimeError(f"Unexpected V4.1 router before auxiliary probe: {self.aux_router_types}")
        if len(backbone.layers) != 3:
            raise ValueError("This probe requires the bounded real three-layer target fixture")
        target.set_aux_hidden_state_layers((1, 2, 3))
        self.model_runner.use_aux_hidden_state_outputs = True
        self.aux_errors = {
            mode: torch.zeros((3,), device=self.device, dtype=torch.float32) for mode in ("eager", "graph")
        }
        self.aux_calls = {mode: torch.zeros((), device=self.device, dtype=torch.int32) for mode in self.aux_errors}
        expected = [None] * 3

        def remember_layer(module, args, output, index):
            hidden = output[0]
            # Explicit FP32 four-stream average is independent of the model's
            # BF16 mean operation. Materialize before the next Engram injection.
            expected[index] = (
                (hidden[:, 0].float() + hidden[:, 1].float() + hidden[:, 2].float() + hidden[:, 3].float()) * 0.25
            ).to(hidden.dtype)

        for index, layer in enumerate(backbone.layers):
            layer.register_forward_hook(
                lambda module, args, output, index=index: remember_layer(module, args, output, index)
            )

        def check_outputs(module, args, output):
            hidden, auxiliary = output
            if len(auxiliary) != 3 or any(value.shape != hidden.shape for value in auxiliary):
                raise AssertionError("Target must export three token-aligned HC means")
            mode = "graph" if torch.npu.is_current_stream_capturing() else "eager"
            self.aux_calls[mode].add_(1)
            for index, value in enumerate(auxiliary):
                error = (value.float() - expected[index].float()).abs().max()
                self.aux_errors[mode][index].copy_(torch.maximum(self.aux_errors[mode][index], error))

        backbone.register_forward_hook(check_outputs)

    def reset_aux_probe(self):
        # Clear capture/warmup results so only actual requests can pass.
        for error in self.aux_errors.values():
            error.zero_()
        for count in self.aux_calls.values():
            count.zero_()
        torch.npu.synchronize()
        return {"rank": self.rank, "aux_layers": [1, 2, 3]}

    def inspect_aux_probe(self):
        return {
            "rank": self.rank,
            "router_types": self.aux_router_types,
            "max_abs_error": {key: value.cpu().tolist() for key, value in self.aux_errors.items()},
            "calls": {key: value.cpu().item() for key, value in self.aux_calls.items()},
        }
