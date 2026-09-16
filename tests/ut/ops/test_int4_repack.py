# SPDX-License-Identifier: Apache-2.0
"""CPU validation must reject unsupported layouts before touching the device."""

import pytest
import torch

from vllm_ascend.ops.triton.int4_repack import repack_int4_moe


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((8, 8), torch.int32),
        ((2, 8, 8), torch.int8),
        ((2, 7, 8), torch.int32),
        ((0, 8, 8), torch.int32),
        ((2, 0, 8), torch.int32),
        ((2, 8, 0), torch.int32),
    ],
)
def test_invalid_repack_inputs_fail_before_npu(shape, dtype):
    with pytest.raises(ValueError):
        repack_int4_moe(torch.empty(shape, dtype=dtype))
    assert not torch.npu.is_initialized()
