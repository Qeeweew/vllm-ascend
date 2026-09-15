// SPDX-License-Identifier: Apache-2.0
#ifndef ENGRAM_GATE_TILING_H
#define ENGRAM_GATE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(EngramGateTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tokens);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
    TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EngramGate, EngramGateTilingData)
}  // namespace optiling
#endif
