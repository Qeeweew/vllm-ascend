// SPDX-License-Identifier: Apache-2.0
#ifndef V41_MOE_ROUTER_TILING_H
#define V41_MOE_ROUTER_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(V41MoeRouterTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, experts);
    TILING_DATA_FIELD_DEF(uint32_t, topK);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
    TILING_DATA_FIELD_DEF(uint32_t, vocabulary);
    TILING_DATA_FIELD_DEF(uint32_t, hasTextBias);
    TILING_DATA_FIELD_DEF(uint32_t, renormalize);
    TILING_DATA_FIELD_DEF(float, scaling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(V41MoeRouter, V41MoeRouterTilingData)
}  // namespace optiling
#endif
