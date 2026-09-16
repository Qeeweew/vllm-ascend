// SPDX-License-Identifier: Apache-2.0
#ifndef V41_ROPE_TILING_H
#define V41_ROPE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(V41RopeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, heads);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, tableRows);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(V41Rope, V41RopeTilingData)
}  // namespace optiling
#endif
