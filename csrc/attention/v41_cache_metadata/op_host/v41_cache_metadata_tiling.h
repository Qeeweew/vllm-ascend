// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(V41CacheMetadataTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, tokens);
    TILING_DATA_FIELD_DEF(uint32_t, inputColumns);
    TILING_DATA_FIELD_DEF(uint32_t, outputColumns);
    TILING_DATA_FIELD_DEF(uint32_t, logicalBlock);
    TILING_DATA_FIELD_DEF(uint32_t, physicalBlock);
    TILING_DATA_FIELD_DEF(uint32_t, ratio);
    TILING_DATA_FIELD_DEF(uint32_t, compressed);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(V41CacheMetadata, V41CacheMetadataTilingData)
}
