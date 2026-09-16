// SPDX-License-Identifier: Apache-2.0
#ifndef V41_MAIN_CACHE_STORE_TILING_H
#define V41_MAIN_CACHE_STORE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(V41MainCacheStoreTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tokens);
    TILING_DATA_FIELD_DEF(uint32_t, page);
    TILING_DATA_FIELD_DEF(uint32_t, blocks);
    TILING_DATA_FIELD_DEF(uint32_t, tableRows);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
    TILING_DATA_FIELD_DEF(uint32_t, ratio);
    TILING_DATA_FIELD_DEF(uint64_t, cacheStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(V41MainCacheStore, V41MainCacheStoreTilingData)
}  // namespace optiling
#endif
