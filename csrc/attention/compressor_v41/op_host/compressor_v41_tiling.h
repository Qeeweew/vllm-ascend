// SPDX-License-Identifier: Apache-2.0
#ifndef COMPRESSOR_V41_HOST_TILING_H
#define COMPRESSOR_V41_HOST_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CompressorV41TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tokens);
    TILING_DATA_FIELD_DEF(uint32_t, requests);
    TILING_DATA_FIELD_DEF(uint32_t, capacity);
    TILING_DATA_FIELD_DEF(uint32_t, stateBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
    TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CompressorV41, CompressorV41TilingData)
}  // namespace optiling
#endif
