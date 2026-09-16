// SPDX-License-Identifier: Apache-2.0
#ifndef V41_DSPARK_METADATA_TILING_H
#define V41_DSPARK_METADATA_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(V41DsparkMetadataTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, tokens);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(V41DsparkMetadata, V41DsparkMetadataTilingData)
}  // namespace optiling
#endif
