#ifndef W4A16_MOE_TILING_H
#define W4A16_MOE_TILING_H
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(W4A16MoeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch_size);
TILING_DATA_FIELD_DEF(uint32_t, hidden_size);
TILING_DATA_FIELD_DEF(uint32_t, inter_size);
TILING_DATA_FIELD_DEF(uint32_t, num_experts);
TILING_DATA_FIELD_DEF(uint32_t, top_k);
TILING_DATA_FIELD_DEF(float, swiglu_limit);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(W4a16Moe, W4A16MoeTilingData)
} // namespace optiling
#endif
