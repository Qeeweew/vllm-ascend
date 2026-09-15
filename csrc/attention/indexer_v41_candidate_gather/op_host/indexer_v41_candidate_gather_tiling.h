// SPDX-License-Identifier: Apache-2.0
#ifndef INDEXER_V41_CANDIDATE_GATHER_TILING_H
#define INDEXER_V41_CANDIDATE_GATHER_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(IndexerV41CandidateGatherTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, positions);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
    TILING_DATA_FIELD_DEF(uint32_t, pageSize);
    TILING_DATA_FIELD_DEF(uint32_t, pages);
    TILING_DATA_FIELD_DEF(uint32_t, tablePages);
    TILING_DATA_FIELD_DEF(uint32_t, reserved);
    TILING_DATA_FIELD_DEF(uint64_t, keyStride);
    TILING_DATA_FIELD_DEF(uint64_t, scaleStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IndexerV41CandidateGather, IndexerV41CandidateGatherTilingData)
}  // namespace optiling
#endif
