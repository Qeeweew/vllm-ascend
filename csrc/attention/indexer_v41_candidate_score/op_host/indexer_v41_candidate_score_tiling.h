// SPDX-License-Identifier: Apache-2.0
#ifndef INDEXER_V41_CANDIDATE_SCORE_TILING_H
#define INDEXER_V41_CANDIDATE_SCORE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(IndexerV41CandidateScoreTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, positions);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IndexerV41CandidateScore, IndexerV41CandidateScoreTilingData)
}  // namespace optiling
#endif
