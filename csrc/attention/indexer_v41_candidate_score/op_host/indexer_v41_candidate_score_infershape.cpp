// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeScore(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }
static ge::graphStatus InferDataTypeScore(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(IndexerV41CandidateScore).InferShape(InferShapeScore).InferDataType(InferDataTypeScore);
}  // namespace ops
