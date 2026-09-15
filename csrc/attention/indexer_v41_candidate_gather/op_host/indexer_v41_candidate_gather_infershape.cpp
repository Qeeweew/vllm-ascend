// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeGather(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }
static ge::graphStatus InferDataTypeGather(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(IndexerV41CandidateGather).InferShape(InferShapeGather).InferDataType(InferDataTypeGather);
}  // namespace ops
