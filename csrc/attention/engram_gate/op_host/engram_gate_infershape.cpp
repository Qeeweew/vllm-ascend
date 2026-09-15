// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeEngramGate(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }
static ge::graphStatus InferDataTypeEngramGate(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(EngramGate).InferShape(InferShapeEngramGate).InferDataType(InferDataTypeEngramGate);
}  // namespace ops
