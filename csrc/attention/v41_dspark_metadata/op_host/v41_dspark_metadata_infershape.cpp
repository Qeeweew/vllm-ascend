// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }
static ge::graphStatus InferType(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(V41DsparkMetadata).InferShape(InferShape).InferDataType(InferType);
}  // namespace ops
