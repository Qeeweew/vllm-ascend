// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeV41Rope(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }
static ge::graphStatus InferTypeV41Rope(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(V41Rope).InferShape(InferShapeV41Rope).InferDataType(InferTypeV41Rope);
}  // namespace ops
