// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeV41IndexCacheStore(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }
static ge::graphStatus InferTypeV41IndexCacheStore(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(V41IndexCacheStore).InferShape(InferShapeV41IndexCacheStore).InferDataType(InferTypeV41IndexCacheStore);
}  // namespace ops
