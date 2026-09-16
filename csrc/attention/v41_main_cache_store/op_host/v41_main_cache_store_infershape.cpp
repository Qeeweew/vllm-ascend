// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeV41MainCacheStore(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }
static ge::graphStatus InferTypeV41MainCacheStore(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(V41MainCacheStore).InferShape(InferShapeV41MainCacheStore).InferDataType(InferTypeV41MainCacheStore);
}  // namespace ops
