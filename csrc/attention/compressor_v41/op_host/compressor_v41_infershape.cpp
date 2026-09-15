// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeCompressorV41(gert::InferShapeContext *)
{
    return ge::GRAPH_SUCCESS;  // Both results are caller-owned input buffers.
}
static ge::graphStatus InferDataTypeCompressorV41(gert::InferDataTypeContext *)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(CompressorV41)
    .InferShape(InferShapeCompressorV41)
    .InferDataType(InferDataTypeCompressorV41);
}  // namespace ops
