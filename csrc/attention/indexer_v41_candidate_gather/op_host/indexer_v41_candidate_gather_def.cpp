// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class IndexerV41CandidateGather : public OpDef {
public:
    explicit IndexerV41CandidateGather(const char *name) : OpDef(name)
    {
        this->Input("key_cache").ParamType(REQUIRED).DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("key_scale_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("sorted_blocks").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_table").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("seqused_k").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cu_seqlens_q").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gathered_key").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gathered_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positions").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("key_stride0").Int();
        this->Attr("key_scale_stride0").Int();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(IndexerV41CandidateGather);
}  // namespace ops
