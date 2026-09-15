// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class IndexerV41CandidateScore : public OpDef {
public:
    explicit IndexerV41CandidateScore(const char *name) : OpDef(name)
    {
        this->Input("qk").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gathered_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positions").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(IndexerV41CandidateScore);
}  // namespace ops
