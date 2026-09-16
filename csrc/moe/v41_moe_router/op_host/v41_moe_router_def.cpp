// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class V41MoeRouter : public OpDef {
public:
    explicit V41MoeRouter(const char *name) : OpDef(name)
    {
        this->Input("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("token_ids").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("image_mask").ParamType(REQUIRED).DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tid2eid").ParamType(OPTIONAL).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("text_bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("image_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("expert_ids").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("top_k").AttrType(OPTIONAL).Int(6);
        this->Attr("renormalize").AttrType(OPTIONAL).Bool(true);
        this->Attr("routed_scaling_factor").AttrType(OPTIONAL).Float(1.0f);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(V41MoeRouter);
}  // namespace ops
