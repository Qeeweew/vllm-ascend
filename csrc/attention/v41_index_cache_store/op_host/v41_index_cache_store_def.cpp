// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class V41IndexCacheStore : public OpDef {
public:
    explicit V41IndexCacheStore(const char *name) : OpDef(name)
    {
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positions").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("slots").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sin").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_cache").ParamType(REQUIRED).DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("scale_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Attr("compress_ratio").Int();
        this->Attr("key_stride0").Int();
        this->Attr("scale_stride0").Int();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(V41IndexCacheStore);
}  // namespace ops
