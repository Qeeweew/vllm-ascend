#include "register/op_def_registry.h"

namespace ops {
class W4a16Moe : public OpDef {
public:
    explicit W4a16Moe(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w13").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w13_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w2").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w2_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("expert_ids").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("topk_weights").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("swiglu_limit").AttrType(OPTIONAL).Float(0.0f);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(W4a16Moe);
} // namespace ops
