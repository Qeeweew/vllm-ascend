// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {
class EngramGate : public OpDef {
public:
    explicit EngramGate(const char *name) : OpDef(name)
    {
        for (const char *input : {"hidden", "kv", "q_weight", "k_weight"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Input("token_mask").ParamType(REQUIRED).DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        // Caller-owned destination permits stable-address graph replay.
        this->Input("output").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("eps").Float();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(EngramGate);
}  // namespace ops
