// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class V41DsparkMetadata : public OpDef {
public:
    explicit V41DsparkMetadata(const char *name) : OpDef(name)
    {
        for (const char *input : {"cu_q", "seqused_kv", "topk_lengths", "schedule"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(V41DsparkMetadata);
}  // namespace ops
