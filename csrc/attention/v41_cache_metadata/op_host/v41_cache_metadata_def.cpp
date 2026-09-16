// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class V41CacheMetadata : public OpDef {
public:
    explicit V41CacheMetadata(const char *name) : OpDef(name)
    {
        for (const char *input : {"positions_in", "cu_q_in", "lengths_in", "table_in", "positions_out",
                                  "cu_q_out", "lengths_out", "table_out", "requests_out", "slots_out",
                                  "cmp_lengths_out", "residual_out"}) {
            const bool wide = std::string(input) == "positions_in" || std::string(input) == "positions_out" ||
                              std::string(input) == "slots_out";
            this->Input(input).ParamType(REQUIRED).DataType({wide ? ge::DT_INT64 : ge::DT_INT32})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Attr("logical_block_size").AttrType(REQUIRED).Int();
        this->Attr("physical_block_size").AttrType(REQUIRED).Int();
        this->Attr("compress_ratio").AttrType(REQUIRED).Int();
        this->Attr("compressed").AttrType(REQUIRED).Bool();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(V41CacheMetadata);
}
