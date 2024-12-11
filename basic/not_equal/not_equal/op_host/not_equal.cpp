
#include "not_equal_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

using namespace platform_ascendc;

constexpr int BLOCK_DIM = 1;
constexpr int BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platform.GetCoreMemSize(CoreMemType::UB, ubSize);
    
    uint32_t dtypeSize = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(),
        dtypeSize
    );

    int64_t dataLength = context->GetInputTensor(0)->GetShapeSize();
    int64_t tileLength = 2048;
    if (tileLength > dataLength) {
        tileLength = dataLength;
    }
    int64_t tileNum = dataLength / tileLength;
    int64_t tailLength = dataLength % tileLength;
    // int64_t tileSize = ubSize / ((2 * dtypeSize + 1) * BUFFER_NUM + 3);
    // int64_t tileLength = tileSize / 256 * 256 / dtypeSize;
    // if (tileLength > dataLength) {
    //     tileLength = dataLength;
    // }
    // int64_t tileNum = dataLength / tileLength;
    // int64_t tailLength = dataLength % tileLength;
    // printf("data length: %ld\n", dataLength);
    // printf("tile length: %ld\n", tileLength);
    // printf("tile number: %ld\n", tileNum);
    // printf("tail length: %ld\n", tailLength);

    NotEqualTilingData tiling;
    tiling.set_tileLength(tileLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tailLength(tailLength);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->SetBlockDim(BLOCK_DIM);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class NotEqual : public OpDef {
public:
    explicit NotEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(NotEqual);
}
