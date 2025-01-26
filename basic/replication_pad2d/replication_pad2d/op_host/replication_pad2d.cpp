
#include "replication_pad2d_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {

using namespace platform_ascendc;

constexpr int BLOCK_DIM = 1;
constexpr int BUFFER_NUM = 2;

int64_t RoundUp32(int64_t val) {
    return (val + 31) / 32 * 32;
}

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
    int64_t elemPerBlk = 32 / dtypeSize;

    ReplicationPad2dTilingData tiling;
    auto paddings = context->GetInputTensor(1)->GetData<int32_t>();
    auto shape = context->GetInputShape(0)->GetStorageShape();
    int64_t totalLen = context->GetInputTensor(0)->GetShapeSize();
    int64_t width = shape.GetDim(shape.GetDimNum() - 1);
    int64_t height = shape.GetDim(shape.GetDimNum() - 2);
    int64_t batch = totalLen / (width * height);

    int64_t leftPaddingAlign32 = RoundUp32(paddings[0] * dtypeSize);
    int64_t rightPaddingAlign32 = RoundUp32(paddings[1] * dtypeSize);
    int64_t widthAlign32 = RoundUp32(width * dtypeSize);
    int64_t tileLen = (ubSize - 128) / ((widthAlign32 + leftPaddingAlign32 + rightPaddingAlign32) * BUFFER_NUM);
    if (tileLen > batch) {
        tileLen = batch;
    }
    int64_t tileNum = batch / tileLen;
    int64_t tailTileLen = batch % tileLen;
    tiling.set_width(width);
    tiling.set_height(height);
    tiling.set_tileLen(tileLen);
    tiling.set_tileNum(tileNum);
    tiling.set_tailTileLen(tailTileLen);

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
class ReplicationPad2d : public OpDef {
public:
    explicit ReplicationPad2d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ReplicationPad2d);
}
