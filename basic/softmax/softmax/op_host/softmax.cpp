
#include "softmax_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace platform_ascendc;
using namespace AscendC;

constexpr int BLOCK_DIM = 1;
constexpr int BLK_SIZE = 32;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = PlatformAscendC(context->GetPlatformInfo());
    uint32_t dataTypeSize = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(),
        dataTypeSize
    );

    int64_t totalSize = context->GetInputTensor(0)->GetShapeSize();
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    int64_t width = xShape.GetDim(xShape.GetDimNum() - 1);
    int64_t widthAlign32 = (width * dataTypeSize + BLK_SIZE - 1) / BLK_SIZE * BLK_SIZE / dataTypeSize;
    int64_t height = totalSize / width;

    SoftmaxTilingData tiling;
    tiling.set_width(width);
    tiling.set_widthAlign32(widthAlign32);
    tiling.set_height(height);
    ge::Shape srcShape({1, widthAlign32});
    const uint32_t localWorkSpaceSize = GetSoftMaxMinTmpSize(
        srcShape, dataTypeSize, false
    );
    SoftMaxTilingFunc(
        srcShape, dataTypeSize, localWorkSpaceSize, tiling.softmaxTilingData
    );

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
class Softmax : public OpDef {
public:
    explicit Softmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Softmax);
}
