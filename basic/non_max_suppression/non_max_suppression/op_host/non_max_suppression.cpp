
#include "non_max_suppression_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace platform_ascendc;
using namespace AscendC;

constexpr int BLOCK_DIM = 1;
constexpr int BLOCK_SIZE = 32;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = PlatformAscendC(context->GetPlatformInfo());
    uint32_t dataTypeSize = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(),
        dataTypeSize
    );

    auto scoreShape = context->GetInputShape(1)->GetStorageShape();
    int64_t batchSize = scoreShape.GetDim(0);
    int64_t classNum = scoreShape.GetDim(1);
    int64_t boxNum = scoreShape.GetDim(2);
    int64_t boxNumAlign32 = (boxNum * dataTypeSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / dataTypeSize;

    NonMaxSuppressionTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_classNum(classNum);
    tiling.set_boxNum(boxNum);

    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
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
class NonMaxSuppression : public OpDef {
public:
    explicit NonMaxSuppression(const char* name) : OpDef(name)
    {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("max_output_boxes_per_class")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("iou_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("score_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("selected_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("cneter_point_box").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(NonMaxSuppression);
}
