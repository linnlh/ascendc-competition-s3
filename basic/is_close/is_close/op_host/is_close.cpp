
#include "is_close_tiling.h"
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
    auto dataType = context->GetInputDesc(0)->GetDataType();
    ge::TypeUtils::GetDataTypeLength(dataType, dtypeSize);

    IsCloseTilingData tiling;
    int64_t dataLen = context->GetInputTensor(0)->GetShapeSize();
    int64_t tileSize = (ubSize - 128) / (2 * BUFFER_NUM * 4 + BUFFER_NUM + 5);
    if (dataType == ge::DT_INT32) {
        tileSize = (ubSize - 128) / (2 * BUFFER_NUM * 4 + BUFFER_NUM + 13);
    }
    int64_t tileLen = tileSize / 64 * 64;
    // int64_t tileLen = 128;
    if (tileLen > dataLen) {
        tileLen = dataLen;
    }
    int64_t tileNum = dataLen / tileLen;
    int64_t tailTileLen = dataLen % tileLen;

    auto attrs = context->GetAttrs();
    const float* rtol = attrs->GetFloat(0);
    const float* atol = attrs->GetFloat(1);
    const bool* equalNan = attrs->GetBool(2);

    auto x1Shape = context->GetInputShape(0)->GetStorageShape();
    auto x2Shape = context->GetInputShape(1)->GetStorageShape();
    int64_t x1ShapePtr[10];
    int64_t x2ShapePtr[10];
    int64_t dimNum1 = x1Shape.GetDimNum();
    int64_t dimNum2 = x2Shape.GetDimNum();
    int64_t dimNum = std::max(dimNum1, dimNum2);
    if (dimNum1 >= dimNum2) {
        for (int i = 0; i < (dimNum1 - dimNum2); i++) {
            x1ShapePtr[i] = x1Shape[i];
            x2ShapePtr[i] = 1;
        }
        for (int i = (dimNum1 - dimNum2); i < dimNum1; i++) {
            x1ShapePtr[i] = x1Shape[i];
            x2ShapePtr[i] = x2Shape[i - dimNum1 + dimNum2];
        }
    }
    else {
        for (int i = 0; i < (dimNum2 - dimNum1); i++) {
            x1ShapePtr[i] = 1;
            x2ShapePtr[i] = x2Shape[i];
        }
        for (int i = (dimNum2 - dimNum1); i < dimNum2; i++) {
            x1ShapePtr[i] = x1Shape[i - dimNum2 + dimNum1];
            x2ShapePtr[i] = x2Shape[i];
        }
    }
    int64_t tilingKey = 1;
    for (int i = 0; i < dimNum; i++) {
        if (x1ShapePtr[i] != x2ShapePtr[i])
            tilingKey = 2;
    }

    tiling.set_x1Shape(x1ShapePtr);
    tiling.set_x2Shape(x2ShapePtr);
    tiling.set_dimNum(dimNum);
    tiling.set_tileLen(tileLen);
    tiling.set_tileNum(tileNum);
    tiling.set_tailTileLen(tailTileLen);
    tiling.set_rtol(*rtol);
    tiling.set_atol(*atol);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(BLOCK_DIM);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

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
class IsClose : public OpDef {
public:
    explicit IsClose(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("rtol").AttrType(OPTIONAL).Float(1e-05);
        this->Attr("atol").AttrType(OPTIONAL).Float(1e-08);
        this->Attr("equal_nan").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(IsClose);
}
