
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

    NotEqualTilingData tiling;
    int64_t dataLen = context->GetInputTensor(0)->GetShapeSize();
    int64_t tileSize = (ubSize - 512) / (2 * BUFFER_NUM * dtypeSize + BUFFER_NUM + 11);
    int64_t tileLen = tileSize / 256 * 256 / dtypeSize;
    // int64_t tileLen = 128;
    if (tileLen > dataLen) {
        tileLen = dataLen;
    }
    int64_t tileNum = dataLen / tileLen;
    int64_t tailTileLen = dataLen % tileLen;

    auto x1Shape = context->GetInputShape(0)->GetStorageShape();
    auto x2Shape = context->GetInputShape(1)->GetStorageShape();
    int64_t x1ShapePtr[10];
    int64_t x2ShapePtr[10];
    int64_t dimNum = x1Shape.GetDimNum();
    int64_t tilingKey = 1;
    for (int i = 0; i < dimNum; i++) {
        x1ShapePtr[i] = x1Shape.GetDim(i);
        x2ShapePtr[i] = x2Shape.GetDim(i);
        if (x1ShapePtr[i] != x2ShapePtr[i])
            tilingKey = 2;
    }

    // printf("data length: %ld\n", dataLen);
    // printf("tile length: %ld\n", tileLen);
    // printf("tile number: %ld\n", tileNum);
    // printf("tail length: %ld\n", tailTileLen);
    // printf("tilingKey: %ld\n", tilingKey);
    tiling.set_x1Shape(x1ShapePtr);
    tiling.set_x2Shape(x2ShapePtr);
    tiling.set_dimNum(dimNum);
    tiling.set_tileLen(tileLen);
    tiling.set_tileNum(tileNum);
    tiling.set_tailTileLen(tailTileLen);

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
