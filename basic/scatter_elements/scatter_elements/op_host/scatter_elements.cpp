
#include "scatter_elements_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {

using namespace platform_ascendc;

constexpr int BLOCK_DIM = 1;
constexpr int BUFFER_NUM = 1;

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

    ScatterElementsTilingData tiling;
    auto attrs = context->GetAttrs();
    const int64_t* axis = attrs->GetInt(0);
    const char* reduce = attrs->GetStr(1);
    int64_t mode = 0;
    if (reduce != nullptr && strcmp(reduce, "add") == 0) {
        mode = 1;
    }
    else if (reduce != nullptr && strcmp(reduce, "mutiply") == 0) {
        mode = 2;
    }

    auto varShape = context->GetInputShape(0)->GetStorageShape();
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();
    int64_t varLen = varShape.GetShapeSize();
    int64_t indicesLen = indicesShape.GetShapeSize();
    
    int64_t varStride = varLen;
    int64_t indiceStride = indicesLen;
    for (int i = 0; i < (*axis + 1); i++) {
        varStride /= varShape.GetDim(i);
        indiceStride /= indicesShape.GetDim(i);
    }
    uint16_t varShapePtr[10], indicesShapePtr[10];
    for (int i = 0; i < varShape.GetDimNum(); i++) {
        varShapePtr[i] = varShape[i];
        indicesShapePtr[i] = indicesShape[i];
    }
    // int64_t varStride = varLen / varShape.GetDim(0);
    // int64_t updateStride = indicesLen / indicesShape.GetDim(0);


    int64_t tileLength = varShape.GetDim(*axis);
    // if (tileLength > indicesLen) {
    //     tileLength = indicesLen;
    // }
    int64_t varTileLen = varShape.GetDim(*axis);
    int64_t indiceTileLen = indicesShape.GetDim(*axis);
    // int64_t indiceTileLen = indicesShape.GetDim(indicesShape.GetDimNum() - 1);
    // int64_t updateTileLen = indiceTileLen;
    int64_t tileNum = indicesLen / indiceTileLen;
    int64_t tailTileLength = 0;
    // printf("[tileNum]: %ld\n", tileNum);
    // printf("[varLen]: %ld\n", varLen);
    // printf("[indicesLen]: %ld\n", indicesLen);
    // printf("[varTileLen]: %ld\n", varTileLen);
    // printf("[indiceTileLen]: %ld\n", indiceTileLen);
    // printf("[varStride]: %ld\n", varStride);
    // printf("[indiceStride]: %ld\n", indiceStride);
    // printf("[mode] %ld\n", mode);

    tiling.set_tileLength(tileLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tailTileLength(tailTileLength);
    tiling.set_varLen(varLen);
    tiling.set_indicesLen(indicesLen);
    tiling.set_varStride(varStride);
    tiling.set_indiceStride(indiceStride);
    tiling.set_mode(mode);
    tiling.set_varTileLen(varTileLen);
    tiling.set_indiceTileLen(indiceTileLen);
    tiling.set_varShape(varShapePtr);
    tiling.set_varShapeLen(varShape.GetDimNum());
    tiling.set_indicesShape(indicesShapePtr);
    tiling.set_indicesShapeLen(indicesShape.GetDimNum());
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
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("reduce").AttrType(OPTIONAL).String("None");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ScatterElements);
}
