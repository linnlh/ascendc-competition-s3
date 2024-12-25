
#include <algorithm>

#include "log_sum_exp_tiling.h"
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

    LogSumExpTilingData tiling;
    auto shape = context->GetInputShape(0)->GetStorageShape();
    int64_t shapeDim = shape.GetDimNum();
    int64_t shapePtr[10];
    for (int i = 0; i < shapeDim; i++) {
        shapePtr[i] = shape.GetDim(i);
    }

    int64_t totalLen = shape.GetShapeSize();

    auto attrs = context->GetAttrs();
    // TODO: 当前仅能处理 dimSize < 2 的情况
    auto dim = attrs->GetListInt(0);
    int64_t dimSize = dim->GetSize();
    const int64_t* dimConstPtr = dim->GetData();
    int64_t tileNum = totalLen;
    int64_t dimPtr[10];
    int64_t stridesPtr[10];
    for (int i = 0; i < dimSize; i++) {
        dimPtr[i] = dimConstPtr[i];
        tileNum /= shape.GetDim(dimConstPtr[i]);
        stridesPtr[i] = totalLen;
        for (int j = 0; j < dimPtr[i] + 1; j++) {
            stridesPtr[i] /= shapePtr[j];
        }
    }
    // 合并连续的轴
    // std::sort(dimPtr, dimPtr + dimSize);
    // std::sort(stridesPtr, stridesPtr + dimSize, std::greater<int>());
    // int index = 0;
    // while (index < dimSize - 1) {
    //     int64_t axis = dimPtr[index];
    //     if (axis + 1 == dimPtr[index + 1]) {
    //         stridesPtr[index] = stridesPtr[index + 1];
    //         for (int j = index + 1; j < dimSize - 1; j++) {
    //             dimPtr[j] = dimPtr[j + 1] - 1;
    //             stridesPtr[j] = stridesPtr[j + 1];
    //         }
    //         shapePtr[axis] *= shapePtr[axis + 1];
    //         for (int j = axis + 1; j < shapeDim - 1; j++) {
    //             shapePtr[j] = shapePtr[j + 1];
    //         }
    //         dimSize -= 1;
    //         shapeDim -= 1;
    //     }
    //     else {
    //         index++;
    //     }
    // }
    if (dimSize == 2 && dimPtr[0] > dimPtr[1]) {
        std::swap<int64_t>(dimPtr[0], dimPtr[1]);
        std::swap<int64_t>(stridesPtr[0], stridesPtr[1]);
    }
    int64_t tileLen = shapePtr[dimPtr[dimSize - 1]];

    int64_t tilingKey = 1;
    if (dimSize == 2 && dimPtr[1] != shapeDim - 1) {
        tileLen = 1;
        tilingKey = 2;
    }

    tiling.set_tileLen(tileLen);
    tiling.set_tileNum(tileNum);
    tiling.set_shape(shapePtr);
    tiling.set_shapeLen(shapeDim);
    tiling.set_dim(dimPtr);
    tiling.set_strides(stridesPtr);
    tiling.set_dimLen(dimSize);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->SetTilingKey(tilingKey);
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
class LogSumExp : public OpDef {
public:
    explicit LogSumExp(const char* name) : OpDef(name)
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
        this->Attr("dim").AttrType(OPTIONAL).ListInt({0});
        this->Attr("keep_dim").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(LogSumExp);
}
