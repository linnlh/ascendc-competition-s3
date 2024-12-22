
#include "nll_loss_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

using namespace platform_ascendc;

constexpr int BLOCK_DIM = 1;
constexpr int BUFFER_NUM = 2;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platform.GetCoreMemSize(CoreMemType::UB, ubSize);
    auto coreNum = platform.GetCoreNum();
    // int64_t coreNum = 1;
    uint32_t dtypeSize = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(),
        dtypeSize
    );

    NLLLossTilingData tiling;
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalSize = xShape.GetShapeSize();
    uint32_t clsNum = xShape.GetDim(xShape.GetDimNum() - 1);
    uint32_t clsNumAlign8 = (clsNum + 7) / 8 * 8;
    uint32_t batchSize = totalSize / clsNum;
    if (batchSize < coreNum) {
        coreNum = batchSize;
    }
    uint32_t tileLen = (ubSize - coreNum * 8 * BUFFER_NUM - (clsNumAlign8 * 4)) / (12 + (4 * BUFFER_NUM) + (4 * clsNumAlign8 * BUFFER_NUM)) / 8 * 8;
    uint32_t remainBatch = batchSize % coreNum;
    uint32_t bigCoreNum = (remainBatch == 0) ? coreNum : remainBatch;
    tiling.set_bigCoreNum(bigCoreNum);

    uint32_t smallCoreBatchSize = batchSize / coreNum;
    uint32_t smallCoreTileLen = tileLen;
    if (smallCoreTileLen > smallCoreBatchSize) {
        smallCoreTileLen = smallCoreBatchSize;
    }
    uint32_t smallCoreTileNum = smallCoreBatchSize / smallCoreTileLen;
    uint32_t smallCoreTailTileLen = smallCoreBatchSize % smallCoreTileLen;
    tiling.set_smallCoreTileLen(smallCoreTileLen);
    tiling.set_smallCoreTileNum(smallCoreTileNum);
    tiling.set_smallCoreTailTileLen(smallCoreTailTileLen);
    tiling.set_smallCoreBatchSize(smallCoreBatchSize);

    uint32_t bigCoreBatchSize = (remainBatch == 0) ? smallCoreBatchSize : (smallCoreBatchSize + 1);
    uint32_t bigCoreTileLen = tileLen;
    if (bigCoreTileLen > bigCoreBatchSize) {
        bigCoreTileLen = bigCoreBatchSize;
    }
    uint32_t bigCoreTileNum = bigCoreBatchSize / bigCoreTileLen;
    uint32_t bigCoreTailTileLen = bigCoreBatchSize % bigCoreTileLen;
    tiling.set_bigCoreTileLen(bigCoreTileLen);
    tiling.set_bigCoreTileNum(bigCoreTileNum);
    tiling.set_bigCoreTailTileLen(bigCoreTailTileLen);
    tiling.set_bigCoreBatchSize(bigCoreBatchSize);
    tiling.set_clsNum(clsNum);

    // printf("batchSize: %u\n", batchSize);
    // printf("coreNum: %d\n", coreNum);

    // printf("bigCoreNum: %u\n", bigCoreNum);
    // printf("bigCoreTileLen: %u\n", bigCoreTileLen);
    // printf("bigCoreTileNum: %u\n", bigCoreTileNum);
    // printf("bigCoreTailTileLen: %u\n", bigCoreTailTileLen);
    // printf("bigCoreBatchSize: %u\n", bigCoreBatchSize);
    // printf("smallCoreTileLen: %u\n", smallCoreTileLen);
    // printf("smallCoreTileNum: %u\n", smallCoreTileNum);
    // printf("smallCoreTailTileLen: %u\n", smallCoreTailTileLen);
    // printf("smallCoreBatchSize: %u\n", smallCoreBatchSize);

    auto attrs = context->GetAttrs();
    const char* reduceStr = attrs->GetStr(0);
    uint32_t reduce = 0;
    if (reduceStr != nullptr && strcmp(reduceStr, "mean") == 0) {
        reduce = 1;
    }
    tiling.set_reduce(reduce);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->SetBlockDim(coreNum);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint32_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    size_t usrSize = 256;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

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
class NLLLoss : public OpDef {
public:
    explicit NLLLoss(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(NLLLoss);
}
