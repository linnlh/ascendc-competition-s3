
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NLLLossTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreBatchSize);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreTailTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreBatchSize);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreTailTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, clsNum);
    TILING_DATA_FIELD_DEF(uint32_t, reduce);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NLLLoss, NLLLossTilingData)
}
