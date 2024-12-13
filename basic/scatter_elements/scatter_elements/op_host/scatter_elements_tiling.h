
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsTilingData)
    TILING_DATA_FIELD_DEF(int64_t, tileLength);
    TILING_DATA_FIELD_DEF(int64_t, tileNum);
    TILING_DATA_FIELD_DEF(int64_t, tailTileLength);
    TILING_DATA_FIELD_DEF(int64_t, varLen);
    TILING_DATA_FIELD_DEF(int64_t, indicesLen);
    TILING_DATA_FIELD_DEF(int64_t, varStride);
    TILING_DATA_FIELD_DEF(int64_t, indiceStride);
    TILING_DATA_FIELD_DEF(int64_t, mode);
    TILING_DATA_FIELD_DEF(int64_t, varTileLen);
    TILING_DATA_FIELD_DEF(int64_t, indiceTileLen);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 10, varShape);
    TILING_DATA_FIELD_DEF(int64_t, varShapeLen);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 10, indicesShape);
    TILING_DATA_FIELD_DEF(int64_t, indicesShapeLen);
    // TILING_DATA_FIELD_DEF_ARR(uint16_t, 10, varShape);
    // TILING_DATA_FIELD_DEF(int64_t, updateTileLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsTilingData)
}
