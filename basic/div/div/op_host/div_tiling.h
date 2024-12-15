
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DivTilingData)
    TILING_DATA_FIELD_DEF(int64_t, tileLength);
    TILING_DATA_FIELD_DEF(int64_t, tileNum);
    TILING_DATA_FIELD_DEF(int64_t, tailTileLength);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, x1Shape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, x2Shape);
    TILING_DATA_FIELD_DEF(int64_t, dimNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Div, DivTilingData)
}
