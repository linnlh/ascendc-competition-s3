
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsinhTilingData)
    TILING_DATA_FIELD_DEF(int64_t, tileLength);
    TILING_DATA_FIELD_DEF(int64_t, tileNum);
    TILING_DATA_FIELD_DEF(int64_t, tailTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Asinh, AsinhTilingData)
}
