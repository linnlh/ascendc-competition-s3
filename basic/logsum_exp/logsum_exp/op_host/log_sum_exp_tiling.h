
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSumExpTilingData)
    TILING_DATA_FIELD_DEF(int64_t, tileLen);
    TILING_DATA_FIELD_DEF(int64_t, tileNum);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, shape);
    TILING_DATA_FIELD_DEF(int64_t, shapeLen);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, dim);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, strides);
    TILING_DATA_FIELD_DEF(int64_t, dimLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSumExp, LogSumExpTilingData)
}
