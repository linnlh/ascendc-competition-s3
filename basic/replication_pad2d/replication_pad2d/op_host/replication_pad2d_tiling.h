
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReplicationPad2dTilingData)
    TILING_DATA_FIELD_DEF(int64_t, batch);
    TILING_DATA_FIELD_DEF(int64_t, width);
    TILING_DATA_FIELD_DEF(int64_t, height);
    TILING_DATA_FIELD_DEF(int64_t, tileLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReplicationPad2d, ReplicationPad2dTilingData)
}
