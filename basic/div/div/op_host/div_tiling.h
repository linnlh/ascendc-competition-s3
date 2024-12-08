
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DivTilingData)
  TILING_DATA_FIELD_DEF(int64_t, width);
  TILING_DATA_FIELD_DEF(int64_t, widthAlign32);
  TILING_DATA_FIELD_DEF(int64_t, height);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Div, DivTilingData)
}
