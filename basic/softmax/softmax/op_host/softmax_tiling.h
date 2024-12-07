#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
  TILING_DATA_FIELD_DEF(int64_t, width);
  TILING_DATA_FIELD_DEF(int64_t, widthAlign32);
  TILING_DATA_FIELD_DEF(int64_t, height);
  TILING_DATA_FIELD_DEF(int64_t, tileLength);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, SoftmaxTilingData)
}
