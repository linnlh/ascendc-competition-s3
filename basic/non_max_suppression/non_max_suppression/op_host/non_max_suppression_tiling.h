
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NonMaxSuppressionTilingData)
  TILING_DATA_FIELD_DEF(int64_t, batchSize);
  TILING_DATA_FIELD_DEF(int64_t, classNum);
  TILING_DATA_FIELD_DEF(int64_t, boxNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonMaxSuppression, NonMaxSuppressionTilingData)
}
