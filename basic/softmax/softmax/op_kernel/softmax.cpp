#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BLK_SIZE = 32;

template <typename T>
class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        int64_t height,
        int64_t width,
        int64_t widthAlign32,
        const SoftMaxTiling &tilingData
    ) {
        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        pipe.InitBuffer(xQueue, BUFFER_NUM, widthAlign32 * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM, widthAlign32 * sizeof(T));
        pipe.InitBuffer(maxQueue, BUFFER_NUM, widthAlign32 * sizeof(T));
        pipe.InitBuffer(sumQueue, BUFFER_NUM, widthAlign32 * sizeof(T));
        this->width = width;
        this->widthAlign32 = widthAlign32;
        this->height = height;
        this->tiling = tilingData;
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < height; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int process) {
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopy(x, xGm[process * width], widthAlign32);
        xQueue.EnQue<T>(x);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<T> sumTemp = sumQueue.AllocTensor<T>();
        LocalTensor<T> maxTemp = maxQueue.AllocTensor<T>();
        LocalTensor<T> y = yQueue.AllocTensor<T>();

        SoftMaxShapeInfo shape = {1, (uint32_t)width, 1, (uint32_t)width};
        SoftMax<T>(y, sumTemp, maxTemp, x, tiling, shape);

        yQueue.EnQue<T>(y);
        maxQueue.FreeTensor(maxTemp);
        sumQueue.FreeTensor(sumTemp);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int progress) {
        LocalTensor<T> y = yQueue.DeQue<T>();
        DataCopy(yGm[progress * width], y, widthAlign32);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue, maxQueue, sumQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    int64_t height;
    int64_t width;
    int64_t widthAlign32;
    SoftMaxTiling tiling;
};

extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelSoftmax<DTYPE_X> op;
    op.Init(x, y, t.height, t.width, t.widthAlign32, t.softmaxTilingData);
    op.Process();
}