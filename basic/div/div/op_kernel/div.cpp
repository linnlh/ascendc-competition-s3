#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLK_SIZE = 32;

template <typename T>
class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR z,
        int64_t width,
        int64_t widthAlign32,
        int64_t height
    ) {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        zGm.SetGlobalBuffer((__gm__ T*)z);

        pipe.InitBuffer(x1Queue, BUFFER_NUM, widthAlign32 * sizeof(T));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, widthAlign32 * sizeof(T));
        pipe.InitBuffer(zQueue, BUFFER_NUM, widthAlign32 * sizeof(T));
        this->width = width;
        this->widthAlign32 = widthAlign32;
        this->height = height;
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
        LocalTensor<T> x1 = x1Queue.AllocTensor<T>();
        LocalTensor<T> x2 = x2Queue.AllocTensor<T>();
        DataCopy(x1, x1Gm[process * width], widthAlign32);
        DataCopy(x2, x2Gm[process * width], widthAlign32);
        x1Queue.EnQue<T>(x1);
        x2Queue.EnQue<T>(x2);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> x1 = x1Queue.DeQue<T>();
        LocalTensor<T> x2 = x2Queue.DeQue<T>();
        LocalTensor<T> z = zQueue.AllocTensor<T>();
        Div(z, x1, x2, widthAlign32);
        zQueue.EnQue<T>(z);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(int progress) {
        LocalTensor<T> z = zQueue.DeQue<T>();
        DataCopy(zGm[progress * width], z, widthAlign32);
        zQueue.FreeTensor(z);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;

    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<T> zGm;

    int64_t height;
    int64_t width;
    int64_t widthAlign32;
};

template <>
class KernelDiv<int32_t> {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR z,
        int64_t width,
        int64_t widthAlign32,
        int64_t height
    ) {
        x1Gm.SetGlobalBuffer((__gm__ int32_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int32_t*)x2);
        zGm.SetGlobalBuffer((__gm__ int32_t*)z);

        pipe.InitBuffer(x1Queue, BUFFER_NUM, widthAlign32 * sizeof(int32_t));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, widthAlign32 * sizeof(int32_t));
        pipe.InitBuffer(zQueue, BUFFER_NUM, widthAlign32 * sizeof(int32_t));
        pipe.InitBuffer(x1Fp32Buf, widthAlign32 * sizeof(float));
        pipe.InitBuffer(x2Fp32Buf, widthAlign32 * sizeof(float));
        pipe.InitBuffer(zFp32Buf, widthAlign32 * sizeof(float));
        this->width = width;
        this->widthAlign32 = widthAlign32;
        this->height = height;
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
        LocalTensor<int32_t> x1 = x1Queue.AllocTensor<int32_t>();
        LocalTensor<int32_t> x2 = x2Queue.AllocTensor<int32_t>();
        DataCopy(x1, x1Gm[process * width], widthAlign32);
        DataCopy(x2, x2Gm[process * width], widthAlign32);
        x1Queue.EnQue<int32_t>(x1);
        x2Queue.EnQue<int32_t>(x2);
    }

    __aicore__ inline void Compute() {
        LocalTensor<int32_t> x1 = x1Queue.DeQue<int32_t>();
        LocalTensor<int32_t> x2 = x2Queue.DeQue<int32_t>();
        LocalTensor<int32_t> z = zQueue.AllocTensor<int32_t>();
        LocalTensor<float> x1Fp32 = x1Fp32Buf.Get<float>();
        LocalTensor<float> x2Fp32 = x2Fp32Buf.Get<float>();
        LocalTensor<float> zFp32 = zFp32Buf.Get<float>();
        Cast(x1Fp32, x1, RoundMode::CAST_TRUNC, widthAlign32);
        Cast(x2Fp32, x2, RoundMode::CAST_TRUNC, widthAlign32);
        Div(zFp32, x1Fp32, x2Fp32, widthAlign32);
        Cast(z, zFp32, RoundMode::CAST_TRUNC, widthAlign32);
        zQueue.EnQue<int32_t>(z);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(int progress) {
        LocalTensor<int32_t> z = zQueue.DeQue<int32_t>();
        DataCopy(zGm[progress * width], z, widthAlign32);
        zQueue.FreeTensor(z);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> x1Fp32Buf, x2Fp32Buf, zFp32Buf;

    GlobalTensor<int32_t> x1Gm, x2Gm;
    GlobalTensor<int32_t> zGm;

    int64_t height;
    int64_t width;
    int64_t widthAlign32;
};

template <>
class KernelDiv<int8_t> {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR z,
        int64_t width,
        int64_t widthAlign32,
        int64_t height
    ) {
        x1Gm.SetGlobalBuffer((__gm__ int8_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2);
        zGm.SetGlobalBuffer((__gm__ int8_t*)z);

        pipe.InitBuffer(x1Queue, BUFFER_NUM, widthAlign32 * sizeof(int8_t));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, widthAlign32 * sizeof(int8_t));
        pipe.InitBuffer(zQueue, BUFFER_NUM, widthAlign32 * sizeof(int8_t));
        pipe.InitBuffer(x1Fp16Buf, widthAlign32 * sizeof(half));
        pipe.InitBuffer(x2Fp16Buf, widthAlign32 * sizeof(half));
        pipe.InitBuffer(zFp16Buf, widthAlign32 * sizeof(half));
        this->width = width;
        this->widthAlign32 = widthAlign32;
        this->height = height;
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
        LocalTensor<int8_t> x1 = x1Queue.AllocTensor<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue.AllocTensor<int8_t>();
        DataCopy(x1, x1Gm[process * width], widthAlign32);
        DataCopy(x2, x2Gm[process * width], widthAlign32);
        x1Queue.EnQue<int8_t>(x1);
        x2Queue.EnQue<int8_t>(x2);
    }

    __aicore__ inline void Compute() {
        LocalTensor<int8_t> x1 = x1Queue.DeQue<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue.DeQue<int8_t>();
        LocalTensor<int8_t> z = zQueue.AllocTensor<int8_t>();
        LocalTensor<half> x1Fp16 = x1Fp16Buf.Get<half>();
        LocalTensor<half> x2Fp16 = x2Fp16Buf.Get<half>();
        LocalTensor<half> zFp16 = zFp16Buf.Get<half>();
        Cast(x1Fp16, x1, RoundMode::CAST_NONE, widthAlign32);
        Cast(x2Fp16, x2, RoundMode::CAST_NONE, widthAlign32);
        Div(zFp16, x1Fp16, x2Fp16, widthAlign32);
        Cast(z, zFp16, RoundMode::CAST_TRUNC, widthAlign32);
        zQueue.EnQue<int8_t>(z);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(int progress) {
        LocalTensor<int8_t> z = zQueue.DeQue<int8_t>();
        DataCopy(zGm[progress * width], z, widthAlign32);
        zQueue.FreeTensor(z);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> x1Fp16Buf, x2Fp16Buf, zFp16Buf;

    GlobalTensor<int8_t> x1Gm, x2Gm;
    GlobalTensor<int8_t> zGm;

    int64_t height;
    int64_t width;
    int64_t widthAlign32;
};

extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelDiv<DTYPE_X1> op;
    op.Init(x1, x2, z, t.width, t.widthAlign32, t.height);
    op.Process();
}