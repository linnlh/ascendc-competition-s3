#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelLogSumExp {
public:
    __aicore__ inline KernelLogSumExp() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        int64_t tileLen,
        int64_t tileNum,
        int64_t* shape,
        int64_t shapeLen,
        int64_t* dim,
        int64_t* strides,
        int64_t dimLen
    ) {
        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        int64_t tileLenAlign32 = (tileLen * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLenAlign32 * 32);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLenAlign32 * sizeof(T));
        pipe.InitBuffer(workBuf, tileLenAlign32 * sizeof(float));
        pipe.InitBuffer(maskBuf, 256);
        pipe.InitBuffer(maxBuf, 32);
        pipe.InitBuffer(sumBuf, 32);
        if constexpr (std::is_same_v<T, half>) {
            pipe.InitBuffer(xfp32Buf, tileLenAlign32 * sizeof(float));
        }

        this->tileLen = tileLen;
        this->tileLenAlign32 = tileLenAlign32;
        this->tileNum = tileNum;
        this->shape = shape;
        this->shapeLen = shapeLen;
        this->dim = dim;
        this->strides = strides;
        this->dimLen = dimLen;
    }

    __aicore__ inline void Process() {
        // printf("[tileLen] %ld\n", tileLen);
        // printf("[tileNum] %ld\n", tileNum);
        // printf("[shapeLen] %ld\n", shapeLen);
        // printf("[shape] ");
        // PrintArray(shape, shapeLen);
        // printf("[dimLen] %ld\n", dimLen);
        // printf("[dim] ");
        // PrintArray(dim, dimLen);
        // printf("[stride] ");
        // PrintArray(strides, dimLen);
        if (dimLen > 1) {
            ProcessDimSizeGT1();
        }
        else {
            ProcessDimSizeEq1();
        }
    }

private:
    __aicore__ inline void ProcessDimSizeEq1() {
        if (strides[0] == 1) {
            for (int i = 0; i < tileNum; i++) {
                CopyIn(i * tileLen, tileLen, 1);
                Compute();
                CopyOut(i, 1, 1);
            }
        }
        else {
            int32_t stride = strides[0];
            for (int i = 0; i < tileNum; i++) {
                int64_t offset = i / stride * stride * shape[dim[0]] + i % stride;
                CopyInUncontinuous(offset, 1, tileLen, stride - 1, 0);
                ComputeUncontinuous();
                CopyOut(i, 1, 1);
            }
        }
    }

    __aicore__ inline void ProcessDimSizeGT1() {
        int32_t dim0ToDim1Size = strides[0] / (strides[1] * shape[dim[1]]);
        int32_t dim0Size = strides[0] * shape[dim[0]];
        int32_t dim1Size = strides[1] * shape[dim[1]];
        if (dimLen == 2 && strides[dimLen - 1] == 1) {
            // int32_t gap = strides[0] / (strides[1] * shape[dim[1]])
            for (int i = 0; i < tileNum; i++) {
                LocalTensor<float> max = maxBuf.Get<float>();
                LocalTensor<float> sum = sumBuf.Get<float>();
                max.SetValue(0, -1e30f);
                sum.SetValue(0, .0f);
                int32_t offset = (i / dim0ToDim1Size * dim0Size) + (i % dim0ToDim1Size * dim1Size);
                // int32_t offset = i / gap * strides[0] * shape[dim[0]] + (i % gap) * shape[dim[1]] * strides[1];
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyIn(offset + j * strides[0], tileLen, 1);
                    ComputeMax();
                }
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyIn(offset + j * strides[0], tileLen, 1);
                    ComputeSum();
                }
                LocalTensor<T> y = yQueue.AllocTensor<T>();
                Ln(sum, sum, 8);
                if constexpr (std::is_same_v<T, half>) {
                    Add(sum, sum, max, 8);
                    Cast(y, sum, RoundMode::CAST_NONE, 8);
                }
                else {
                    Add(y, sum, max, 8);
                }
                yQueue.EnQue<T>(y);
                CopyOut(i, 1, 1);
            }
        }
        else {
            for (int i = 0; i < tileNum; i++) {
                LocalTensor<float> max = maxBuf.Get<float>();
                LocalTensor<float> sum = sumBuf.Get<float>();
                max.SetValue(0, -1e30f);
                sum.SetValue(0, .0f);
                // [xxx, [i], xxx, [j] xxx]
                // dim0ToDim1Size: 1
                // strides1: 64
                // strides0: 2048
                int32_t part1 = i / (dim0ToDim1Size * strides[1]) * dim0Size;
                int32_t part2 = i % (dim0ToDim1Size * strides[1]) / strides[1] * dim1Size;
                int32_t part3 = i % strides[1];
                int32_t offset = part1 + part2 + part3;
                // printf("[part1] %d, [part2] %d, [part3] %d\n", part1, part2, part3);
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyInUncontinuous(offset + j * strides[0], 1, tileLen, strides[1] - 1, 0);
                    ComputeMaxUncontinuous();
                }
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyInUncontinuous(offset + j * strides[0], 1, tileLen, strides[1] - 1, 0);
                    ComputeSumUncontinuous();
                }
                LocalTensor<T> y = yQueue.AllocTensor<T>();
                Ln(sum, sum, 8);
                if constexpr (std::is_same_v<T, half>) {
                    Add(sum, sum, max, 8);
                    Cast(y, sum, RoundMode::CAST_NONE, 8);
                }
                else {
                    Add(y, sum, max, 8);
                }
                yQueue.EnQue<T>(y);
                CopyOut(i, 1, 1);
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<T>(x);
    }

    __aicore__ inline void CopyInUncontinuous(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime,
        uint32_t srcStride,
        uint32_t dstStride
    ) {
        uint32_t blockLen = dataLen * sizeof(T);
        uint32_t srcStrideSize = srcStride * sizeof(T);
        uint32_t dstStrideSize = dstStride * sizeof(T);
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, srcStrideSize, dstStrideSize, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<T>(x);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<T> y = yQueue.AllocTensor<T>();
        LocalTensor<float> work = workBuf.Get<float>();

        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
            Ln(work, work, 8);
            Adds(work, work, maxVal, 8);
            Cast(y, work, RoundMode::CAST_NONE, 8);
        }
        else {
            ReduceMax(work, x, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
            Ln(work, work, 8);
            Adds(y, work, maxVal, 8);
        }
        yQueue.EnQue<T>(y);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeUncontinuous() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<T> y = yQueue.AllocTensor<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Duplicate<uint16_t>(mask, 0x8000u, 128);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
            Ln(work, work, 8);
            Adds(work, work, maxVal, 8);
            Cast(y, work, RoundMode::CAST_NONE, 8);
        }
        else {
            LocalTensor<uint32_t> mask = maskBuf.Get<uint32_t>();
            Duplicate<uint32_t>(mask, 0x80808080u, 64);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            ReduceMax(work, x, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
            Ln(work, work, 8);
            Adds(y, work, maxVal, 8);
        }
        yQueue.EnQue<T>(y);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeMax() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
        }
        else {
            ReduceMax(work, x, work, tileLen);
        }
        Max(max, max, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeSum() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> sum = sumBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        float maxVal = max.GetValue(0);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
        }
        else {
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
        }
        Add(sum, sum, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> y = yQueue.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

    __aicore__ inline void ComputeMaxUncontinuous() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Duplicate<uint16_t>(mask, 0x8000u, 128);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
        }
        else {
            LocalTensor<uint32_t> mask = maskBuf.Get<uint32_t>();
            Duplicate<uint32_t>(mask, 0x80808080u, 64);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            ReduceMax(work, x, work, tileLen);
        }
        Max(max, max, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeSumUncontinuous() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> sum = sumBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        float maxVal = max.GetValue(0);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Duplicate<uint16_t>(mask, 0x8000u, 128);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
        }
        else {
            LocalTensor<uint32_t> mask = maskBuf.Get<uint32_t>();
            Duplicate<uint32_t>(mask, 0x80808080u, 64);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
        }
        Add(sum, sum, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void PrintArray(int64_t* arr, int64_t arrLen) {
        for (int i = 0; i < arrLen; i++) {
            printf("%ld, ", arr[i]);
        }
        printf("\n");
    }

private:
    TPipe pipe;

    GlobalTensor<T> xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> xfp32Buf, workBuf, maskBuf;
    TBuf<QuePosition::VECCALC> maxBuf, sumBuf;

    int64_t tileLen;
    int64_t tileLenAlign32;
    int64_t tileNum;
    int64_t* shape;
    int64_t shapeLen;
    int64_t* dim;
    int64_t* strides;
    int64_t dimLen;
};


extern "C" __global__ __aicore__ void log_sum_exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelLogSumExp<DTYPE_X> op;
    op.Init(x, y, t.tileLen, t.tileNum, t.shape, t.shapeLen, t.dim, t.strides, t.dimLen);
    op.Process();
}