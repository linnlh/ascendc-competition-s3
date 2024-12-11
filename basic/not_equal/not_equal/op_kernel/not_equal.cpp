#include "kernel_operator.h"

using namespace AscendC;

constexpr int BUFFER_NUM = 1;

template <typename T>
class KernelNotEqual {
public:
    __aicore__ inline KernelNotEqual() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);

        size_t dtypeSize = sizeof(T);
        tileLengthAlign256 = (tileLength * dtypeSize + 255) / 256 * 256 / dtypeSize;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign256 * dtypeSize);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign256 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign256);
        pipe.InitBuffer(yFp16Buf, tileLengthAlign256 * sizeof(half));
        pipe.InitBuffer(maskBuf, tileLengthAlign256);

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailLength = tailLength;
        this->tileLengthAlign256 = tileLengthAlign256;
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength);
            Compute();
            CopyOut(i * tileLength, tileLength);
        }
        if (tailLength > 0) {
            CopyIn(tileNum * tileLength, tailLength);
            Compute();
            CopyOut(tileNum * tileLength, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int offset, int dataSize) {
        uint32_t blockLen = dataSize * sizeof(T);
        LocalTensor<T> x1 = x1Queue.AllocTensor<T>();
        LocalTensor<T> x2 = x2Queue.AllocTensor<T>();
        DataCopyExtParams copyParams {1, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<T>(x1);
        x2Queue.EnQue<T>(x2);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> x1 = x1Queue.DeQue<T>();
        LocalTensor<T> x2 = x2Queue.DeQue<T>();
        LocalTensor<half> yFp16 = yFp16Buf.Get<half>();
        LocalTensor<int8_t> mask = maskBuf.Get<int8_t>();
        LocalTensor<int8_t> y = yQueue.AllocTensor<int8_t>();
        Duplicate<half>(yFp16, 0x0000, tileLengthAlign256);
        Compare(mask, x1, x2, CMPMODE::EQ, tileLengthAlign256);
        PipeBarrier<PIPE_V>();
        Select<half>(yFp16, mask, yFp16, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLengthAlign256);
        PipeBarrier<PIPE_V>();
        Cast(y, yFp16, RoundMode::CAST_ROUND, tileLengthAlign256);
        yQueue.EnQue<int8_t>(y);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(int offset, int dataSize) {
        uint32_t blockLen = dataSize;
        LocalTensor<int8_t> y = yQueue.DeQue<int8_t>();
        DataCopyExtParams copyParams {1, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> maskBuf, yFp16Buf;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailLength;
    int64_t tileLengthAlign256;
};

// template <>
// class KernelNotEqual<half> {
// public:
//     __aicore__ inline KernelNotEqual() {}
//     __aicore__ inline void Init(
//         GM_ADDR x1,
//         GM_ADDR x2,
//         GM_ADDR y,
//         int64_t tileLength,
//         int64_t tileNum,
//         int64_t tailLength
//     ) {
//         x1Gm.SetGlobalBuffer((__gm__ uint16_t*)x1);
//         x2Gm.SetGlobalBuffer((__gm__ uint16_t*)x2);
//         yGm.SetGlobalBuffer((__gm__ int8_t*)y);
//         pipe.InitBuffer(x1Queue, BUFFER_NUM, 8);

//         this->tileLength = tileLength;
//         this->tileNum = tileNum;
//         this->tailLength = tailLength;
//     }

//     __aicore__ inline void Process() {
//         CopyIn();
//         LocalTensor<half> x1 = x1Queue.DeQue<half>();
//         int total = tileLength * tileNum + tailLength;
//         for (int i = 0; i < total; i++) {
//             if (x1Gm.GetValue(i) != x2Gm.GetValue(i)) {
//                 yGm.SetValue(i, 1);
//             } else {
//                 yGm.SetValue(i, 0);
//             }
//         }
//     }

//     __aicore__ inline void CopyIn() {
//         LocalTensor<uint16_t> x1 = x1Queue.AllocTensor<uint16_t>();
//         DataCopy(x1, x1Gm, 10000003);
//         x1Queue.EnQue<uint16_t>(x1);
//     }

// private:
//     TPipe pipe;

//     GlobalTensor<uint16_t> x1Gm, x2Gm;
//     GlobalTensor<int8_t> yGm;
//     TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;

//     int64_t tileLength;
//     int64_t tileNum;
//     int64_t tailLength;
// };

template <>
class KernelNotEqual<int32_t> {
public:
    __aicore__ inline KernelNotEqual() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ int32_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int32_t*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);

        int64_t tileLengthAlign256 = (tileLength + 63) / 64 * 64;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign256 * sizeof(int32_t));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign256 * sizeof(int32_t));
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign256);
        pipe.InitBuffer(x1Fp32Buf, tileLengthAlign256 * sizeof(float));
        pipe.InitBuffer(x2Fp32Buf, tileLengthAlign256 * sizeof(float));
        pipe.InitBuffer(yFp16Buf, tileLengthAlign256 * sizeof(half));
        pipe.InitBuffer(maskBuf, tileLengthAlign256);

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailLength = tailLength;
        this->tileLengthAlign256 = tileLengthAlign256;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength);
            Compute();
            CopyOut(i * tileLength, tileLength);
        }
        if (tailLength > 0) {
            CopyIn(tileNum * tileLength, tailLength);
            Compute();
            CopyOut(tileNum * tileLength, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int offset, int dataSize) {
        uint32_t blockLen = dataSize * sizeof(int32_t);
        LocalTensor<int32_t> x1 = x1Queue.AllocTensor<int32_t>();
        LocalTensor<int32_t> x2 = x2Queue.AllocTensor<int32_t>();
        DataCopyExtParams copyParams {1, blockLen, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<int32_t>(x1);
        x2Queue.EnQue<int32_t>(x2);
    }

    __aicore__ inline void Compute() {
        LocalTensor<int32_t> x1 = x1Queue.DeQue<int32_t>();
        LocalTensor<int32_t> x2 = x2Queue.DeQue<int32_t>();
        LocalTensor<float> x1Fp32 = x1Fp32Buf.Get<float>();
        LocalTensor<float> x2Fp32 = x2Fp32Buf.Get<float>();
        LocalTensor<int8_t> y = yQueue.AllocTensor<int8_t>();
        LocalTensor<half> yFp16 = yFp16Buf.Get<half>();
        LocalTensor<int8_t> mask = maskBuf.Get<int8_t>();

        Cast(x1Fp32, x1, RoundMode::CAST_ROUND, tileLengthAlign256);
        Cast(x2Fp32, x2, RoundMode::CAST_ROUND, tileLengthAlign256);
        PipeBarrier<PIPE_V>();
        Compare(mask, x1Fp32, x2Fp32, CMPMODE::EQ, tileLengthAlign256);
        Duplicate<half>(yFp16, .0f, tileLengthAlign256);
        PipeBarrier<PIPE_V>();
        Select<half>(yFp16, mask, yFp16, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLengthAlign256);
        PipeBarrier<PIPE_V>();
        Cast(y, yFp16, RoundMode::CAST_ROUND, tileLengthAlign256);
        yQueue.EnQue<int8_t>(y);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(int offset, int dataSize) {
        uint32_t blockLen = dataSize;
        LocalTensor<int8_t> y = yQueue.DeQue<int8_t>();
        DataCopyExtParams copyParams {1, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<int32_t> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> x1Fp32Buf, x2Fp32Buf, yFp16Buf;
    TBuf<QuePosition::VECCALC> maskBuf;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailLength;
    int64_t tileLengthAlign256;
};

template <>
class KernelNotEqual<int8_t> {
public:
    __aicore__ inline KernelNotEqual() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ int8_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);

        int64_t tileLengthAlign32 = (tileLength + 31) / 32 * 32;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign32);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign32);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32);

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailLength = tailLength;
        this->tileLengthAlign32 = tileLengthAlign32;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength);
            Compute();
            CopyOut(i * tileLength, tileLength);
        }
        if (tailLength > 0) {
            CopyIn(tileNum * tileLength, tailLength);
            Compute();
            CopyOut(tileNum * tileLength, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int offset, int dataSize) {
        uint32_t blockLen = dataSize;
        LocalTensor<int8_t> x1 = x1Queue.AllocTensor<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue.AllocTensor<int8_t>();
        DataCopyExtParams copyParams {1, blockLen, 0, 0, 0};
        DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<int8_t>(x1);
        x2Queue.EnQue<int8_t>(x2);
    }

    __aicore__ inline void Compute() {
        LocalTensor<int8_t> x1 = x1Queue.DeQue<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue.DeQue<int8_t>();
        LocalTensor<int8_t> y = yQueue.AllocTensor<int8_t>();

        for (int i = 0; i < tileLength; i++) {
            if (x1.GetValue(i) != x2.GetValue(i)) {
                y.SetValue(i, 1);
            } else {
                y.SetValue(i, 0);
            }
        }
        yQueue.EnQue<int8_t>(y);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(int offset, int dataSize) {
        uint32_t blockLen = dataSize;
        LocalTensor<int8_t> y = yQueue.DeQue<int8_t>();
        DataCopyExtParams copyParams {1, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<int8_t> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailLength;
    int64_t tileLengthAlign32;
};

extern "C" __global__ __aicore__ void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelNotEqual<DTYPE_X1> op;
    op.Init(x1, x2, y, t.tileLength, t.tileNum, t.tailLength);
    op.Process();
}