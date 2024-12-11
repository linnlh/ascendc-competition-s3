#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelAsinh {
public:
    __aicore__ inline KernelAsinh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        int64_t dtypeSize = sizeof(T);
        int64_t tileLengthAlign32 = (tileLength * dtypeSize + 31) / 32 * 32 / dtypeSize;
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(posBuf, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(posTempBuf, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(negBuf, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(negTempBuf, tileLengthAlign32 * dtypeSize);

        this->tileLengthAlign32 = tileLengthAlign32;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
    }

    __aicore__ inline void Process() {
        uint64_t offset = 0;
        for (int i = 0; i < tileNum; i++) {
            CopyIn(offset, tileLength, 1);
            Compute();
            CopyOut(offset, tileLength, 1);
            offset += tileLength;
        }
        if (tailTileLength > 0) {
            CopyIn(offset, tailTileLength, 1);
            Compute();
            CopyOut(offset, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(T);
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<T>(x);
        // LocalTensor<T> x = xQueue.AllocTensor<T>();
        // DataCopy(x, xGm[offset], blockLen);
        // xQueue.EnQue<T>(x);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<T> y = yQueue.AllocTensor<T>();
        LocalTensor<T> pos = posBuf.Get<T>();
        LocalTensor<T> posTemp = posTempBuf.Get<T>();
        LocalTensor<T> neg = negBuf.Get<T>();
        LocalTensor<T> negTemp = negTempBuf.Get<T>();

        T factor = 0.5f;
        T one = 1.0f;
        T zero = +0.0f;
        Maxs(pos, x, zero, tileLengthAlign32);
        Mins(neg, x, zero, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Mul(posTemp, pos, pos, tileLengthAlign32);
        Mul(negTemp, neg, neg, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Adds(posTemp, posTemp, one, tileLengthAlign32);
        Adds(negTemp, negTemp, one, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Ln(posTemp, posTemp, tileLengthAlign32);
        Ln(negTemp, negTemp, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Muls(posTemp, posTemp, factor, tileLengthAlign32);
        Muls(negTemp, negTemp, factor, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Exp(posTemp, posTemp, tileLengthAlign32);
        Exp(negTemp, negTemp, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Add(pos, pos, posTemp, tileLengthAlign32);
        Sub(neg, negTemp, neg, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Ln(pos, pos, tileLengthAlign32);
        Ln(neg, neg, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Sub(y, pos, neg, tileLengthAlign32);
        yQueue.EnQue<T>(y);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(T);
        LocalTensor<T> y = yQueue.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
        // LocalTensor<T> y = yQueue.DeQue<T>();
        // DataCopy(yGm[offset], y, blockLen);
        // yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<T> xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> posBuf, negBuf;
    TBuf<QuePosition::VECCALC> posTempBuf, negTempBuf;

    int64_t tileLength;
    int64_t tileLengthAlign32;
    int64_t tileNum;
    int64_t tailTileLength;
};


template <>
class KernelAsinh<float> {
public:
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        int64_t dtypeSize = 4;
        int64_t tileLengthAlign32 = (tileLength * dtypeSize + 31) / 32 * 32 / dtypeSize;
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(posBuf, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(posTempBuf, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(negBuf, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(negTempBuf, tileLengthAlign32 * dtypeSize);

        this->tileLengthAlign32 = tileLengthAlign32;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
    }

    __aicore__ inline void Process() {
        uint64_t offset = 0;
        for (int i = 0; i < tileNum; i++) {
            CopyIn(offset, tileLength, 1);
            Compute();
            CopyOut(offset, tileLength, 1);
            offset += tileLength;
        }
        if (tailTileLength > 0) {
            CopyIn(offset, tailTileLength, 1);
            Compute();
            CopyOut(offset, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(float);
        LocalTensor<float> x = xQueue.AllocTensor<float>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<float>(x);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> x = xQueue.DeQue<float>();
        LocalTensor<float> y = yQueue.AllocTensor<float>();
        LocalTensor<float> pos = posBuf.Get<float>();
        LocalTensor<float> posTemp = posTempBuf.Get<float>();
        LocalTensor<float> neg = negBuf.Get<float>();
        LocalTensor<float> negTemp = negTempBuf.Get<float>();

        float factor = 0.5f;
        float one = 1.0f;
        float zero = +0.0f;
        Maxs(pos, x, +0.0f, tileLengthAlign32);
        Mins(neg, x, +0.0f, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Mul(posTemp, pos, pos, tileLengthAlign32);
        Mul(negTemp, neg, neg, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Adds(posTemp, posTemp, one, tileLengthAlign32);
        Adds(negTemp, negTemp, one, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Ln(posTemp, posTemp, tileLengthAlign32);
        Ln(negTemp, negTemp, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Muls(posTemp, posTemp, factor, tileLengthAlign32);
        Muls(negTemp, negTemp, factor, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Exp(posTemp, posTemp, tileLengthAlign32);
        Exp(negTemp, negTemp, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Add(pos, pos, posTemp, tileLengthAlign32);
        Sub(neg, negTemp, neg, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Ln(pos, pos, tileLengthAlign32);
        Ln(neg, neg, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Sub(y, pos, neg, tileLengthAlign32);
        yQueue.EnQue<float>(y);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(float);
        LocalTensor<float> y = yQueue.DeQue<float>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<float> xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> posBuf, negBuf;
    TBuf<QuePosition::VECCALC> posTempBuf, negTempBuf;

    int64_t tileLength;
    int64_t tileLengthAlign32;
    int64_t tileNum;
    int64_t tailTileLength;
};

extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelAsinh<DTYPE_X> op;
    op.Init(x, y, t.tileLength, t.tileNum, t.tailTileLength);
    op.Process();
}