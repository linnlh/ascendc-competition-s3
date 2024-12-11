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
class KernelAsinh<half> {
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);

        int64_t dtypeSize = 2;
        int64_t tileLengthAlign32 = (tileLength + 15) / 16 * 16;
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(xfp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(yfp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(posBuf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(negBuf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(tempBuf, tileLengthAlign32 * sizeof(float));

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
        uint32_t blockSize = blockLen * sizeof(half);
        LocalTensor<half> x = xQueue.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<half>(x);
    }

    __aicore__ inline void Compute() {
        LocalTensor<half> x = xQueue.DeQue<half>();
        LocalTensor<half> y = yQueue.AllocTensor<half>();
        LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
        LocalTensor<float> yfp32 = yfp32Buf.Get<float>();
        LocalTensor<float> pos = posBuf.Get<float>();
        LocalTensor<float> neg = negBuf.Get<float>();
        LocalTensor<float> temp = tempBuf.Get<float>();

        Cast(xfp32, x, RoundMode::CAST_NONE, tileLengthAlign32);
        Cast(yfp32, y, RoundMode::CAST_NONE, tileLengthAlign32);

        // 处理 x > 0
        Maxs(pos, xfp32, .0f, tileLengthAlign32);
        Mul(temp, pos, pos, tileLengthAlign32);
        Adds(temp, temp, 1.0f, tileLengthAlign32);
        Sqrt(temp, temp, tileLengthAlign32);
        Add(pos, pos, temp, tileLengthAlign32);
        Ln(pos, pos, tileLengthAlign32);

        // 处理 x < 0
        Mins(neg, xfp32, .0f, tileLengthAlign32);
        Mul(temp, neg, neg, tileLengthAlign32);
        Adds(temp, temp, 1.0f, tileLengthAlign32);
        Sqrt(temp, temp, tileLengthAlign32);
        Sub(neg, temp, neg, tileLengthAlign32);
        Ln(neg, neg, tileLengthAlign32);

        // 合并
        Sub(yfp32, pos, neg, tileLengthAlign32);
        Cast(y, yfp32, RoundMode::CAST_ROUND, tileLengthAlign32);

        yQueue.EnQue<half>(y);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(half);
        LocalTensor<half> y = yQueue.DeQue<half>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<half> xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> xfp32Buf, yfp32Buf;
    TBuf<QuePosition::VECCALC> posBuf, negBuf;
    TBuf<QuePosition::VECCALC> tempBuf;

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