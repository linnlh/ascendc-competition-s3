#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelAsinhGrad {
public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(
        GM_ADDR y,
        GM_ADDR dy,
        GM_ADDR z,
        TPipe* pipe,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        this->pipe = pipe;
        yGm.SetGlobalBuffer((__gm__ T*)y);
        dyGm.SetGlobalBuffer((__gm__ T*)dy);
        zGm.SetGlobalBuffer((__gm__ T*)z);

        int64_t dtypeSize = sizeof(T);
        int64_t tileLengthAlign32 = (tileLength * dtypeSize + 31) / 32 * 32 / dtypeSize;
        pipe->InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe->InitBuffer(dyQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe->InitBuffer(zQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe->InitBuffer(temp1Buf, tileLengthAlign32 * dtypeSize);
        pipe->InitBuffer(temp2Buf, tileLengthAlign32 * dtypeSize);

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
        LocalTensor<T> y = yQueue.AllocTensor<T>();
        LocalTensor<T> dy = dyQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(y, yGm[offset], copyParams, padParams);
        DataCopyPad(dy, dyGm[offset], copyParams, padParams);
        yQueue.EnQue<T>(y);
        dyQueue.EnQue<T>(dy);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> y = yQueue.DeQue<T>();
        LocalTensor<T> dy = dyQueue.DeQue<T>();
        LocalTensor<T> z = zQueue.AllocTensor<T>();
        LocalTensor<T> temp1 = temp1Buf.Get<T>();
        LocalTensor<T> temp2 = temp2Buf.Get<T>();
        Exp(temp1, y, tileLengthAlign32);
        Muls<T>(temp2, y, -1.0f, tileLengthAlign32);
        Exp(temp2, temp2, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Add(y, temp1, temp2, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Muls<T>(y, y, 0.5f, tileLengthAlign32);
        PipeBarrier<PIPE_V>();
        Div(z, dy, y, tileLengthAlign32);
        zQueue.EnQue<T>(z);
        yQueue.FreeTensor(y);
        dyQueue.FreeTensor(dy);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(T);
        LocalTensor<T> z = zQueue.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        zQueue.FreeTensor(z);
    }

private:
    TPipe* pipe;

    GlobalTensor<T> yGm, dyGm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue, dyQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> temp1Buf, temp2Buf;

    int64_t tileLength;
    int64_t tileLengthAlign32;
    int64_t tileNum;
    int64_t tailTileLength;
};

template <>
class KernelAsinhGrad<half> {
public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(
        GM_ADDR y,
        GM_ADDR dy,
        GM_ADDR z,
        TPipe* pipe,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        this->pipe = pipe;
        yGm.SetGlobalBuffer((__gm__ half*)y);
        dyGm.SetGlobalBuffer((__gm__ half*)dy);
        zGm.SetGlobalBuffer((__gm__ half*)z);

        // uint16_t tileLengthAlign32 = (tileLength + 15) / 16 * 16;
        pipe->InitBuffer(yQueue, BUFFER_NUM, tileLength * sizeof(half));
        pipe->InitBuffer(dyQueue, BUFFER_NUM, tileLength * sizeof(half));
        pipe->InitBuffer(zQueue, BUFFER_NUM, tileLength * sizeof(half));
        pipe->InitBuffer(work1Buf, tileLength * sizeof(float));
        pipe->InitBuffer(work2Buf, tileLength * sizeof(float));
        pipe->InitBuffer(onesBuf, tileLength * sizeof(float));
        LocalTensor<float> ones = onesBuf.Get<float>();
        Duplicate(ones, 1.0f, tileLength);

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
    }

    __aicore__ inline void Process() {
        uint64_t offset = 0;
        uint32_t dataSize = tileLength * sizeof(half);
        for (int i = 0; i < tileNum; i++) {
            CopyIn(offset, dataSize, 1);
            Compute(tileLength);
            CopyOut(offset, dataSize, 1);
            offset += tileLength;
        }
        dataSize = tailTileLength * sizeof(half);
        CopyIn(offset, dataSize, 1);
        Compute(tailTileLength);
        CopyOut(offset, dataSize, 1);
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t dataSize, uint16_t repeatTime) {
        // uint32_t dataSize = dataLen * sizeof(half);
        LocalTensor<half> y = yQueue.AllocTensor<half>();
        LocalTensor<half> dy = dyQueue.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(y, yGm[offset], copyParams, padParams);
        DataCopyPad(dy, dyGm[offset], copyParams, padParams);
        yQueue.EnQue<half>(y);
        dyQueue.EnQue<half>(dy);
    }

    __aicore__ inline void Compute(int computeLength) {
        LocalTensor<half> y = yQueue.DeQue<half>();
        LocalTensor<half> dy = dyQueue.DeQue<half>();
        LocalTensor<half> z = zQueue.AllocTensor<half>();
        // LocalTensor<float> y2 = temp1fp32Buf.Get<float>();
        // LocalTensor<float> work = temp2fp32Buf.Get<float>();
        // --fp32 taylor expansion--
        // Duplicate(work, 1.0f, computeLength);
        // Duplicate(zfp32, 1.0f, computeLength);
        // Cast(yfp32, y, RoundMode::CAST_NONE, computeLength);
        // Mul(y2, yfp32, yfp32, computeLength);
        // for (int i = 0; i < 8; i++) {
        //     Mul(work, work, y2, computeLength);
        //     Axpy(zfp32, work, coefs[i], computeLength);
        // }
        // Cast(yfp32, dy, RoundMode::CAST_NONE, computeLength);
        // Div(zfp32, yfp32, zfp32, computeLength);
        // Cast(z, zfp32, RoundMode::CAST_NONE, computeLength);
        // --end--
        // --fp32 taylor expansion2--
        // LocalTensor<float> yfp32 = yfp32Buf.Get<float>();
        // LocalTensor<float> dyfp32 = dyfp32Buf.Get<float>();
        // LocalTensor<float> y2 = temp1fp32Buf.Get<float>();
        // LocalTensor<float> term = temp2fp32Buf.Get<float>();
        // Cast(yfp32, y, RoundMode::CAST_NONE, computeLength);
        // Mul(y2, yfp32, yfp32, computeLength);
        // Duplicate(yfp32, 1.0f, computeLength);  // result
        // Duplicate(term, 1.0f, computeLength);
        // for (int i = 1; i < 5; i++) {
        //     Muls(dyfp32, y2, 1.0f / (2 * i) / (2 * i - 1), computeLength);
        //     Mul(term, term, dyfp32, computeLength);
        //     Add(yfp32, term, yfp32, computeLength);
        // }
        // Cast(dyfp32, dy, RoundMode::CAST_NONE, computeLength);
        // Div(dyfp32, dyfp32, yfp32, computeLength);
        // Cast(z, dyfp32, RoundMode::CAST_NONE, computeLength);
        // --end

        // --fp16 taylor expansion--
        // LocalTensor<half> y2 = temp1fp32Buf.Get<half>();
        // LocalTensor<half> work = temp2fp32Buf.Get<half>();
        // Duplicate<half>(work, 1.0f, computeLength);
        // Duplicate<half>(z, 1.0f, computeLength);
        // Mul(y2, y, y, computeLength);
        // for (int i = 0; i < 5; i++) {
        //     Mul(work, work, y2, computeLength);
        //     Axpy(z, work, coefs[i], computeLength);
        // }
        // Div(z, dy, z, computeLength);
        // --end--
        // --fp16 taylor expansion2--
        // LocalTensor<half> y2 = temp1fp32Buf.Get<half>();
        // LocalTensor<half> term = temp2fp32Buf.Get<half>();
        // Duplicate<half>(term, (half)1.0, computeLength);
        // Duplicate<half>(z, (half)1.0, computeLength);
        // Mul(y2, y, y, computeLength);
        // for (int i = 0; i < 5; i++) {
        //     Muls<half>(y, y2, (half)(1.0f / (2 * i) / (2 * i - 1)), computeLength);
        //     Mul(term, term, y, computeLength);
        //     // ReduceMax(y, term, y, computeLength);
        //     // if ((float)y.GetValue(0) < 1e-12f)
        //     //     break;
        //     Add(z, term, z, computeLength);
        // }
        // Div(z, dy, z, computeLength);
        // --end--

        // normal
        LocalTensor<float> work1 = work1Buf.Get<float>();
        LocalTensor<float> work2 = work2Buf.Get<float>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        Cast(work1, y, RoundMode::CAST_NONE, computeLength);
        Exp(work1, work1, computeLength);
        Div(work2, ones, work1, computeLength);
        Add(work1, work1, work2, computeLength);
        Cast(y, work1, RoundMode::CAST_NONE, computeLength);
        Add(dy, dy, dy, computeLength);
        Div(z, dy, y, computeLength);
        // ---
        // Exp(temp1fp32, yfp32, computeLength);
        // Muls(temp2fp32, yfp32, -1.0f, computeLength);
        // Exp(temp2fp32, temp2fp32, computeLength);
        // Add(yfp32, temp1fp32, temp2fp32, computeLength);
        // Muls(dyfp32, dyfp32, 2.0f, computeLength);
        // Div(temp1fp32, dyfp32, yfp32, computeLength);
        // Cast(z, temp1fp32, RoundMode::CAST_ROUND, computeLength);

        zQueue.EnQue<half>(z);
        yQueue.FreeTensor(y);
        dyQueue.FreeTensor(dy);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataSize, uint16_t repeatTime) {
        // uint32_t dataSize = dataLen * sizeof(half);
        LocalTensor<half> z = zQueue.DeQue<half>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        zQueue.FreeTensor(z);
    }

private:
    TPipe* pipe;

    GlobalTensor<half> yGm, dyGm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue, dyQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> work1Buf, work2Buf;
    TBuf<QuePosition::VECCALC> onesBuf;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailTileLength;
};

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    TPipe pipe;
    KernelAsinhGrad<DTYPE_Y> op;
    op.Init(y, dy, z, &pipe, t.tileLength, t.tileNum, t.tailTileLength);
    op.Process();
}