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
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        yGm.SetGlobalBuffer((__gm__ T*)y);
        dyGm.SetGlobalBuffer((__gm__ T*)dy);
        zGm.SetGlobalBuffer((__gm__ T*)z);

        int64_t dtypeSize = sizeof(T);
        int64_t tileLengthAlign32 = (tileLength * dtypeSize + 31) / 32 * 32 / dtypeSize;
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(dyQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(zQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(temp1Buf, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(temp2Buf, tileLengthAlign32 * dtypeSize);

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
    TPipe pipe;

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
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        yGm.SetGlobalBuffer((__gm__ half*)y);
        dyGm.SetGlobalBuffer((__gm__ half*)dy);
        zGm.SetGlobalBuffer((__gm__ half*)z);

        int64_t dtypeSize = 2;
        int64_t tileLengthAlign32 = (tileLength * dtypeSize + 31) / 32 * 32 / dtypeSize;
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(dyQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(zQueue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(yfp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(dyfp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(temp1fp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(temp2fp32Buf, tileLengthAlign32 * sizeof(float));

        this->tileLengthAlign32 = tileLengthAlign32;
        this->tileLength = tileLength;
        this->tileNum = tileNum;

        int64_t tailTileLengthAlign32 = (tailTileLength + 15) / 16 * 16;
        this->tailTileLength = tailTileLength;
        this->tailTileLengthAlign32 = tailTileLengthAlign32;
    }

    __aicore__ inline void Process() {
        uint64_t offset = 0;
        for (int i = 0; i < tileNum; i++) {
            CopyIn(offset, tileLength, 1);
            Compute(tileLengthAlign32);
            CopyOut(offset, tileLength, 1);
            offset += tileLength;
        }
        if (tailTileLength > 0) {
            CopyIn(offset, tailTileLength, 1);
            Compute(tailTileLengthAlign32);
            CopyOut(offset, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(half);
        LocalTensor<half> y = yQueue.AllocTensor<half>();
        LocalTensor<half> dy = dyQueue.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
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
        LocalTensor<float> yfp32 = yfp32Buf.Get<float>();
        LocalTensor<float> dyfp32 = dyfp32Buf.Get<float>();
        LocalTensor<float> temp1fp32 = temp1fp32Buf.Get<float>();
        LocalTensor<float> temp2fp32 = temp2fp32Buf.Get<float>();

        Cast(dyfp32, dy, RoundMode::CAST_NONE, computeLength);
        Cast(yfp32, y, RoundMode::CAST_NONE, computeLength);
        Exp(temp1fp32, yfp32, computeLength);
        Muls(temp2fp32, yfp32, -1.0f, computeLength);
        Exp(temp2fp32, temp2fp32, computeLength);
        Add(yfp32, temp1fp32, temp2fp32, computeLength);
        Muls(dyfp32, dyfp32, 2.0f, computeLength);
        Div(temp1fp32, dyfp32, yfp32, computeLength);
        Cast(z, temp1fp32, RoundMode::CAST_ROUND, computeLength);
        zQueue.EnQue<half>(z);
        yQueue.FreeTensor(y);
        dyQueue.FreeTensor(dy);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(half);
        LocalTensor<half> z = zQueue.DeQue<half>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        zQueue.FreeTensor(z);
    }

private:
    TPipe pipe;

    GlobalTensor<half> yGm, dyGm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue, dyQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> yfp32Buf, dyfp32Buf;
    TBuf<QuePosition::VECCALC> temp1fp32Buf, temp2fp32Buf;

    int64_t tileLength;
    int64_t tileLengthAlign32;
    int64_t tileNum;
    int64_t tailTileLength;
    int64_t tailTileLengthAlign32;
};

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelAsinhGrad<DTYPE_Y> op;
    op.Init(y, dy, z, t.tileLength, t.tileNum, t.tailTileLength);
    op.Process();
}