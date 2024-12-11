#include "kernel_operator.h"

using namespace AscendC;

constexpr int BUFFER_NUM = 2;

template <typename T>
class KernelDiv {};

template <>
class KernelDiv<float> {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR z,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ float*)x1);
        x2Gm.SetGlobalBuffer((__gm__ float*)x2);
        zGm.SetGlobalBuffer((__gm__ float*)z);

        int64_t tileLengthAlign32 = (tileLength + 7) / 8 * 8;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(zQueue, BUFFER_NUM, tileLengthAlign32 * sizeof(float));

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
        this->tileLengthAlign32 = tileLengthAlign32;
        this->tailTileLengthAlign32 = (tailTileLength + 7) / 8 * 8;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength, 1);
            Compute(tileLengthAlign32);
            CopyOut(i * tileLength, tileLength, 1);
        }
        if (tailTileLength > 0) {
            CopyIn(tileNum * tileLength, tailTileLength, 1);
            Compute(tailTileLengthAlign32);
            CopyOut(tileNum * tileLength, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint64_t offset,
        uint32_t length,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = length * sizeof(float);
        LocalTensor<float> x1 = x1Queue.AllocTensor<float>();
        LocalTensor<float> x2 = x2Queue.AllocTensor<float>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<float>(x1);
        x2Queue.EnQue<float>(x2);
    }
    __aicore__ inline void Compute(int64_t computeLen) {
        LocalTensor<float> x1 = x1Queue.DeQue<float>();
        LocalTensor<float> x2 = x2Queue.DeQue<float>();
        LocalTensor<float> z = zQueue.AllocTensor<float>();
        Div(z, x1, x2, computeLen);
        zQueue.EnQue<float>(z);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }
    __aicore__ inline void CopyOut(
        uint64_t offset,
        uint32_t length,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = length * sizeof(float);
        LocalTensor<float> z = zQueue.DeQue<float>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        zQueue.FreeTensor(z);
    }

private:
    TPipe pipe;
    GlobalTensor<float> x1Gm, x2Gm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailTileLength;
    int64_t tileLengthAlign32;
    int64_t tailTileLengthAlign32;
};

template <>
class KernelDiv<half> {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR z,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ half*)x1);
        x2Gm.SetGlobalBuffer((__gm__ half*)x2);
        zGm.SetGlobalBuffer((__gm__ half*)z);

        int64_t tileLengthAlign32 = (tileLength + 15) / 16 * 16;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign32 * sizeof(half));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign32 * sizeof(half));
        pipe.InitBuffer(zQueue, BUFFER_NUM, tileLengthAlign32 * sizeof(half));
        pipe.InitBuffer(x1fp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(x2fp32Buf, tileLengthAlign32 * sizeof(float));

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
        this->tileLengthAlign32 = tileLengthAlign32;
        this->tailTileLengthAlign32 = (tailTileLength + 15) / 16 * 16;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength, 1);
            Compute(tileLengthAlign32);
            CopyOut(i * tileLength, tileLength, 1);
        }
        if (tailTileLength > 0) {
            CopyIn(tileNum * tileLength, tailTileLength, 1);
            Compute(tailTileLengthAlign32);
            CopyOut(tileNum * tileLength, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint64_t offset,
        uint32_t length,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = length * sizeof(half);
        LocalTensor<half> x1 = x1Queue.AllocTensor<half>();
        LocalTensor<half> x2 = x2Queue.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<half>(x1);
        x2Queue.EnQue<half>(x2);
    }
    __aicore__ inline void Compute(int64_t computeLen) {
        LocalTensor<half> x1 = x1Queue.DeQue<half>();
        LocalTensor<half> x2 = x2Queue.DeQue<half>();
        LocalTensor<half> z = zQueue.AllocTensor<half>();
        LocalTensor<float> x1fp32 = x1fp32Buf.Get<float>();
        LocalTensor<float> x2fp32 = x2fp32Buf.Get<float>();

        Cast(x1fp32, x1, RoundMode::CAST_NONE, computeLen);
        Cast(x2fp32, x2, RoundMode::CAST_NONE, computeLen);
        Div(x1fp32, x1fp32, x2fp32, computeLen);
        Cast(z, x1fp32, RoundMode::CAST_RINT, computeLen);
        zQueue.EnQue<half>(z);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }
    __aicore__ inline void CopyOut(
        uint64_t offset,
        uint32_t length,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = length * sizeof(half);
        LocalTensor<half> z = zQueue.DeQue<half>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        zQueue.FreeTensor(z);
    }

private:
    TPipe pipe;
    GlobalTensor<half> x1Gm, x2Gm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> x1fp32Buf, x2fp32Buf;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailTileLength;
    int64_t tileLengthAlign32;
    int64_t tailTileLengthAlign32;
};

template <>
class KernelDiv<int32_t> {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR z,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ int32_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int32_t*)x2);
        zGm.SetGlobalBuffer((__gm__ int32_t*)z);

        int64_t tileLengthAlign32 = (tileLength + 7) / 8 * 8;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign32 * sizeof(int32_t));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign32 * sizeof(int32_t));
        pipe.InitBuffer(zQueue, BUFFER_NUM, tileLengthAlign32 * sizeof(int32_t));
        pipe.InitBuffer(x1fp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(x2fp32Buf, tileLengthAlign32 * sizeof(float));

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
        this->tileLengthAlign32 = tileLengthAlign32;
        this->tailTileLengthAlign32 = (tailTileLength + 7) / 8 * 8;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength, 1);
            Compute(tileLengthAlign32);
            CopyOut(i * tileLength, tileLength, 1);
        }
        if (tailTileLength > 0) {
            CopyIn(tileNum * tileLength, tailTileLength, 1);
            Compute(tailTileLengthAlign32);
            CopyOut(tileNum * tileLength, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint64_t offset,
        uint32_t length,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = length * sizeof(int32_t);
        LocalTensor<int32_t> x1 = x1Queue.AllocTensor<int32_t>();
        LocalTensor<int32_t> x2 = x2Queue.AllocTensor<int32_t>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<int32_t>(x1);
        x2Queue.EnQue<int32_t>(x2);
    }
    __aicore__ inline void Compute(int64_t computeLen) {
        LocalTensor<int32_t> x1 = x1Queue.DeQue<int32_t>();
        LocalTensor<int32_t> x2 = x2Queue.DeQue<int32_t>();
        LocalTensor<int32_t> z = zQueue.AllocTensor<int32_t>();
        LocalTensor<float> x1fp32 = x1fp32Buf.Get<float>();
        LocalTensor<float> x2fp32 = x2fp32Buf.Get<float>();

        Cast(x1fp32, x1, RoundMode::CAST_NONE, computeLen);
        Cast(x2fp32, x2, RoundMode::CAST_NONE, computeLen);
        Div(x1fp32, x1fp32, x2fp32, computeLen);
        Cast(z, x1fp32, RoundMode::CAST_TRUNC, computeLen);
        zQueue.EnQue<int32_t>(z);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }
    __aicore__ inline void CopyOut(
        uint64_t offset,
        uint32_t length,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = length * sizeof(int32_t);
        LocalTensor<int32_t> z = zQueue.DeQue<int32_t>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        zQueue.FreeTensor(z);
    }

private:
    TPipe pipe;
    GlobalTensor<int32_t> x1Gm, x2Gm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> x1fp32Buf, x2fp32Buf;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailTileLength;
    int64_t tileLengthAlign32;
    int64_t tailTileLengthAlign32;
};

template <>
class KernelDiv<int8_t> {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR z,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ int8_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2);
        zGm.SetGlobalBuffer((__gm__ int8_t*)z);

        int64_t totalLength = tileLength * tileNum + tailTileLength;
        this->totalLength = totalLength;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < totalLength; i++) {
            int8_t value = x1Gm.GetValue(i) / x2Gm.GetValue(i);
            zGm.SetValue(i, value);
        }
    }

private:
    GlobalTensor<int8_t> x1Gm, x2Gm, zGm;
    int64_t totalLength;
};

extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelDiv<DTYPE_X1> op;
    op.Init(x1, x2, z, t.tileLength, t.tileNum, t.tailTileLength);
    op.Process();
}