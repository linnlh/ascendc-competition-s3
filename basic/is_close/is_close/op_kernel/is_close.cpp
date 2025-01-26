#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelIsClose {
public:
    __aicore__ inline KernelIsClose() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        float rtol,
        float atol,
        int64_t* x1Shape,
        int64_t* x2Shape,
        int64_t dimNum
    ) {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);

        this->rtol = rtol;
        this->atol = atol;
        this->dimNum = dimNum;
        for (int i = dimNum - 1; i >= 0; i--) {
            shapes[0][i] = x1Shape[i];
            shapes[1][i] = x2Shape[i];
            if (x1Shape[i] >= x2Shape[i])
                shapes[2][i] = x1Shape[i];
            else
                shapes[2][i] = x2Shape[i];
        }
        for (int i = 0; i < 3; i++) {
            strides[i][dimNum - 1] = 1;
        }
        this->totalNum = shapes[2][dimNum - 1];
        for (int i = dimNum - 2; i >= 0; i--) {
            strides[0][i] = strides[0][i + 1] * shapes[0][i + 1];
            strides[1][i] = strides[1][i + 1] * shapes[1][i + 1];
            strides[2][i] = strides[2][i + 1] * shapes[2][i + 1];
            totalNum *= shapes[2][i];
        }
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < totalNum; i++) {
            int x1Offset = 0;
            int x2Offset = 0;
            int outOffset = 0;
            for (int j = 0; j < dimNum; j++) {
                int index = i / strides[2][j] % shapes[2][j];
                x1Offset += (index % shapes[0][j]) * strides[0][j];
                x2Offset += (index % shapes[1][j]) * strides[1][j];
                outOffset += index * strides[2][j];
            }
            T x1 = x1Gm.GetValue(x1Offset);
            T x2 = x2Gm.GetValue(x2Offset);
            float x1fp32, x2fp32;
            if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int32_t>) {
                x1fp32 = (float)x1;
                x2fp32 = (float)x2;
            }
            else if constexpr(std::is_same_v<T, uint8_t>) {
                int32_t x1i32 = (int32_t)x1;
                int32_t x2i32 = (int32_t)x2;
                x1fp32 = (float)x1i32;
                x2fp32 = (float)x2i32;
            }
            else {
                x1fp32 = x1;
                x2fp32 = x2;
            }
            float threshold = atol + rtol * FAbs(x2fp32);
            float diff = FAbs(x1fp32 - x2fp32);
            if (diff <= threshold)
                yGm.SetValue(i, 1);
            else
                yGm.SetValue(i, 0);
        }
    }

private:
    __aicore__ inline float FAbs(const float &val) {
        if (val >= 0)
            return val;
        return -val;
    }

private:
    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;

    // 分别存储用于 x1, x2, y
    int64_t shapes[3][10], strides[3][10];
    float rtol, atol;
    int64_t dimNum, totalNum;
};

template <typename T>
class KernelIsCloseNoBrcb {
public:
    __aicore__ inline KernelIsCloseNoBrcb() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        float rtol,
        float atol,
        int64_t tileLen,
        int64_t tileNum,
        int64_t tailTileLen
    ) {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);

        int64_t dtypeSize = sizeof(T);
        int64_t tileLenAlign64 = (tileLen + 63) / 64 * 64;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLenAlign64 * dtypeSize);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLenAlign64 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLenAlign64);
        pipe.InitBuffer(yfp16Buf, tileLenAlign64 * sizeof(half));
        pipe.InitBuffer(maskBuf, tileLenAlign64);
        if constexpr (
            std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint8_t> ||
            std::is_same_v<T, half>
        ) {
            pipe.InitBuffer(buf1, tileLenAlign64 * sizeof(float));
            pipe.InitBuffer(buf2, tileLenAlign64 * sizeof(float));
        }
        pipe.InitBuffer(zeroBuf, tileLenAlign64 * sizeof(half));
        LocalTensor<half> zeros = zeroBuf.Get<half>();
        Duplicate(zeros, half(0), tileLenAlign64);

        this->rtol = rtol;
        this->atol = atol;
        this->tileLen = tileLen;
        this->tileNum = tileNum;
        this->tailTileLen = tailTileLen;
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLen, tileLen, 1);
            Compute(tileLen);
            CopyOut(i * tileLen, tileLen, 1);
        }
        if (tailTileLen > 0) {
            CopyIn(tileNum * tileLen, tailTileLen, 1);
            Compute(tailTileLen);
            CopyOut(tileNum * tileLen, tailTileLen, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(T);
        LocalTensor<T> x1 = x1Queue.AllocTensor<T>();
        LocalTensor<T> x2 = x2Queue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<T>(x1);
        x2Queue.EnQue<T>(x2);
    }

    __aicore__ inline void Compute(int64_t computeLen) {
        LocalTensor<T> x1 = x1Queue.DeQue<T>();
        LocalTensor<T> x2 = x2Queue.DeQue<T>();
        LocalTensor<int8_t> y = yQueue.AllocTensor<int8_t>();
        LocalTensor<half> yfp16 = yfp16Buf.Get<half>();
        LocalTensor<int8_t> mask = maskBuf.Get<int8_t>();
        LocalTensor<half> zeros = zeroBuf.Get<half>();

        if constexpr (std::is_same_v<T, float>) {
            Sub(x1, x1, x2, computeLen);
            Abs(x1, x1, computeLen);
            Abs(x2, x2, computeLen);
            Muls(x2, x2, rtol, computeLen);
            Adds(x2, x2, atol, computeLen);
            Compare(mask, x1, x2, CMPMODE::GT, (computeLen + 63) / 64 * 64);
        }
        else {
            LocalTensor<float> x1fp32 = buf1.Get<float>();
            LocalTensor<float> x2fp32 = buf2.Get<float>();
            if constexpr (std::is_same_v<T, uint8_t>) {
                LocalTensor<half> x1fp16 = buf1.Get<half>();
                LocalTensor<half> x2fp16 = buf2.Get<half>();
                Cast(x1fp16, x1, RoundMode::CAST_NONE, computeLen);
                Cast(x2fp16, x2, RoundMode::CAST_NONE, computeLen);
                Cast(x1fp32, x1fp16, RoundMode::CAST_NONE, computeLen);
                Cast(x2fp32, x2fp16, RoundMode::CAST_NONE, computeLen);
            }
            else {
                Cast(x1fp32, x1, RoundMode::CAST_NONE, computeLen);
                Cast(x2fp32, x2, RoundMode::CAST_NONE, computeLen);
            }
            Sub(x1fp32, x1fp32, x2fp32, computeLen);
            Abs(x1fp32, x1fp32, computeLen);
            Abs(x2fp32, x2fp32, computeLen);
            Muls(x2fp32, x2fp32, rtol, computeLen);
            Adds(x2fp32, x2fp32, atol, computeLen);
            Compare(mask, x1fp32, x2fp32, CMPMODE::GT, (computeLen + 63) / 64 * 64);
        }
        Select(yfp16, mask, zeros, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLen);
        Cast(y, yfp16, RoundMode::CAST_NONE, computeLen);
        yQueue.EnQue<int8_t>(y);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(int8_t);
        LocalTensor<int8_t> y = yQueue.DeQue<int8_t>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> buf1, buf2, yfp16Buf;
    TBuf<QuePosition::VECCALC> maskBuf, zeroBuf;

    int64_t tileLen, tileNum, tailTileLen;
    float rtol, atol;
};


extern "C" __global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    if (TILING_KEY_IS(1)) {
        KernelIsCloseNoBrcb<DTYPE_X1> op;
        op.Init(x1, x2, y, t.rtol, t.atol, t.tileLen, t.tileNum, t.tailTileLen);
        op.Process();
    }
    else if (TILING_KEY_IS(2)) {
        KernelIsClose<DTYPE_X1> op;
        op.Init(x1, x2, y, t.rtol, t.atol, t.x1Shape, t.x2Shape, t.dimNum);
        op.Process();
    }
}