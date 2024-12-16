#include "kernel_operator.h"

using namespace AscendC;

constexpr int BUFFER_NUM = 2;

template <typename T>
class KernelNotEqual {
public:
    __aicore__ inline KernelNotEqual() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        int64_t* x1Shape,
        int64_t* x2Shape,
        int64_t dimNum
    ) {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);

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
            if constexpr (std::is_same_v<T, half>) {
                if ((float)x1 != (float)x2)
                    yGm.SetValue(outOffset, 1);
                else
                    yGm.SetValue(outOffset, 0);
            }
            else {
                if (x1 != x2)
                    yGm.SetValue(outOffset, 1);
                else
                    yGm.SetValue(outOffset, 0);
            }
        }
    }

private:
    __aicore__ inline void PrintArray(int64_t* arr, int64_t len) {
        for (int i = 0; i < len; i++) {
            printf("%ld, ", arr[i]);
        }
        printf("\n");
    }

private:
    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;
    int64_t dimNum, totalNum;
    int64_t shapes[3][10];
    int64_t strides[3][10];
};

template <typename T>
class KernelNotEqualNoBrcb {
public:
    __aicore__ inline KernelNotEqualNoBrcb() {}
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        int64_t tileLen,
        int64_t tileNum,
        int64_t tailTileLen
    ) {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);

        size_t dtypeSize = sizeof(T);
        tileLenAlign256 = (tileLen * dtypeSize + 255) / 256 * 256 / dtypeSize;
        tailTileLenAlign256 = (tailTileLen * dtypeSize + 255) / 256 * 256 / dtypeSize;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLenAlign256 * dtypeSize);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLenAlign256 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLenAlign256);
        pipe.InitBuffer(yfp16Buf, tileLenAlign256 * sizeof(half));
        pipe.InitBuffer(maskBuf, tileLenAlign256);
        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>) {
            pipe.InitBuffer(buf1, tileLenAlign256 * sizeof(float));
            pipe.InitBuffer(buf2, tileLenAlign256 * sizeof(float));
        }

        this->tileLen = tileLen;
        this->tileNum = tileNum;
        this->tailTileLen = tailTileLen;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLen, tileLen, 1);
            Compute(tileLenAlign256);
            CopyOut(i * tileLen, tileLen, 1);
        }
        if (tailTileLen > 0) {
            CopyIn(tileNum * tileLen, tailTileLen, 1);
            Compute(tailTileLenAlign256);
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
        Duplicate<half>(yfp16, .0f, computeLen);
        if constexpr (std::is_same_v<T, int32_t>) {
            LocalTensor<float> x1fp32 = buf1.Get<float>();
            LocalTensor<float> x2fp32 = buf2.Get<float>();
            Cast(x1fp32, x1, RoundMode::CAST_NONE, computeLen);
            Cast(x2fp32, x2, RoundMode::CAST_NONE, computeLen);
            Compare(mask, x1fp32, x2fp32, CMPMODE::EQ, computeLen);
            Select<half>(yfp16, mask, yfp16, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLen);
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            LocalTensor<half> x1fp16 = buf1.Get<half>();
            LocalTensor<half> x2fp16 = buf2.Get<half>();
            Cast(x1fp16, x1, RoundMode::CAST_NONE, computeLen);
            Cast(x2fp16, x2, RoundMode::CAST_NONE, computeLen);
            Compare(mask, x1fp16, x2fp16, CMPMODE::EQ, computeLen);
            Select<half>(yfp16, mask, yfp16, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLen);
        }
        else {
            Compare(mask, x1, x2, CMPMODE::EQ, computeLen);
            Select<half>(yfp16, mask, yfp16, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLen);
        }
        Cast(y, yfp16, RoundMode::CAST_ROUND, computeLen);
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
    TPipe pipe;
    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> buf1, buf2, yfp16Buf;
    TBuf<QuePosition::VECCALC> maskBuf;

    int64_t tileLen, tileNum, tailTileLen;
    int64_t tileLenAlign256, tailTileLenAlign256;
};


extern "C" __global__ __aicore__ void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    if (TILING_KEY_IS(1)) {
        KernelNotEqualNoBrcb<DTYPE_X1> op;
        op.Init(x1, x2, y, t.tileLen, t.tileNum, t.tailTileLen);
        op.Process();
    }
    else if (TILING_KEY_IS(2)) {
        KernelNotEqual<DTYPE_X1> op;
        op.Init(x1, x2, y, t.x1Shape, t.x2Shape, t.dimNum);
        op.Process();
    }
}