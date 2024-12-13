#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

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
        bool equalNan,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y);

        this->rtol = rtol;
        this->atol = atol;
        this->equalNan = equalNan;
        this->tileLengthAlign32 = tileLengthAlign32;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
    }

    __aicore__ inline void Process() {
        int total = tileNum * tileLength + tailTileLength;
        for (int i = 0; i < total; i++) {
            T x1 = x1Gm.GetValue(i);
            T x2 = x2Gm.GetValue(i);
            float x1fp32 = (float)x1;
            float x2fp32 = (float)x2;
            float threshold = atol + rtol * FloatAbs(x2fp32);
            float diff = FloatAbs(x1fp32 - x2fp32);
            if (diff <= threshold)
                yGm.SetValue(i, 1);
            else
                yGm.SetValue(i, 0);
        }
    }

private:
    __aicore__ inline float FloatAbs(const float &val) {
        if (val >= 0)
            return val;
        return -val;
    }
    __aicore__ inline void CopyIn(uint64_t offset, int blockLen, uint16_t repeatTime) {}

private:
    TPipe pipe;

    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<uint8_t> yGm;

    float rtol;
    float atol;
    bool equalNan;
    int64_t tileLength;
    int64_t tileLengthAlign32;
    int64_t tileNum;
    int64_t tailTileLength;
};

// template <>
// class KernelIsClose<int32_t> {
// public:
//     __aicore__ inline KernelIsClose() {}

//     __aicore__ inline void Init(
//         GM_ADDR x1,
//         GM_ADDR x2,
//         GM_ADDR y,
//         float rtol,
//         float atol,
//         bool equalNan,
//         int64_t tileLength,
//         int64_t tileNum,
//         int64_t tailTileLength
//     ) {
        
//     }

//     __aicore__ inline void Process() {}
// };

template <>
class KernelIsClose<half> {
public:
    __aicore__ inline KernelIsClose() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        float rtol,
        float atol,
        bool equalNan,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ half*)x1);
        x2Gm.SetGlobalBuffer((__gm__ half*)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y);

        int64_t dtypeSize = 4;
        int64_t tileLengthAlign32 = (tileLength * dtypeSize + 31) / 32 * 32 / dtypeSize;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32);
        pipe.InitBuffer(x1fp32Buf, tileLengthAlign32 * sizeof(float));
        pipe.InitBuffer(x2fp32Buf, tileLengthAlign32 * sizeof(float));

        this->rtol = rtol;
        this->atol = atol;
        this->equalNan = equalNan;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength, 1);
            Compute(tileLength);
            CopyOut(i * tileLength, tileLength, 1);
        }
        if (tailTileLength > 0) {
            CopyIn(tileNum * tileLength, tailTileLength, 1);
            Compute(tailTileLength);
            CopyOut(tileNum * tileLength, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(half);
        LocalTensor<half> x1 = x1Queue.AllocTensor<half>();
        LocalTensor<half> x2 = x2Queue.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<half>(x1);
        x2Queue.EnQue<half>(x2);
    }

    __aicore__ inline void Compute(int64_t computeLen) {
        LocalTensor<half> x1 = x1Queue.DeQue<half>();
        LocalTensor<half> x2 = x2Queue.DeQue<half>();
        LocalTensor<uint8_t> y = yQueue.AllocTensor<uint8_t>();
        LocalTensor<float> x1fp32 = x1fp32Buf.Get<float>();
        LocalTensor<float> x2fp32 = x2fp32Buf.Get<float>();
        Cast(x1fp32, x1, RoundMode::CAST_NONE, computeLen);
        Cast(x2fp32, x2, RoundMode::CAST_NONE, computeLen);
        
        Sub(x1fp32, x1fp32, x2fp32, computeLen);
        Abs(x1fp32, x1fp32, computeLen);
        Abs(x2fp32, x2fp32, computeLen);
        Muls(x2fp32, x2fp32, rtol, computeLen);
        Adds(x2fp32, x2fp32, atol, computeLen);
        Sub(x1fp32, x1fp32, x2fp32, computeLen);
        // Cast(x1, x1fp32, RoundMode::CAST_NONE, computeLen);
        for (int i = 0; i < computeLen; i++) {
            if (x1fp32.GetValue(i) > 0) {
                y.SetValue(i, 0);
            }
            else {
                y.SetValue(i, 1);
            }
        }
        yQueue.EnQue<uint8_t>(y);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(uint8_t);
        LocalTensor<uint8_t> y = yQueue.DeQue<uint8_t>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<half> x1Gm, x2Gm;
    GlobalTensor<uint8_t> yGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> x1fp32Buf, x2fp32Buf;

    float rtol;
    float atol;
    bool equalNan;
    int64_t tileLength;
    int64_t tileNum;
    int64_t tailTileLength;
};


template <>
class KernelIsClose<float> {
public:
    __aicore__ inline KernelIsClose() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        float rtol,
        float atol,
        bool equalNan,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength
    ) {
        x1Gm.SetGlobalBuffer((__gm__ float*)x1);
        x2Gm.SetGlobalBuffer((__gm__ float*)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y);

        int64_t dtypeSize = 4;
        int64_t tileLengthAlign32 = (tileLength * dtypeSize + 31) / 32 * 32 / dtypeSize;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, tileLengthAlign32 * dtypeSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLengthAlign32);
        pipe.InitBuffer(maskBuf, tileLengthAlign32);
        pipe.InitBuffer(zeroBuf, tileLengthAlign32 * sizeof(half));
        pipe.InitBuffer(resultBuf, tileLengthAlign32 * sizeof(half));

        this->rtol = rtol;
        this->atol = atol;
        this->equalNan = equalNan;
        this->tileLengthAlign32 = tileLengthAlign32;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
        this->tailTileLengthAlign32 = (tailTileLengthAlign32 + 7) / 8 * 8;
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            CopyIn(i * tileLength, tileLength, 1);
            Compute(tileLength);
            CopyOut(i * tileLength, tileLength, 1);
        }
        if (tailTileLength > 0) {
            CopyIn(tileNum * tileLength, tailTileLength, 1);
            Compute(tailTileLength);
            CopyOut(tileNum * tileLength, tailTileLength, 1);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(float);
        LocalTensor<float> x1 = x1Queue.AllocTensor<float>();
        LocalTensor<float> x2 = x2Queue.AllocTensor<float>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        DataCopyPad(x2, x2Gm[offset], copyParams, padParams);
        x1Queue.EnQue<float>(x1);
        x2Queue.EnQue<float>(x2);
    }

    __aicore__ inline void Compute(int64_t computeLen) {
        LocalTensor<float> x1 = x1Queue.DeQue<float>();
        LocalTensor<float> x2 = x2Queue.DeQue<float>();
        LocalTensor<uint8_t> y = yQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> zero = zeroBuf.Get<half>();
        LocalTensor<half> result = resultBuf.Get<half>();
        Duplicate(zero, half(0), computeLen);
        Sub(x1, x1, x2, computeLen);
        Abs(x1, x1, computeLen);
        Abs(x2, x2, computeLen);
        Muls(x2, x2, rtol, computeLen);
        Adds(x2, x2, atol, computeLen);
        Compare(mask, x1, x2, CMPMODE::GT, computeLen);
        Select(result, mask, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLen);
        Cast(y, result, RoundMode::CAST_ROUND, computeLen);
        // Sub(x1, x1, x2, computeLen);
        // for (int i = 0; i < computeLen; i++) {
        //     if (x1.GetValue(i) > 0) {
        //         y.SetValue(i, 0);
        //     }
        //     else {
        //         y.SetValue(i, 1);
        //     }
        // }
        yQueue.EnQue<uint8_t>(y);
        x1Queue.FreeTensor(x1);
        x2Queue.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        uint32_t blockSize = blockLen * sizeof(uint8_t);
        LocalTensor<uint8_t> y = yQueue.DeQue<uint8_t>();
        DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<float> x1Gm, x2Gm;
    GlobalTensor<uint8_t> yGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> maskBuf, zeroBuf, resultBuf;

    float rtol;
    float atol;
    bool equalNan;
    int64_t tileLength;
    int64_t tileLengthAlign32;
    int64_t tileNum;
    int64_t tailTileLength;
    int64_t tailTileLengthAlign32;
};

extern "C" __global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelIsClose<DTYPE_X1> op;
    op.Init(
        x1,
        x2,
        y,
        t.rtol,
        t.atol,
        t.equalNan,
        t.tileLength,
        t.tileNum,
        t.tailTileLength
    );
    op.Process();
}