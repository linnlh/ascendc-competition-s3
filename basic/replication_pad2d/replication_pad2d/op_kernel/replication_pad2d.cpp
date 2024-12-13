#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelReplicationPad2d {
public:
    __aicore__ inline KernelReplicationPad2d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR paddings,
        GM_ADDR y,
        int64_t batch,
        int64_t width,
        int64_t height,
        int64_t tileLen
    ) {
        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        paddingsGm.SetGlobalBuffer((__gm__ int32_t*)paddings);

        leftPadding = paddingsGm.GetValue(0);
        rightPadding = paddingsGm.GetValue(1);
        topPadding = paddingsGm.GetValue(2);
        bottomPadding = paddingsGm.GetValue(3);

        int64_t dtypeSize = sizeof(T);
        int64_t tileLenAlign32 = (tileLen * dtypeSize + 31) / 32 * 32;
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLenAlign32);
        pipe.InitBuffer(paddingsInQ, BUFFER_NUM, 32);
        if (leftPadding >= rightPadding) {
            int64_t leftPaddingAlign32 = (leftPadding * dtypeSize + 31) / 32 * 32;
            pipe.InitBuffer(paddingsOutQ, BUFFER_NUM, leftPaddingAlign32);
        }
        else {
            int64_t rightPaddingAlign32 = (rightPadding * dtypeSize + 31) / 32 * 32;
            pipe.InitBuffer(paddingsOutQ, BUFFER_NUM, rightPaddingAlign32);
        }

        this->batch = batch;
        this->width = width;
        this->height = height;
        this->tileLen = tileLen;
        this->tileLenAlign32 = tileLenAlign32;

        // printf("batch: %ld\n", batch);
        // printf("width: %ld\n", width);
        // printf("height: %ld\n", height);
        // printf("tileLen: %ld\n", tileLen);
    }

    __aicore__ inline void Process() {
        int32_t xOffset = 0;
        int32_t yOffset = 0;
        int32_t leftPaddingAlign32 = (leftPadding * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        int32_t rightPaddingAlign32 = (rightPadding * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < topPadding; i++) {
                CopyInPadding(xOffset, 1, 1);
                LocalTensor<T> leftPadIn = paddingsInQ.DeQue<T>();
                LocalTensor<T> leftPadOut = paddingsOutQ.AllocTensor<T>();
                Duplicate(leftPadOut, leftPadIn.GetValue(0), leftPadding);
                paddingsOutQ.EnQue<T>(leftPadOut);
                paddingsInQ.FreeTensor(leftPadIn);
                CopyOutPadding(yOffset, leftPadding, 1);
                yOffset += leftPadding;

                CopyInData(xOffset, tileLen, 1);
                CopyOutData(yOffset, tileLen, 1);
                yOffset += tileLen;

                CopyInPadding(xOffset + tileLen - 1, 1, 1);
                LocalTensor<T> rightPadIn = paddingsInQ.DeQue<T>();
                LocalTensor<T> rightPadOut = paddingsOutQ.AllocTensor<T>();
                Duplicate(rightPadOut, rightPadIn.GetValue(0), rightPadding);
                paddingsOutQ.EnQue<T>(rightPadOut);
                paddingsInQ.FreeTensor(rightPadIn);
                CopyOutPadding(yOffset, rightPadding, 1);
                yOffset += rightPadding;
            }
            for (int i = 0; i < height; i++) {
                CopyInPadding(xOffset, 1, 1);
                LocalTensor<T> leftPadIn = paddingsInQ.DeQue<T>();
                LocalTensor<T> leftPadOut = paddingsOutQ.AllocTensor<T>();
                Duplicate(leftPadOut, leftPadIn.GetValue(0), leftPadding);
                paddingsOutQ.EnQue<T>(leftPadOut);
                paddingsInQ.FreeTensor(leftPadIn);
                CopyOutPadding(yOffset, leftPadding, 1);
                yOffset += leftPadding;

                CopyInData(xOffset, tileLen, 1);
                CopyOutData(yOffset, tileLen, 1);
                yOffset += tileLen;

                CopyInPadding(xOffset + tileLen - 1, 1, 1);
                LocalTensor<T> rightPadIn = paddingsInQ.DeQue<T>();
                LocalTensor<T> rightPadOut = paddingsOutQ.AllocTensor<T>();
                Duplicate(rightPadOut, rightPadIn.GetValue(0), rightPadding);
                paddingsOutQ.EnQue<T>(rightPadOut);
                paddingsInQ.FreeTensor(rightPadIn);
                CopyOutPadding(yOffset, rightPadding, 1);
                yOffset += rightPadding;
                xOffset += tileLen;
            }
            for (int i = 0; i < bottomPadding; i++) {
                CopyInPadding(xOffset - tileLen, 1, 1);
                LocalTensor<T> leftPadIn = paddingsInQ.DeQue<T>();
                LocalTensor<T> leftPadOut = paddingsOutQ.AllocTensor<T>();
                Duplicate(leftPadOut, leftPadIn.GetValue(0), leftPadding);
                paddingsOutQ.EnQue<T>(leftPadOut);
                paddingsInQ.FreeTensor(leftPadIn);
                CopyOutPadding(yOffset, leftPadding, 1);
                yOffset += leftPadding;

                CopyInData(xOffset - tileLen, tileLen, 1);
                CopyOutData(yOffset, tileLen, 1);
                yOffset += tileLen;

                CopyInPadding(xOffset -  1, 1, 1);
                LocalTensor<T> rightPadIn = paddingsInQ.DeQue<T>();
                LocalTensor<T> rightPadOut = paddingsOutQ.AllocTensor<T>();
                Duplicate(rightPadOut, rightPadIn.GetValue(0), rightPadding);
                paddingsOutQ.EnQue<T>(rightPadOut);
                paddingsInQ.FreeTensor(rightPadIn);
                CopyOutPadding(yOffset, rightPadding, 1);
                yOffset += rightPadding;
            }
        }
    }

private:
    __aicore__ inline void CopyInPadding(int64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> paddingData = paddingsInQ.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(paddingData, xGm[offset], copyParams, padParams);
        paddingsInQ.EnQue<T>(paddingData);
    }
    __aicore__ inline void CopyInData(int64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> data = xQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(data, xGm[offset], copyParams, padParams);
        xQueue.EnQue<T>(data);
    }
    __aicore__ inline void CopyOutPadding(int64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> paddingData = paddingsOutQ.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], paddingData, copyParams);
        paddingsOutQ.FreeTensor(paddingData);
    }
    __aicore__ inline void CopyOutData(int64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> data = xQueue.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], data, copyParams);
        xQueue.FreeTensor(data);
    }
    // __aicore__ inline void CopyInTop(int64_t offset) {
    //     LocalTensor<T> leftPadding = leftPaddingQ.AllocTensor<T>();
    //     LocalTensor<T> data = xQueue.AllocTensor<T>();
    //     LocalTensor<T> rightPadding = rightPaddingQ.AllocTensor<T>();

    //     DataCopy(leftPadding, xGm[xOffset], 32 / sizeof(T));
    //     leftPaddingQ.EnQue<T>(leftPadding);

    //     DataCopy(data, xGm[xOffset], tileLenAlign32);
    //     xQueue.EnQue<T>(data);

    //     DataCopy(rightPadding, xGm[xOffset + tileLen - 1], 32 / sizeof(T));
    //     rightPaddingQ.EnQue<T>(rightPadding);

    //     // for (int i = 0; i < paddings[3]; i++) {
    //     //     DataCopy(leftPadding, xGm[xOffset], 32 / sizeof(T));
    //     //     Duplicate(leftPadding, leftPadding.GetValue(0), paddings[0]);
    //     //     leftPaddingQ.EnQue<T>(leftPadding);
    //     //     leftPadding = leftPaddingQ.DeQue<T>();
    //     //     DataCopy(yGm[yOffset], leftPadding, paddingsAlign32[0]);
    //     //     yOffset += paddings[0];

    //     //     DataCopy(data, xGm[xOffset], tileLenAlign32);
    //     //     xQueue.EnQue<T>(data);
    //     //     data = xQueue.DeQue<T>();
    //     //     DataCopy(yGm[yOffset], data, tileLenAlign32);
    //     //     yOffset += tileLen;

    //     //     DataCopy(rightPadding, xGm[xOffset + tileLen - 1], 32 / sizeof(T));
    //     //     Duplicate(rightPadding, rightPadding.GetValue(0), paddings[1]);
    //     //     rightPaddingQ.EnQue<T>(rightPadding);
    //     //     rightPadding = rightPaddingQ.DeQue<T>();
    //     //     DataCopy(yGm[yOffset], rightPadding, paddingsAlign32[1]);
    //     //     yOffset += paddings[1];
    //     // }
    //     // leftPaddingQ.FreeTensor(leftPadding);
    //     // xQueue.FreeTensor(data);
    //     // rightPaddingQ.FreeTensor(rightPadding);
    // }
    // __aicore__ inline void CopyOutTop(int64_t offset) {
    //     LocalTensor<T> leftPadding = leftPaddingQ.DeQue<T>();
    //     LocalTensor<T> data = xQueue.DeQue<T>();
    //     LocalTensor<T> rightPadding = rightPaddingQ.DeQue<T>();

    //     Duplicate(leftPadding, leftPadding.GetValue(0), paddings[0]);

    //     leftPaddingQ.FreeTensor(leftPadding);
    //     xQueue.FreeTensor(data);
    //     rightPaddingQ.FreeTensor(rightPadding);
    // }
    // __aicore__ inline void CopyInData(uint64_t offset) {
    //     // uint32_t blockSize = blockLen * sizeof(T);
    //     // LocalTensor<T> x = xQueue.AllocTensor<T>();
    //     // DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
    //     // DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
    //     // DataCopyPad(x, xGm[offset], copyParams, padParams);
    //     // xQueue.EnQue<T>(x);
    // }
    // __aicore__ inline void CopyInBottom(uint64_t offset) {

    // }

    __aicore__ inline void Compute() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint64_t offset, int blockLen, uint16_t repeatTime) {
        // uint32_t blockSize = blockLen * sizeof(T);
        // LocalTensor<T> y = yQueue.DeQue<T>();
        // DataCopyExtParams copyParams {repeatTime, blockSize, 0, 0, 0};
        // DataCopyPad(yGm[offset], y, copyParams);
        // yQueue.FreeTensor(y);
    }

private:
    TPipe pipe;

    GlobalTensor<T> xGm, yGm;
    GlobalTensor<int32_t> paddingsGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> paddingsInQ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> paddingsOutQ;

    int64_t tileLen;
    int64_t tileLenAlign32;
    int64_t batch;
    int64_t width;
    int64_t height;
    int64_t leftPadding, rightPadding, topPadding, bottomPadding;
};

extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelReplicationPad2d<DTYPE_X> op;
    op.Init(
        x,
        paddings,
        y,
        t.batch,
        t.width,
        t.height,
        t.tileLen
    );
    op.Process();
}