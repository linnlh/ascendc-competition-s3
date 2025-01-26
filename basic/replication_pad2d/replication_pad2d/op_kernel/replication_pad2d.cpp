#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelReplicationPad2d {

constexpr static int64_t elemPerBlk = 32 / sizeof(T);

public:
    __aicore__ inline KernelReplicationPad2d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR paddings,
        GM_ADDR y,
        int64_t width,
        int64_t height,
        int64_t tileLen,
        int64_t tileNum,
        int64_t tailTileLen
    ) {
        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        paddingsGm.SetGlobalBuffer((__gm__ int32_t*)paddings);
        for (int i = 0; i < 4; i++) {
            this->paddings[0][i] = paddingsGm.GetValue(i);
            this->paddings[1][i] =  (this->paddings[0][i] + elemPerBlk - 1) / elemPerBlk * elemPerBlk;
        }

        int64_t widthAlign32 = (width + elemPerBlk - 1) / elemPerBlk * elemPerBlk;
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLen * widthAlign32 * sizeof(T));
        pipe.InitBuffer(paddingQ1, BUFFER_NUM, tileLen * this->paddings[1][0] * sizeof(T));
        pipe.InitBuffer(paddingQ2, BUFFER_NUM, tileLen * this->paddings[1][1] * sizeof(T));

        this->width = width;
        this->height = height;
        this->tileLen = tileLen;
        this->tileNum = tileNum;
        this->tailTileLen = tailTileLen;
        
        for (int i = 0; i < 4; i++) {
            strides[i] = 0;
        }
        if (tileLen > 1) {
            int64_t widthPad = width + this->paddings[0][0] + this->paddings[0][1];
            int64_t heightPad = height + this->paddings[0][2] + this->paddings[0][3];
            strides[0] = (width * (height - 1)) * sizeof(T);
            strides[1] = (widthPad * heightPad - this->paddings[0][0]) * sizeof(T);
            strides[2] = (widthPad * heightPad - width) * sizeof(T);
            strides[3] = (widthPad * heightPad - this->paddings[0][1]) * sizeof(T);
        }
    }

    __aicore__ inline void Process() {
        // printf("width: %ld\n", width);
        // printf("height: %ld\n", height);
        // printf("tileLen: %ld\n", tileLen);
        // printf("tileNum: %ld\n", tileNum);
        // printf("tailTileLen: %ld\n", tailTileLen);
        // printf("paddings: ");
        // PrintArray(paddings[0], 4);
        // printf("paddings align 32: ");
        // PrintArray(paddings[1], 4);
        // printf("strides: ");
        // PrintArray(strides, 4);
        int64_t widthPad = width + paddings[0][0] + paddings[0][1];
        int64_t heightPad = height + paddings[0][2] + paddings[0][3];
        for (int i = 0; i < tileNum; i++) {
            int64_t xOffset = i * tileLen * width * height;
            int64_t yOffset = i * tileLen * widthPad * heightPad;
            CopyIn(xOffset, width, tileLen);
            CopyOut(paddings[0][2] + 1, yOffset, width, tileLen);
            xOffset += width;
            yOffset += widthPad * (paddings[0][2] + 1);
            
            for (int j = 1; j < height - 1; j++) {
                CopyIn(xOffset, width, tileLen);
                CopyOut(1, yOffset, width, tileLen);
                xOffset += width;
                yOffset += widthPad;
            }

            CopyIn(xOffset, width, tileLen);
            CopyOut(paddings[0][3] + 1, yOffset, width, tileLen);
            xOffset += width;
            yOffset += widthPad * (paddings[0][3] + 1);
        }

        if (tailTileLen) {
            int64_t xOffset = tileNum * tileLen * width * height;
            int64_t yOffset = tileNum * tileLen * widthPad * heightPad;
            CopyIn(xOffset, width, tailTileLen);
            CopyOut(paddings[0][2] + 1, yOffset, width, tailTileLen);
            xOffset += width;
            yOffset += widthPad * (paddings[0][2] + 1);
            
            for (int j = 1; j < height - 1; j++) {
                CopyIn(xOffset, width, tailTileLen);
                CopyOut(1, yOffset, width, tailTileLen);
                xOffset += width;
                yOffset += widthPad;
            }

            CopyIn(xOffset, width, tailTileLen);
            CopyOut(paddings[0][3] + 1, yOffset, width, tailTileLen);
            xOffset += width;
            yOffset += widthPad * (paddings[0][3] + 1);
        }
    }

private:
    __aicore__ inline void PrintArray(int64_t* arr, int64_t len) {
        for (int i = 0; i < len; i++) {
            printf("%ld, ", arr[i]);
        }
        printf("\n");
    }
    __aicore__ inline void CopyIn(
        int64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(T);
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, dataSize, (uint32_t)strides[0], 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<T>(x);

        LocalTensor<T> padding1 = paddingQ1.AllocTensor<T>();
        LocalTensor<T> padding2 = paddingQ2.AllocTensor<T>();
        for (int i = 0; i < tileLen; i++) {
            T left = xGm.GetValue(offset + i * width * height);
            T right = xGm.GetValue(offset + i * width * height + width - 1);
            Duplicate(padding1[i * paddings[1][0]], left, paddings[0][0]);
            Duplicate(padding2[i * paddings[1][1]], right, paddings[0][1]);
        }
        paddingQ1.EnQue(padding1);
        paddingQ2.EnQue(padding2);
    }

    __aicore__ inline void CopyOut(
        uint16_t loop,
        int64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(T);
        uint32_t paddingSize1 = paddings[0][0] * sizeof(T);
        uint32_t paddingSize2 = paddings[0][1] * sizeof(T);
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<T> padding1 = paddingQ1.DeQue<T>();
        LocalTensor<T> padding2 = paddingQ2.DeQue<T>();
        for (int i = 0; i < loop; i++) {
            DataCopyPad(yGm[offset], padding1, {repeatTime, paddingSize1, 0, (uint32_t)strides[1], 0});
            offset += paddings[0][0];
            DataCopyPad(yGm[offset], x, {repeatTime, dataSize, 0, (uint32_t)strides[2], 0});
            offset += dataLen;
            DataCopyPad(yGm[offset], padding2, {repeatTime, paddingSize2, 0, (uint32_t)strides[3], 0});
            offset += paddings[0][1];
        }
        paddingQ1.FreeTensor(padding1);
        paddingQ2.FreeTensor(padding2);
        xQueue.FreeTensor(x);
    }


private:
    TPipe pipe;

    GlobalTensor<T> xGm, yGm;
    GlobalTensor<int32_t> paddingsGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> paddingQ1, paddingQ2;

    int64_t tileLen, tileNum, tailTileLen;
    int64_t width, height;
    int64_t paddings[2][4];
    // 分别存储 copyout 时 leftPadding, x, rightPadding 的步长
    int64_t strides[4];
};

extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelReplicationPad2d<DTYPE_X> op;
    op.Init(x, paddings, y, t.width, t.height, t.tileLen, t.tileNum, t.tailTileLen);
    op.Process();
}