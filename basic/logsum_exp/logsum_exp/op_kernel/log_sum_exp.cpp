#include "kernel_operator.h"

using namespace AscendC;
constexpr int BUFFER_NUM = 2;
constexpr int BLOCK_SIZE = 32;

template <typename T>
class KernelLogSumExp {
public:
    __aicore__ inline KernelLogSumExp() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        int64_t tileLen,
        int64_t tileNum,
        int64_t* shape,
        int64_t shapeLen,
        int64_t* dim,
        int64_t* strides,
        int64_t dimLen
    ) {
        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        int64_t tileLenAlign32 = (tileLen * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLenAlign32 * 32);
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLenAlign32 * sizeof(T));
        pipe.InitBuffer(workBuf, tileLenAlign32 * sizeof(float));
        pipe.InitBuffer(maskBuf, 256);
        pipe.InitBuffer(maxBuf, 32);
        pipe.InitBuffer(sumBuf, 32);
        if constexpr (std::is_same_v<T, half>) {
            pipe.InitBuffer(xfp32Buf, tileLenAlign32 * sizeof(float));
        }

        this->tileLen = tileLen;
        this->tileLenAlign32 = tileLenAlign32;
        this->tileNum = tileNum;
        this->shape = shape;
        this->shapeLen = shapeLen;
        this->dim = dim;
        this->strides = strides;
        this->dimLen = dimLen;
    }

    __aicore__ inline void Process() {
        // printf("[tileLen] %ld\n", tileLen);
        // printf("[tileNum] %ld\n", tileNum);
        // printf("[shapeLen] %ld\n", shapeLen);
        // printf("[shape] ");
        // PrintArray(shape, shapeLen);
        // printf("[dimLen] %ld\n", dimLen);
        // printf("[dim] ");
        // PrintArray(dim, dimLen);
        // printf("[stride] ");
        // PrintArray(strides, dimLen);
        if (dimLen > 1) {
            ProcessDimSizeGT1();
        }
        else {
            ProcessDimSizeEq1();
        }
    }

private:
    __aicore__ inline void ProcessDimSizeEq1() {
        if (strides[0] == 1) {
            for (int i = 0; i < tileNum; i++) {
                CopyIn(i * tileLen, tileLen, 1);
                Compute();
                CopyOut(i, 1, 1);
            }
        }
        else {
            int32_t stride = strides[0];
            for (int i = 0; i < tileNum; i++) {
                int64_t offset = i / stride * stride * shape[dim[0]] + i % stride;
                CopyInUncontinuous(offset, 1, tileLen, stride - 1, 0);
                ComputeUncontinuous();
                CopyOut(i, 1, 1);
            }
        }
    }

    __aicore__ inline void ProcessDimSizeGT1() {
        int32_t dim0ToDim1Size = strides[0] / (strides[1] * shape[dim[1]]);
        int32_t dim0Size = strides[0] * shape[dim[0]];
        int32_t dim1Size = strides[1] * shape[dim[1]];
        if (dimLen == 2 && strides[dimLen - 1] == 1) {
            // int32_t gap = strides[0] / (strides[1] * shape[dim[1]])
            for (int i = 0; i < tileNum; i++) {
                LocalTensor<float> max = maxBuf.Get<float>();
                LocalTensor<float> sum = sumBuf.Get<float>();
                max.SetValue(0, -1e30f);
                sum.SetValue(0, .0f);
                int32_t offset = (i / dim0ToDim1Size * dim0Size) + (i % dim0ToDim1Size * dim1Size);
                // int32_t offset = i / gap * strides[0] * shape[dim[0]] + (i % gap) * shape[dim[1]] * strides[1];
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyIn(offset + j * strides[0], tileLen, 1);
                    ComputeMax();
                }
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyIn(offset + j * strides[0], tileLen, 1);
                    ComputeSum();
                }
                LocalTensor<T> y = yQueue.AllocTensor<T>();
                Ln(sum, sum, 8);
                if constexpr (std::is_same_v<T, half>) {
                    Add(sum, sum, max, 8);
                    Cast(y, sum, RoundMode::CAST_NONE, 8);
                }
                else {
                    Add(y, sum, max, 8);
                }
                yQueue.EnQue<T>(y);
                CopyOut(i, 1, 1);
            }
        }
        else {
            for (int i = 0; i < tileNum; i++) {
                LocalTensor<float> max = maxBuf.Get<float>();
                LocalTensor<float> sum = sumBuf.Get<float>();
                max.SetValue(0, -1e30f);
                sum.SetValue(0, .0f);
                // [xxx, [i], xxx, [j] xxx]
                // dim0ToDim1Size: 1
                // strides1: 64
                // strides0: 2048
                int32_t part1 = i / (dim0ToDim1Size * strides[1]) * dim0Size;
                int32_t part2 = i % (dim0ToDim1Size * strides[1]) / strides[1] * dim1Size;
                int32_t part3 = i % strides[1];
                int32_t offset = part1 + part2 + part3;
                // printf("[part1] %d, [part2] %d, [part3] %d\n", part1, part2, part3);
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyInUncontinuous(offset + j * strides[0], 1, tileLen, strides[1] - 1, 0);
                    ComputeMaxUncontinuous();
                }
                for (int j = 0; j < shape[dim[0]]; j++) {
                    CopyInUncontinuous(offset + j * strides[0], 1, tileLen, strides[1] - 1, 0);
                    ComputeSumUncontinuous();
                }
                LocalTensor<T> y = yQueue.AllocTensor<T>();
                Ln(sum, sum, 8);
                if constexpr (std::is_same_v<T, half>) {
                    Add(sum, sum, max, 8);
                    Cast(y, sum, RoundMode::CAST_NONE, 8);
                }
                else {
                    Add(y, sum, max, 8);
                }
                yQueue.EnQue<T>(y);
                CopyOut(i, 1, 1);
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<T>(x);
    }

    __aicore__ inline void CopyInUncontinuous(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime,
        uint32_t srcStride,
        uint32_t dstStride
    ) {
        uint32_t blockLen = dataLen * sizeof(T);
        uint32_t srcStrideSize = srcStride * sizeof(T);
        uint32_t dstStrideSize = dstStride * sizeof(T);
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, srcStrideSize, dstStrideSize, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<T>(x);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<T> y = yQueue.AllocTensor<T>();
        LocalTensor<float> work = workBuf.Get<float>();

        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
            Ln(work, work, 8);
            Adds(work, work, maxVal, 8);
            Cast(y, work, RoundMode::CAST_NONE, 8);
        }
        else {
            ReduceMax(work, x, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
            Ln(work, work, 8);
            Adds(y, work, maxVal, 8);
        }
        yQueue.EnQue<T>(y);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeUncontinuous() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<T> y = yQueue.AllocTensor<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Duplicate<uint16_t>(mask, 0x8000u, 128);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
            Ln(work, work, 8);
            Adds(work, work, maxVal, 8);
            Cast(y, work, RoundMode::CAST_NONE, 8);
        }
        else {
            LocalTensor<uint32_t> mask = maskBuf.Get<uint32_t>();
            Duplicate<uint32_t>(mask, 0x80808080u, 64);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            ReduceMax(work, x, work, tileLen);
            float maxVal = work.GetValue(0);
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
            Ln(work, work, 8);
            Adds(y, work, maxVal, 8);
        }
        yQueue.EnQue<T>(y);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeMax() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
        }
        else {
            ReduceMax(work, x, work, tileLen);
        }
        Max(max, max, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeSum() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> sum = sumBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        float maxVal = max.GetValue(0);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
        }
        else {
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
        }
        Add(sum, sum, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> y = yQueue.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

    __aicore__ inline void ComputeMaxUncontinuous() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Duplicate<uint16_t>(mask, 0x8000u, 128);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            ReduceMax(work, xfp32, work, tileLen);
        }
        else {
            LocalTensor<uint32_t> mask = maskBuf.Get<uint32_t>();
            Duplicate<uint32_t>(mask, 0x80808080u, 64);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            ReduceMax(work, x, work, tileLen);
        }
        Max(max, max, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeSumUncontinuous() {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> sum = sumBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        float maxVal = max.GetValue(0);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
            LocalTensor<float> xfp32 = xfp32Buf.Get<float>();
            Duplicate<uint16_t>(mask, 0x8000u, 128);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Cast(xfp32, x, RoundMode::CAST_NONE, tileLen);
            Adds(xfp32, xfp32, -maxVal, tileLen);
            Exp(xfp32, xfp32, tileLen);
            ReduceSum(work, xfp32, work, tileLen);
        }
        else {
            LocalTensor<uint32_t> mask = maskBuf.Get<uint32_t>();
            Duplicate<uint32_t>(mask, 0x80808080u, 64);
            uint16_t repeatTime = tileLenAlign32 / 8;
            uint64_t rsvdCnt = 0;
            GatherMask(x, x, mask, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
            Adds(x, x, -maxVal, tileLen);
            Exp(x, x, tileLen);
            ReduceSum(work, x, work, tileLen);
        }
        Add(sum, sum, work, 1);
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void PrintArray(int64_t* arr, int64_t arrLen) {
        for (int i = 0; i < arrLen; i++) {
            printf("%ld, ", arr[i]);
        }
        printf("\n");
    }

private:
    TPipe pipe;

    GlobalTensor<T> xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> xfp32Buf, workBuf, maskBuf;
    TBuf<QuePosition::VECCALC> maxBuf, sumBuf;

    int64_t tileLen;
    int64_t tileLenAlign32;
    int64_t tileNum;
    int64_t* shape;
    int64_t shapeLen;
    int64_t* dim;
    int64_t* strides;
    int64_t dimLen;
};

// 处理 dims 维度大于 1 的情况
// template <typename T>
// class KernelLogSumExp2 {
// constexpr static int elemPerBlk = BLOCK_SIZE / sizeof(T);
// public:
//     __aicore__ inline KernelLogSumExp2() {}
//     __aicore__ inline void Init(
//         TPipe* pipe,
//         GM_ADDR x,
//         GM_ADDR y,
//         int64_t tileLen,
//         int64_t tileNum,
//         int64_t* shape,
//         int64_t shapeLen,
//         int64_t* dim,
//         int64_t dimLen
//     ) {
//         // ASSERT(dim[dimLen - 1] != (shapeLen - 1) && "无法处理 dim 包含最后一轴的情况");
//         this->pipe = pipe;
//         this->outerTileLen = elemPerBlk;
//         this->outerTileNum = shape[shapeLen - 1] / outerTileLen;
//         this->outerTailTileLen = shape[shapeLen - 1] % outerTileLen;
//         this->innerTileLen = 128;
//         if (this->innerTileLen > shape[dim[dimLen - 1]]) {
//             this->innerTileLen = shape[dim[dimLen - 1]];
//         }
//         this->innerTileNum = shape[dim[dimLen - 1]] / this->innerTileLen;
//         this->innerTailTileLen = shape[dim[dimLen - 1]] % this->innerTileLen;

//         xGm.SetGlobalBuffer((__gm__ T*)x);
//         yGm.SetGlobalBuffer((__gm__ T*)y);

//         // 计算 dim 去除最后一维后的数据长度
//         this->tileElemBase = 1;
//         for (int i = 0; i < dimLen - 1; i++) {
//             tileElemBase *= shape[dim[i]];
//         }
//         pipe->InitBuffer(xQueue, BUFFER_NUM, (tileElemBase * innerTileLen * outerTileLen + elemPerBlk - 1) / elemPerBlk * BLOCK_SIZE);
//         pipe->InitBuffer(yQueue, BUFFER_NUM, (outerTileLen + elemPerBlk - 1) / elemPerBlk * elemPerBlk);
//         pipe->InitBuffer(workBuf, tileElemBase * innerTileLen * sizeof(float));
//         pipe->InitBuffer(maskBuf, 8 * BLOCK_SIZE);
//         pipe->InitBuffer(maxBuf, outerTileLen * BLOCK_SIZE);
//         pipe->InitBuffer(sumBuf, outerTileLen * BLOCK_SIZE);
//         this->tileLen = tileLen;
//         this->tileNum = tileNum;
//         this->shapeLen = shapeLen;
//         this->dimLen = dimLen;

//         // 对轴的定义进行改写，原先第一个轴称为第 0 轴，
//         // 现在第一个轴称为第 shapeLen - 1 轴
//         int dimOffset = 0;
//         for (int i = 0; i < shapeLen; i++) {
//             this->shape[i] = shape[shapeLen - 1 - i];
//             if (dim[dimOffset] == i) {
//                 this->dim[dimOffset] = shapeLen - 1 - i;
//                 dimOffset += 1;
//             }
//             else {
//                 tileDim[i - dimOffset] = shapeLen - 1 - i;
//             }
//         }
//         InitSliceInfo();
//     }

//     __aicore__ inline void Process() {
//         // printf("[tileLen] %ld\n", tileLen);
//         // printf("[tileNum] %ld\n", tileNum);
//         // printf("[shapeLen] %ld\n", shapeLen);
//         // printf("[shape] ");
//         // PrintArray(shape, shapeLen);
//         // printf("[dimLen] %ld\n", dimLen);
//         // printf("[dim] ");
//         // PrintArray(dim, dimLen);
//         // printf("[tileDim] ");
//         // printf("innerTileLen: %ld, innerTileNum: %ld, innerTailTileLen: %ld\n", innerTileLen, innerTileNum, innerTailTileLen);
//         // printf("outerTileLen: %ld, outerTileNum: %ld, outerTailTileLen: %ld\n", outerTileLen, outerTileNum, outerTailTileLen);
//         // PrintArray(tileDim, shapeLen - dimLen);
//         // printf("tileElemBase: %ld\n", tileElemBase);
//         LocalTensor<float> max = maxBuf.Get<float>();
//         LocalTensor<float> sum = sumBuf.Get<float>();
//         int64_t loopCount = tileNum / shape[0];
//         int64_t yOffset = 0;
//         for (int i = 0; i < loopCount; i++) {
//             SetUpOuter();
//             for (int j = 0 ; j < outerTileNum; j++) {
//                 Duplicate(max, -1e30f, outerTileLen * elemPerBlk);
//                 Duplicate(sum, .0f, outerTileLen * elemPerBlk);
//                 UpdateOuter(outerTileLen);
//                 // 计算最大值
//                 SetUpInner();
//                 for (int k = 0; k < innerTileNum; k++) {
//                     UpdateInner(innerTileLen);
//                     CopyIn();
//                     ComputeMax(innerTileLen * tileElemBase, outerTileLen);
//                 }
//                 if (innerTailTileLen) {
//                     UpdateInner(innerTailTileLen);
//                     CopyIn();
//                     ComputeMax(innerTailTileLen * tileElemBase, outerTileLen);
//                 }
//                 // 求和
//                 SetUpInner();
//                 for (int k = 0; k < innerTileNum; k++) {
//                     UpdateInner(innerTileLen);
//                     CopyIn();
//                     ComputeSum(innerTileLen * tileElemBase, outerTileLen);
//                 }
//                 if (innerTailTileLen) {
//                     UpdateInner(innerTailTileLen);
//                     CopyIn();
//                     ComputeSum(innerTailTileLen * tileElemBase, outerTileLen);
//                 }
//                 // 计算最终结果
//                 Compute(outerTileLen);
//                 CopyOut(yOffset, outerTileLen, 1);
//                 yOffset += outerTileLen;
//             }
//             if (outerTailTileLen) {
//                 Duplicate(max, -1e30f, outerTileLen * elemPerBlk);
//                 Duplicate(sum, .0f, outerTileLen * elemPerBlk);
//                 UpdateOuter(outerTileLen);
//                 SetUpInner();
//                 for (int k = 0; k < innerTileNum; k++) {
//                     UpdateInner(innerTileLen);
//                     CopyIn();
//                     ComputeMax(innerTileLen * tileElemBase, outerTailTileLen);
//                 }
//                 if (innerTailTileLen) {
//                     UpdateInner(innerTailTileLen);
//                     CopyIn();
//                     ComputeMax(innerTailTileLen * tileElemBase, outerTailTileLen);
//                 }

//                 SetUpInner();
//                 for (int k = 0; k < innerTileNum; k++) {
//                     UpdateInner(innerTileLen);
//                     CopyIn();
//                     ComputeSum(innerTileLen * tileElemBase, outerTailTileLen);
//                 }
//                 if (innerTailTileLen) {
//                     UpdateInner(innerTailTileLen);
//                     CopyIn();
//                     ComputeSum(innerTailTileLen * tileElemBase, outerTailTileLen);
//                 }
//                 Compute(outerTailTileLen);
//                 CopyOut(yOffset, outerTailTileLen, 1);
//                 yOffset += outerTailTileLen;
//             }
//             UpdateSliceInfo();
//         }
//     }

// private:
//     __aicore__ inline void PrintArray(int64_t* arr, int64_t arrLen) {
//         for (int i = 0; i < arrLen; i++) {
//             printf("%ld, ", arr[i]);
//         }
//         printf("\n");
//     }

//     __aicore__ inline void CopyIn() {
//         LocalTensor<T> x = xQueue.AllocTensor<T>();
//         DataCopy(x, xGm, dstSlice, srcSlice, shapeLen);
//         xQueue.EnQue<T>(x);
//     }

//     __aicore__ inline void ComputeMax(int computeLen, int outerLen) {
//         LocalTensor<T> x = xQueue.DeQue<T>();
//         LocalTensor<float> work = workBuf.Get<float>();
//         LocalTensor<float> max = maxBuf.Get<float>();
//         uint16_t repeatTime = (computeLen + elemPerBlk - 1) / elemPerBlk;
//         uint64_t rsvdCnt = 0;
//         for (int i = 0; i < outerLen; i++) {
//             if constexpr (std::is_same_v<T, half>) {
//             }
//             else {
//                 LocalTensor<uint32_t> pattern = maskBuf.Get<uint32_t>();
//                 Duplicate<uint32_t>(pattern, mask32[i], 64);
//                 GatherMask(work, x, pattern, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
//                 ReduceMax(work, work, work, computeLen);
//             }
//             Max(max[i * elemPerBlk], max[i * elemPerBlk], work, 1);
//         }
//         xQueue.FreeTensor(x);
//     }

//     __aicore__ inline void ComputeSum(int computeLen, int outerLen) {
//         LocalTensor<T> x = xQueue.DeQue<T>();
//         LocalTensor<float> work = workBuf.Get<float>();
//         LocalTensor<float> sum = sumBuf.Get<float>();
//         LocalTensor<float> max = maxBuf.Get<float>();
//         uint16_t repeatTime = (computeLen + elemPerBlk - 1) / elemPerBlk;
//         uint64_t rsvdCnt = 0;
//         for (int i = 0; i < outerLen; i++) {
//             float maxVal = max.GetValue(i * elemPerBlk);
//             if constexpr (std::is_same_v<T, half>) {
//             }
//             else {
//                 LocalTensor<uint32_t> pattern = maskBuf.Get<uint32_t>();
//                 Duplicate<uint32_t>(pattern, mask32[i], 64);
//                 GatherMask(work, x, pattern, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
//                 Adds(work, work, -maxVal, computeLen);
//                 Exp(work, work, computeLen);
//                 ReduceSum(work, work, work, computeLen);
//             }
//             Add(sum[i * elemPerBlk], sum[i * elemPerBlk], work, 1);
//         }
//         xQueue.FreeTensor(x);
//     }

//     __aicore__ inline void Compute(int outerLen) {
//         LocalTensor<T> y = yQueue.AllocTensor<T>();
//         LocalTensor<float> max = maxBuf.Get<float>();
//         LocalTensor<float> sum = sumBuf.Get<float>();
//         Ln(sum, sum, outerLen * elemPerBlk);
//         Add(sum, sum, max, outerLen * elemPerBlk);
//         for (int i = 0; i < outerLen; i++) {
//             y.SetValue(i, T(sum.GetValue(i * elemPerBlk)));
//         }
//         yQueue.EnQue<T>(y);
//     }

//     __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
//         uint32_t blockLen = dataLen * sizeof(T);
//         LocalTensor<T> y = yQueue.DeQue<T>();
//         DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
//         DataCopyPad(yGm[offset], y, copyParams);
//         yQueue.FreeTensor(y);
//     }

//     __aicore__ inline void InitSliceInfo() {
//         for (int i = 0; i < dimLen; i++) {
//             int sliceOffset = dim[i];
//             srcSlice[sliceOffset].startIndex = 0;
//             srcSlice[sliceOffset].endIndex = shape[sliceOffset] - 1;
//             srcSlice[sliceOffset].stride = 0;
//             srcSlice[sliceOffset].burstLen = 1;
//             srcSlice[sliceOffset].shapeValue = shape[sliceOffset];
//             dstSlice[sliceOffset].startIndex = 0;
//             dstSlice[sliceOffset].endIndex = shape[sliceOffset] - 1;
//             dstSlice[sliceOffset].stride = 0;
//             dstSlice[sliceOffset].burstLen = 1;
//             dstSlice[sliceOffset].shapeValue = shape[sliceOffset];
//         }
//         for (int i = 0; i < (shapeLen - dimLen); i++) {
//             int sliceOffset = tileDim[i];
//             srcSlice[sliceOffset].startIndex = 0;
//             srcSlice[sliceOffset].endIndex = 0;
//             srcSlice[sliceOffset].stride = 0;
//             srcSlice[sliceOffset].burstLen = 1;
//             srcSlice[sliceOffset].shapeValue = shape[sliceOffset];
//             dstSlice[sliceOffset].startIndex = 0;
//             dstSlice[sliceOffset].endIndex = 0;
//             dstSlice[sliceOffset].stride = 0;
//             dstSlice[sliceOffset].burstLen = 1;
//             dstSlice[sliceOffset].shapeValue = 1;
//         }
//         dstSlice[0].burstLen = (outerTileLen + elemPerBlk - 1) / elemPerBlk;
//         dstSlice[0].shapeValue = dstSlice[0].burstLen * elemPerBlk;
//         dstSlice[0].endIndex = dstSlice[0].shapeValue - 1;
//         srcSlice[0].endIndex = dstSlice[0].endIndex;
//         srcSlice[0].burstLen = dstSlice[0].burstLen;
//         int64_t innerTileDim = dim[dimLen - 1];
//         dstSlice[innerTileDim].shapeValue = innerTileLen;
//         dstSlice[innerTileDim].endIndex = dstSlice[innerTileDim].shapeValue - 1;
//         srcSlice[innerTileDim].endIndex = dstSlice[innerTileDim].endIndex;        
//     }

//     __aicore__ inline void UpdateSliceInfo() {
//         uint32_t offset, nextOffset;
//         offset = tileDim[shapeLen - dimLen - 2];
//         srcSlice[offset].startIndex += 1;
//         srcSlice[offset].endIndex = srcSlice[offset].startIndex;
//         for (int i = shapeLen - dimLen - 2; i > 0; i--) {
//             offset = tileDim[i];
//             nextOffset = tileDim[i - 1];
//             srcSlice[nextOffset].startIndex += srcSlice[offset].startIndex / shape[offset];
//             srcSlice[nextOffset].endIndex = srcSlice[nextOffset].startIndex;
//             srcSlice[offset].startIndex = srcSlice[offset].startIndex % shape[offset];
//             srcSlice[offset].endIndex = srcSlice[offset].endIndex % shape[offset];
//         }
//     }

//     __aicore__ inline void PrintSliceInfo() {
//         for (int i = 0; i < shapeLen; i++) {
//             printf("[src %d] startIndex: %d, endIndex: %d\n", i, srcSlice[i].startIndex, srcSlice[i].endIndex);
//             printf("[dst %d] startIndex: %d, endIndex: %d\n", i, dstSlice[i].startIndex, dstSlice[i].endIndex);
//         }
//     }

//     __aicore__ inline void SetUpInner() {
//         int64_t innerTileDim = dim[dimLen - 1];
//         srcSlice[innerTileDim].startIndex = -innerTileLen;
//         srcSlice[innerTileDim].endIndex = -1;
//     }

//     __aicore__ inline void SetUpOuter() {
//         int64_t outerTileDim = 0;
//         srcSlice[outerTileDim].startIndex = -outerTileLen;
//         srcSlice[outerTileDim].endIndex = -1;
//     }

//     __aicore__ inline void UpdateInner(int length) {
//         int64_t innerTileDim = dim[dimLen - 1];
//         srcSlice[innerTileDim].startIndex = srcSlice[innerTileDim].endIndex + 1;
//         srcSlice[innerTileDim].endIndex += length;
//         dstSlice[innerTileDim].endIndex = length - 1;
//         dstSlice[innerTileDim].shapeValue = length;
//     }

//     __aicore__ inline void UpdateOuter(int length) {
//         int64_t outerTileDim = 0;
//         srcSlice[outerTileDim].startIndex = srcSlice[outerTileDim].endIndex + 1;
//         srcSlice[outerTileDim].endIndex += length;
//         srcSlice[outerTileDim].burstLen = (length + elemPerBlk - 1) / elemPerBlk;
//         dstSlice[outerTileDim].endIndex = length - 1;
//         dstSlice[outerTileDim].burstLen = srcSlice[outerTileDim].burstLen;
//         dstSlice[outerTileDim].shapeValue = length;
//     }

// private:
//     TPipe* pipe;
    
//     GlobalTensor<T> xGm, yGm;
//     TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
//     TBuf<QuePosition::VECCALC> xfp32Buf, workBuf, maskBuf;
//     TBuf<QuePosition::VECCALC> maxBuf, sumBuf;

//     int64_t tileLen, tileNum;
//     int64_t shape[8];
//     int64_t dim[8], tileDim[8];
//     int64_t shapeLen, dimLen;
//     int64_t dimSize;
//     SliceInfo srcSlice[8], dstSlice[8];

//     // outer 用于控制每次处理的 batch 数
//     int64_t outerTileLen, outerTileNum, outerTailTileLen;
//     // inner 用于控制每个 batch 每次搬运的数据数量
//     int64_t innerTileLen, innerTileNum, innerTailTileLen;
//     int64_t tileElemBase;

//     uint32_t mask32[8] = {
//         0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u,
//         0x10101010u, 0x20202020u, 0x40404040u, 0x80808080u
//     };
//     uint16_t mask16[16] = {
//         0x0001u, 0x0002u, 0x0004u, 0x0008u, 0x0010u, 0x0020u, 0x0040u, 0x0080u,
//         0x0100u, 0x0200u, 0x0400u, 0x0800u, 0x1000u, 0x2000u, 0x4000u, 0x8000u
//     };
// };

template <typename T>
class KernelLogSumExp2 {
constexpr static int elemPerBlk = BLOCK_SIZE / sizeof(T);
public:
    __aicore__ inline KernelLogSumExp2() {}
    __aicore__ inline void Init(
        TPipe* pipe,
        GM_ADDR x,
        GM_ADDR y,
        int64_t tileLen,
        int64_t tileNum,
        int64_t* shape,
        int64_t shapeLen,
        int64_t* dim,
        int64_t dimLen
    ) {
        // ASSERT(dim[dimLen - 1] != (shapeLen - 1) && "无法处理 dim 包含最后一轴的情况");
        this->pipe = pipe;
        this->outerTileLen = elemPerBlk;
        this->outerTileNum = shape[shapeLen - 1] / outerTileLen;
        this->outerTailTileLen = shape[shapeLen - 1] % outerTileLen;
        this->innerTileLen = 16;
        if (this->innerTileLen > shape[dim[dimLen - 1]]) {
            this->innerTileLen = shape[dim[dimLen - 1]];
        }
        this->innerTileNum = shape[dim[dimLen - 1]] / this->innerTileLen;
        this->innerTailTileLen = shape[dim[dimLen - 1]] % this->innerTileLen;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        // 计算 dim 去除最后一维后的数据长度
        this->tileElemBase = 1;
        for (int i = 0; i < dimLen - 1; i++) {
            tileElemBase *= shape[dim[i]];
        }
        pipe->InitBuffer(xQueue, BUFFER_NUM, (tileElemBase * innerTileLen * outerTileLen + elemPerBlk - 1) / elemPerBlk * BLOCK_SIZE);
        pipe->InitBuffer(yQueue, BUFFER_NUM, (outerTileLen + elemPerBlk - 1) / elemPerBlk * elemPerBlk);
        pipe->InitBuffer(workBuf, tileElemBase * innerTileLen * sizeof(float));
        pipe->InitBuffer(maskBuf, 8 * BLOCK_SIZE);
        pipe->InitBuffer(maxBuf, outerTileLen * BLOCK_SIZE);
        pipe->InitBuffer(sumBuf, outerTileLen * BLOCK_SIZE);
        this->tileLen = tileLen;
        this->tileNum = tileNum;
        this->shapeLen = shapeLen;
        this->dimLen = dimLen;

        // 对轴的定义进行改写，原先第一个轴称为第 0 轴，
        // 现在第一个轴称为第 shapeLen - 1 轴
        int dimOffset = 0;
        for (int i = 0; i < shapeLen; i++) {
            this->shape[i] = shape[shapeLen - 1 - i];
            if (dim[dimOffset] == i) {
                this->dim[dimOffset] = shapeLen - 1 - i;
                dimOffset += 1;
            }
            else {
                tileDim[i - dimOffset] = shapeLen - 1 - i;
            }
        }
        InitSliceInfo();
    }

    __aicore__ inline void Process() {
        // printf("[tileLen] %ld\n", tileLen);
        // printf("[tileNum] %ld\n", tileNum);
        // printf("[shapeLen] %ld\n", shapeLen);
        // printf("[shape] ");
        // PrintArray(shape, shapeLen);
        // printf("[dimLen] %ld\n", dimLen);
        // printf("[dim] ");
        // PrintArray(dim, dimLen);
        // printf("[tileDim] ");
        // printf("innerTileLen: %ld, innerTileNum: %ld, innerTailTileLen: %ld\n", innerTileLen, innerTileNum, innerTailTileLen);
        // printf("outerTileLen: %ld, outerTileNum: %ld, outerTailTileLen: %ld\n", outerTileLen, outerTileNum, outerTailTileLen);
        // PrintArray(tileDim, shapeLen - dimLen);
        // printf("tileElemBase: %ld\n", tileElemBase);
        LocalTensor<float> sum = sumBuf.Get<float>();
        int64_t loopCount = tileNum / shape[0];
        int64_t yOffset = 0;
        for (int i = 0; i < loopCount; i++) {
            SetUpOuter();
            for (int j = 0 ; j < outerTileNum; j++) {
                Duplicate(sum, .0f, outerTileLen * elemPerBlk);
                UpdateOuter(outerTileLen);
                // 求和
                SetUpInner();
                for (int k = 0; k < innerTileNum; k++) {
                    UpdateInner(innerTileLen);
                    CopyIn();
                    ComputeSum(innerTileLen * tileElemBase, outerTileLen);
                }
                if (innerTailTileLen) {
                    UpdateInner(innerTailTileLen);
                    CopyIn();
                    ComputeSum(innerTailTileLen * tileElemBase, outerTileLen);
                }
                // 计算最终结果
                Compute(outerTileLen);
                CopyOut(yOffset, outerTileLen, 1);
                yOffset += outerTileLen;
            }
            if (outerTailTileLen) {
                Duplicate(sum, .0f, outerTileLen * elemPerBlk);
                UpdateOuter(outerTileLen);
                SetUpInner();

                for (int k = 0; k < innerTileNum; k++) {
                    UpdateInner(innerTileLen);
                    CopyIn();
                    ComputeSum(innerTileLen * tileElemBase, outerTailTileLen);
                }
                if (innerTailTileLen) {
                    UpdateInner(innerTailTileLen);
                    CopyIn();
                    ComputeSum(innerTailTileLen * tileElemBase, outerTailTileLen);
                }
                Compute(outerTailTileLen);
                CopyOut(yOffset, outerTailTileLen, 1);
                yOffset += outerTailTileLen;
            }
            UpdateSliceInfo();
        }
    }

private:
    __aicore__ inline void PrintArray(int64_t* arr, int64_t arrLen) {
        for (int i = 0; i < arrLen; i++) {
            printf("%ld, ", arr[i]);
        }
        printf("\n");
    }

    __aicore__ inline void CopyIn() {
        LocalTensor<T> x = xQueue.AllocTensor<T>();
        DataCopy(x, xGm, dstSlice, srcSlice, shapeLen);
        xQueue.EnQue<T>(x);
    }

    __aicore__ inline void ComputeMax(int computeLen, int outerLen) {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> max = maxBuf.Get<float>();
        uint16_t repeatTime = (computeLen + elemPerBlk - 1) / elemPerBlk;
        uint64_t rsvdCnt = 0;
        for (int i = 0; i < outerLen; i++) {
            if constexpr (std::is_same_v<T, half>) {
            }
            else {
                LocalTensor<uint32_t> pattern = maskBuf.Get<uint32_t>();
                Duplicate<uint32_t>(pattern, mask32[i], 64);
                GatherMask(work, x, pattern, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
                ReduceMax(work, work, work, computeLen);
            }
            Max(max[i * elemPerBlk], max[i * elemPerBlk], work, 1);
        }
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeSum(int computeLen, int outerLen) {
        LocalTensor<T> x = xQueue.DeQue<T>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> sum = sumBuf.Get<float>();
        uint16_t repeatTime = (computeLen + elemPerBlk - 1) / elemPerBlk;
        uint64_t rsvdCnt = 0;
        if constexpr (std::is_same_v<T, half>) {
        }
        else {
            // int64_t totalSize = computeLen * outerLen;
            // int32_t repeatTime = (computeLen * outerTileLen + 63) / 64;
            // Exp(x, x, computeLen * outerTileLen);
            // for (int i = 0; i < outerLen; i++) {
            //     uint64_t mask[2] = {mask64[i], 0};
            //     // printf("mask: [%lu, %lu]\n", mask[0], mask[1]);
            //     ReduceSum<float>(work, x, work, mask, repeatTime, 8);
            //     printf("reduce sum: %f\n", work.GetValue(0));
            //     Add(sum[i * elemPerBlk], sum[i * elemPerBlk], work, 1);
            // }
            for (int i = 0; i < outerLen; i++) {
                LocalTensor<uint32_t> pattern = maskBuf.Get<uint32_t>();
                Duplicate<uint32_t>(pattern, mask32[i], 64);
                GatherMask(work, x, pattern, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
                Exp(work, work, computeLen);
                ReduceSum(work, work, work, computeLen);
                Add(sum[i * elemPerBlk], sum[i * elemPerBlk], work, 1);
            }
        }
        xQueue.FreeTensor(x);
    }

    __aicore__ inline void Compute(int outerLen) {
        LocalTensor<T> y = yQueue.AllocTensor<T>();
        LocalTensor<float> sum = sumBuf.Get<float>();
        Ln(sum, sum, outerLen * elemPerBlk);
        for (int i = 0; i < outerLen; i++) {
            y.SetValue(i, T(sum.GetValue(i * elemPerBlk)));
        }
        yQueue.EnQue<T>(y);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(T);
        LocalTensor<T> y = yQueue.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQueue.FreeTensor(y);
    }

    __aicore__ inline void InitSliceInfo() {
        for (int i = 0; i < dimLen; i++) {
            int sliceOffset = dim[i];
            srcSlice[sliceOffset].startIndex = 0;
            srcSlice[sliceOffset].endIndex = shape[sliceOffset] - 1;
            srcSlice[sliceOffset].stride = 0;
            srcSlice[sliceOffset].burstLen = 1;
            srcSlice[sliceOffset].shapeValue = shape[sliceOffset];
            dstSlice[sliceOffset].startIndex = 0;
            dstSlice[sliceOffset].endIndex = shape[sliceOffset] - 1;
            dstSlice[sliceOffset].stride = 0;
            dstSlice[sliceOffset].burstLen = 1;
            dstSlice[sliceOffset].shapeValue = shape[sliceOffset];
        }
        for (int i = 0; i < (shapeLen - dimLen); i++) {
            int sliceOffset = tileDim[i];
            srcSlice[sliceOffset].startIndex = 0;
            srcSlice[sliceOffset].endIndex = 0;
            srcSlice[sliceOffset].stride = 0;
            srcSlice[sliceOffset].burstLen = 1;
            srcSlice[sliceOffset].shapeValue = shape[sliceOffset];
            dstSlice[sliceOffset].startIndex = 0;
            dstSlice[sliceOffset].endIndex = 0;
            dstSlice[sliceOffset].stride = 0;
            dstSlice[sliceOffset].burstLen = 1;
            dstSlice[sliceOffset].shapeValue = 1;
        }
        dstSlice[0].burstLen = (outerTileLen + elemPerBlk - 1) / elemPerBlk;
        dstSlice[0].shapeValue = dstSlice[0].burstLen * elemPerBlk;
        dstSlice[0].endIndex = dstSlice[0].shapeValue - 1;
        srcSlice[0].endIndex = dstSlice[0].endIndex;
        srcSlice[0].burstLen = dstSlice[0].burstLen;
        int64_t innerTileDim = dim[dimLen - 1];
        dstSlice[innerTileDim].shapeValue = innerTileLen;
        dstSlice[innerTileDim].endIndex = dstSlice[innerTileDim].shapeValue - 1;
        srcSlice[innerTileDim].endIndex = dstSlice[innerTileDim].endIndex;        
    }

    __aicore__ inline void UpdateSliceInfo() {
        uint32_t offset, nextOffset;
        offset = tileDim[shapeLen - dimLen - 2];
        srcSlice[offset].startIndex += 1;
        srcSlice[offset].endIndex = srcSlice[offset].startIndex;
        for (int i = shapeLen - dimLen - 2; i > 0; i--) {
            offset = tileDim[i];
            nextOffset = tileDim[i - 1];
            srcSlice[nextOffset].startIndex += srcSlice[offset].startIndex / shape[offset];
            srcSlice[nextOffset].endIndex = srcSlice[nextOffset].startIndex;
            srcSlice[offset].startIndex = srcSlice[offset].startIndex % shape[offset];
            srcSlice[offset].endIndex = srcSlice[offset].endIndex % shape[offset];
        }
    }

    __aicore__ inline void PrintSliceInfo() {
        for (int i = 0; i < shapeLen; i++) {
            printf("[src %d] startIndex: %d, endIndex: %d\n", i, srcSlice[i].startIndex, srcSlice[i].endIndex);
            printf("[dst %d] startIndex: %d, endIndex: %d\n", i, dstSlice[i].startIndex, dstSlice[i].endIndex);
        }
    }

    __aicore__ inline void SetUpInner() {
        int64_t innerTileDim = dim[dimLen - 1];
        srcSlice[innerTileDim].startIndex = -innerTileLen;
        srcSlice[innerTileDim].endIndex = -1;
    }

    __aicore__ inline void SetUpOuter() {
        int64_t outerTileDim = 0;
        srcSlice[outerTileDim].startIndex = -outerTileLen;
        srcSlice[outerTileDim].endIndex = -1;
    }

    __aicore__ inline void UpdateInner(int length) {
        int64_t innerTileDim = dim[dimLen - 1];
        srcSlice[innerTileDim].startIndex = srcSlice[innerTileDim].endIndex + 1;
        srcSlice[innerTileDim].endIndex += length;
        dstSlice[innerTileDim].endIndex = length - 1;
        dstSlice[innerTileDim].shapeValue = length;
    }

    __aicore__ inline void UpdateOuter(int length) {
        int64_t outerTileDim = 0;
        srcSlice[outerTileDim].startIndex = srcSlice[outerTileDim].endIndex + 1;
        srcSlice[outerTileDim].endIndex += length;
        srcSlice[outerTileDim].burstLen = (length + elemPerBlk - 1) / elemPerBlk;
        dstSlice[outerTileDim].endIndex = length - 1;
        dstSlice[outerTileDim].burstLen = srcSlice[outerTileDim].burstLen;
        dstSlice[outerTileDim].shapeValue = length;
    }

private:
    TPipe* pipe;
    
    GlobalTensor<T> xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> xfp32Buf, workBuf, maskBuf;
    TBuf<QuePosition::VECCALC> maxBuf, sumBuf;

    int64_t tileLen, tileNum;
    int64_t shape[8];
    int64_t dim[8], tileDim[8];
    int64_t shapeLen, dimLen;
    int64_t dimSize;
    SliceInfo srcSlice[8], dstSlice[8];

    // outer 用于控制每次处理的 batch 数
    int64_t outerTileLen, outerTileNum, outerTailTileLen;
    // inner 用于控制每个 batch 每次搬运的数据数量
    int64_t innerTileLen, innerTileNum, innerTailTileLen;
    int64_t tileElemBase;

    uint32_t mask32[8] = {
        0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u,
        0x10101010u, 0x20202020u, 0x40404040u, 0x80808080u
    };
    uint16_t mask16[16] = {
        0x0001u, 0x0002u, 0x0004u, 0x0008u, 0x0010u, 0x0020u, 0x0040u, 0x0080u,
        0x0100u, 0x0200u, 0x0400u, 0x0800u, 0x1000u, 0x2000u, 0x4000u, 0x8000u
    };
    uint64_t mask64[8] {
        0x0101010101010101u, 0x0202020202020202u,
        0x0404040404040404u, 0x0808080808080808u,
        0x1010101010101010u, 0x2020202020202020u,
        0x4040404040404040u, 0x8080808080808080u
    };
};


extern "C" __global__ __aicore__ void log_sum_exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    if (TILING_KEY_IS(1)) {
        KernelLogSumExp<DTYPE_X> op;
        op.Init(x, y, t.tileLen, t.tileNum, t.shape, t.shapeLen, t.dim, t.strides, t.dimLen);
        op.Process();
    }
    else if(TILING_KEY_IS(2)) {
        KernelLogSumExp2<DTYPE_X> op;
        TPipe pipe;
        op.Init(&pipe, x, y, t.tileLen, t.tileNum, t.shape, t.shapeLen, t.dim, t.dimLen);
        op.Process();
    }
}