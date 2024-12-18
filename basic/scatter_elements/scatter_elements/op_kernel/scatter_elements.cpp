#include "kernel_operator.h"

using namespace AscendC;

constexpr int BUFFER_NUM = 1;

template <typename T>
class KernelScatterElements {
public:
    __aicore__ inline KernelScatterElements() {}
    __aicore__ inline void Init(
        GM_ADDR var,
        GM_ADDR indices,
        GM_ADDR updates,
        int64_t* varShape,
        int64_t* indicesShape,
        int64_t dimNum,
        int64_t mode,
        int64_t axis
    ) {
        varGm.SetGlobalBuffer((__gm__ T*)var);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
        updatesGm.SetGlobalBuffer((__gm__ T*)updates);

        for (int i = 0; i < dimNum; i++) {
            shapes[0][i] = varShape[i];
            shapes[1][i] = indicesShape[i];
        }
        this->loopCount = indicesShape[dimNum - 1];
        strides[0][dimNum - 1] = 1;
        strides[1][dimNum - 1] = 1;
        for (int i = dimNum - 2; i >= 0; i--) {
            strides[0][i] = strides[0][i + 1] * shapes[0][i + 1];
            strides[1][i] = strides[1][i + 1] * shapes[1][i + 1];
            loopCount *= shapes[1][i];
        }
        this->dimNum = dimNum;
        this->mode = mode;
        this->axis = axis;
    }

    __aicore__ inline void Process() {
        // printf("loopCount: %ld\n", loopCount);
        // printf("axis: %ld\n", axis);
        // printf("mode: %ld\n", mode);
        // printf("dimNum: %ld\n", dimNum);
        // printf("shape0: ");
        // PrintArray(shapes[0], dimNum);
        // printf("shape1: ");
        // PrintArray(shapes[1], dimNum);
        // printf("stride0: ");
        // PrintArray(strides[0], dimNum);
        // printf("stride1: ");
        // PrintArray(strides[1], dimNum);
        for (int i = 0; i < loopCount; i++) {
            int index = 0;
            int offset = 0;
            int32_t indice = indicesGm.GetValue(i);
            T update = updatesGm.GetValue(i);
            for (int j = 0; j < axis; j++) {
                index = i / strides[1][j] % shapes[1][j];
                offset += (index * strides[0][j]);
            }
            offset += indice * strides[0][axis];
            for (int j = axis + 1; j < dimNum; j++) {
                index = i / strides[1][j] % shapes[1][j];
                offset += (index * strides[0][j]);
            }
            T value = update;
            if (mode == 1) {
                value += varGm.GetValue(offset);
            }
            else if (mode == 2) {
                value *= varGm.GetValue(offset);
            }
            varGm.SetValue(offset, value);
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
    GlobalTensor<T> varGm, updatesGm;
    GlobalTensor<int32_t> indicesGm;

    // shapes 和 strides 依次存储 var 和 indices
    int64_t shapes[2][10], strides[2][10];
    int64_t dimNum, mode, axis;
    int64_t loopCount;
};

template <>
class KernelScatterElements<half> {
public:
    __aicore__ inline KernelScatterElements() {}
    __aicore__ inline void Init(
        GM_ADDR var,
        GM_ADDR indices,
        GM_ADDR updates,
        int64_t* varShape,
        int64_t* indicesShape,
        int64_t dimNum,
        int64_t mode,
        int64_t axis
    ) {
        varGm.SetGlobalBuffer((__gm__ half*)var);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
        updatesGm.SetGlobalBuffer((__gm__ half*)updates);

        for (int i = 0; i < dimNum; i++) {
            shapes[0][i] = varShape[i];
            shapes[1][i] = indicesShape[i];
        }
        this->loopCount = indicesShape[dimNum - 1];
        strides[0][dimNum - 1] = 1;
        strides[1][dimNum - 1] = 1;
        for (int i = dimNum - 2; i >= 0; i--) {
            strides[0][i] = strides[0][i + 1] * shapes[0][i + 1];
            strides[1][i] = strides[1][i + 1] * shapes[1][i + 1];
            loopCount *= shapes[1][i];
        }
        loopCount /= shapes[1][axis];

        pipe.InitBuffer(varQ, BUFFER_NUM, shapes[0][axis] * 32);
        pipe.InitBuffer(indicesQ, BUFFER_NUM, shapes[1][axis] * 32);
        pipe.InitBuffer(updatesQ, BUFFER_NUM, shapes[1][axis] * 32);
        pipe.InitBuffer(outQ, BUFFER_NUM, shapes[0][axis] * 32);
        pipe.InitBuffer(varsfp32Buf, shapes[0][axis] * 32 * 2);
        pipe.InitBuffer(updatesfp32Buf, shapes[1][axis] * 32 * 2);
        this->dimNum = dimNum;
        this->mode = mode;
        this->axis = axis;
    }

    __aicore__ inline void Process() {
        // printf("loopCount: %ld\n", loopCount);
        // printf("axis: %ld\n", axis);
        // printf("mode: %ld\n", mode);
        // printf("dimNum: %ld\n", dimNum);
        // printf("shape0: ");
        // PrintArray(shapes[0], dimNum);
        // printf("shape1: ");
        // PrintArray(shapes[1], dimNum);
        // printf("stride0: ");
        // PrintArray(strides[0], dimNum);
        // printf("stride1: ");
        // PrintArray(strides[1], dimNum);
        for (int i = 0; i < loopCount; i++) {
            int varOffset = 0;
            int indiceOffset = 0;
            for (int j = 0; j < axis; j++) {
                int index = (i * shapes[1][axis] / strides[1][j]) % shapes[1][j];
                varOffset += (index * strides[0][j]);
                indiceOffset += (index * strides[1][j]);
            }
            for (int j = axis + 1; j < dimNum; j++) {
                int index = (i / strides[1][j]) % shapes[1][j];
                varOffset += (index * strides[0][j]);
                indiceOffset += (index * strides[1][j]);
            }
            // printf("[varoffset] %d\n", varOffset);
            // printf("[indiceOffset] %d\n", indiceOffset);
            CopyVars(varOffset, 1, shapes[0][axis]);
            CopyIndices(indiceOffset, 1, shapes[1][axis]);
            CopyUpdates(indiceOffset, 1, shapes[1][axis]);
            Compute();
            CopyOut(varOffset, 1, shapes[1][axis]);
            // CopyVars(varOffset, 1, 1);
            // CopyIndices(indiceOffset, 1, 1);
            // CopyUpdates(indiceOffset, 1, 1);
            // Compute();
            // CopyOut(indiceOffset, 1, 1);
        }
    }

private:
    __aicore__ inline void PrintArray(int64_t* arr, int64_t len) {
        for (int i = 0; i < len; i++) {
            printf("%ld, ", arr[i]);
        }
        printf("\n");
    }

    __aicore__ inline void CopyVars(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(half);
        uint32_t srcStride = (strides[0][axis] - 1) * sizeof(half);
        LocalTensor<half> vars = varQ.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, dataSize, srcStride, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(vars, varGm[offset], copyParams, padParams);
        varQ.EnQue<half>(vars);
    }
    __aicore__ inline void CopyUpdates(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(half);
        uint32_t srcStride = (strides[1][axis] - 1) * sizeof(half);
        LocalTensor<half> updates = updatesQ.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, dataSize, srcStride, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(updates, updatesGm[offset], copyParams, padParams);
        updatesQ.EnQue<half>(updates);
    }
    __aicore__ inline void CopyIndices(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(int32_t);
        uint32_t srcStride = (strides[1][axis] - 1) * sizeof(int32_t);
        LocalTensor<int32_t> indices = indicesQ.AllocTensor<int32_t>();
        DataCopyExtParams copyParams {repeatTime, dataSize, srcStride, 0, 0};
        DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
        DataCopyPad(indices, indicesGm[offset], copyParams, padParams);
        indicesQ.EnQue<int32_t>(indices);
    }

    __aicore__ inline void Compute() {
        LocalTensor<half> vars = varQ.DeQue<half>();
        LocalTensor<half> updates = updatesQ.DeQue<half>();
        LocalTensor<int32_t> indices = indicesQ.DeQue<int32_t>();
        LocalTensor<half> out = outQ.AllocTensor<half>();
        LocalTensor<float> varsfp32 = varsfp32Buf.Get<float>();
        LocalTensor<float> updatesfp32 = updatesfp32Buf.Get<float>();
        Cast(varsfp32, vars, RoundMode::CAST_NONE, shapes[0][axis] * 16);
        Cast(updatesfp32, updates, RoundMode::CAST_NONE, shapes[1][axis] * 16);
        // printf("[shape 1] ");
        // PrintArray(shapes[1], dimNum);
        // printf("here: %ld\n", shapes[1][axis]);
        // for (int i = 0; i < shapes[1][axis]; i++) {
        //     printf("indice: %d, updated: %f\n", indices.GetValue(i * 8), (float)updates.GetValue(i * 16));
        // }
        for (int i = 0; i < shapes[1][axis]; i++) {
            int32_t indice = indices.GetValue(i * 8);
            float updated = updatesfp32.GetValue(i * 16);
            float var = varsfp32.GetValue(indice * 16);
            // printf("indice: %d, updated: %f, var: %f\n", indice, updated, var);
            if (mode == 1) {
                updated += var;
            }
            else if(mode == 2) {
                updated *= var;
            }
            varsfp32.SetValue(indice * 16, updated);
        }
        Cast(out, varsfp32, RoundMode::CAST_NONE, shapes[0][axis] * 16);
        outQ.EnQue<half>(out);
        varQ.FreeTensor(vars);
        updatesQ.FreeTensor(updates);
        indicesQ.FreeTensor(indices);
    }
    
    __aicore__ inline void CopyOut(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(half);
        uint32_t dstStride = (strides[0][axis] - 1) * sizeof(half);
        LocalTensor<half> out = outQ.DeQue<half>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, dstStride, 0};
        DataCopyPad(varGm[offset], out, copyParams);
        outQ.FreeTensor(out);
    }

private:
    GlobalTensor<half> varGm, updatesGm;
    GlobalTensor<int32_t> indicesGm;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> varQ, indicesQ, updatesQ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQ;
    TBuf<QuePosition::VECCALC> varsfp32Buf, updatesfp32Buf;

    // shapes 和 strides 依次存储 var 和 indices
    int64_t shapes[2][10], strides[2][10];
    int64_t dimNum, mode, axis;
    int64_t loopCount;
};

// template <typename T>
// class KernelScatterElements {
// public:
//     __aicore__ inline KernelScatterElements() {}
//     __aicore__ inline void Init(
//         GM_ADDR var,
//         GM_ADDR indices,
//         GM_ADDR updates,
//         int64_t tileLength,
//         int64_t tileNum,
//         int64_t tailTileLength,
//         int64_t varLen,
//         int64_t indicesLen,
//         int64_t varStride,
//         int64_t indiceStride,
//         int64_t mode,
//         int64_t varTileLen,
//         int64_t indiceTileLen
//     ) {
//         varGm.SetGlobalBuffer((__gm__ T*)var);
//         indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
//         updatesGm.SetGlobalBuffer((__gm__ T*)updates);

//         this->tileLength = tileLength;
//         this->tileNum = tileNum;
//         this->tailTileLength = tailTileLength;
//         this->varLen = varLen;
//         this->indicesLen = indicesLen;
//         this->varStride = varStride;
//         this->indiceStride = indiceStride;
//         this->varTileLen = varTileLen;
//         this->indiceTileLen = indiceTileLen;
//         this->mode = mode;
//     }

//     // [i][j][k]
//     // 1, 2, 3, ... k
//     // j * k + 1, j * 
    
//     // i * k
//     // [i][j][k][l]
//     // dim = 1, ... k * l = indiceStride
//     // j * k * l
//     // num = i * j * l
//     __aicore__ inline void Process() {
//         // for (int i = 0; i < indicesLen)

//         // for (int i = 0; i < tileNum; i++) {
//         //     int32_t indiceStart = i / indiceStride * indiceStride * indiceTileLen + (i % indiceStride);
//         //     int32_t varStart = i / indiceStride * varStride * varTileLen + (i % indiceStride);
//         //     // int32_t indiceStart = i;
//         //     // int32_t varStart = i;
//         //     if (indiceStride == 1) {
//         //         indiceStart = i * indiceTileLen;
//         //         varStart = i * varTileLen;
//         //     }
//         //     printf("[indiceStart] %d\n", indiceStart);
//         //     printf("[varStart] %d\n", varStart);
//         //     printf("");
//         //     for (int ii = 0; ii < indiceTileLen; ii++) {
//         //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
//         //         int32_t indiceOffset = indiceStart + (ii * indiceStride);
//         //         int32_t indice = indicesGm.GetValue(indiceOffset);
//         //         T update = updatesGm.GetValue(indiceOffset);
//         //         int32_t varOffset = varStart + (indice * varStride);
//         //         T var = varGm.GetValue(varOffset);
//         //         T res = update;
//         //         if (mode == 1)
//         //             res = update + var;
//         //         if (mode == 2)
//         //             res = update * var;
//         //         printf("[indiceOffset] %d, [varOffset] %d\n", indiceOffset, varOffset);
//         //         printf("[var] %d, [indice] %d, [update] %d, [res] %d\n", var, indice, update, res);
//         //         varGm.SetValue(varOffset, res);
//         //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
//         //     }
//         // }


//         // for (int i = 0; i < indicesLen; i++) {
//         //     DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
//         //     int32_t indice = indicesGm.GetValue(i);
//         //     int32_t offset = (indice * varStride) + (i % indiceStride);
//         //     // int32_t offset = (i / updateStride) * varStride + (indices % updateStride);
//         //     T update = updatesGm.GetValue(i);
//         //     T var = varGm.GetValue(offset);
//         //     if constexpr (std::is_same_v<T, half>) {
//         //         float res = static_cast<float>(update);
//         //         if (mode == 1) {
//         //             res = res + static_cast<float>(var);
//         //         }
//         //         else if (mode == 2) {
//         //             res = res * static_cast<float>(var);
//         //         }
//         //         varGm.SetValue(offset, static_cast<half>(res));
//         //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
//         //     }
//         //     else {
//         //         T res = update;
//         //         if (mode == 1) {
//         //             res = update + var;
//         //         }
//         //         else if (mode == 2) {
//         //             res = update * var;
//         //         }
//         //         // printf("[offset]: %d\n", offset);
//         //         // printf("[var] %d, [update] %d, [res]: %d\n", var, update, res);
//         //         varGm.SetValue(offset, res);
//         //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
//         //     }
//         // }
//     }
// private:
//     GlobalTensor<T> varGm, updatesGm;
//     GlobalTensor<int32_t> indicesGm;

//     int64_t tileLength;
//     int64_t tileNum;
//     int64_t tailTileLength;
//     int64_t varLen;
//     int64_t indicesLen;
//     int64_t varStride;
//     int64_t indiceStride;
//     int64_t mode;
//     int64_t varTileLen;
//     int64_t indiceTileLen;
// };


// template <>
// class KernelScatterElements<half> {
// public:
//     __aicore__ inline KernelScatterElements() {}
//     __aicore__ inline void Init(
//         GM_ADDR var,
//         GM_ADDR indices,
//         GM_ADDR updates,
//         int64_t tileLength,
//         int64_t tileNum,
//         int64_t tailTileLength,
//         int64_t varLen,
//         int64_t indicesLen,
//         int64_t varStride,
//         int64_t indiceStride,
//         int64_t mode,
//         int64_t varTileLen,
//         int64_t indiceTileLen
//     ) {
//         varsGm.SetGlobalBuffer((__gm__ half*)var);
//         indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
//         updatesGm.SetGlobalBuffer((__gm__ half*)updates);

//         int64_t varTileLenAlign32 = (varTileLen + 15) / 16 * 16;
//         int64_t indiceTileLenAlign32 = (indiceTileLen + 7) / 8 * 8;
//         int64_t updateTileLenAlign32 = (indiceTileLen + 15) / 16 * 16;
//         pipe.InitBuffer(varsQ, BUFFER_NUM, varTileLenAlign32 * 16 * sizeof(half));
//         pipe.InitBuffer(indicesQ, BUFFER_NUM, indiceTileLenAlign32 * 8 * sizeof(int32_t));
//         pipe.InitBuffer(updatesQ, BUFFER_NUM, updateTileLenAlign32 * 16 * sizeof(half));
//         pipe.InitBuffer(outQ, BUFFER_NUM, varTileLenAlign32 * 16 * sizeof(half));
//         pipe.InitBuffer(varsfp32Buf, varTileLenAlign32 * sizeof(float));
//         pipe.InitBuffer(updatesfp32Buf, updateTileLenAlign32 * sizeof(float));
//         pipe.InitBuffer(mask1Buf, 256);
//         pipe.InitBuffer(mask2Buf, 256);

//         this->tileLength = tileLength;
//         this->tileNum = tileNum;
//         this->tailTileLength = tailTileLength;
//         this->varLen = varLen;
//         this->indicesLen = indicesLen;
//         this->varStride = varStride;
//         this->indiceStride = indiceStride;
//         this->mode = mode;
//         // this->tileLengthAlign32 = tileLengthAlign32;
//         this->varTileLen = varTileLen;
//         this->varTileLenAlign32 = varTileLenAlign32;
//         this->indiceTileLen = indiceTileLen;
//         this->indiceTileLenAlign32 = indiceTileLenAlign32;
//         this->updateTileLen = indiceTileLen;
//         this->updateTileLenAlign32 = updateTileLenAlign32;
//     }
//     __aicore__ inline void Process() {
//         if (indiceStride == 1) {
//             // TODO: 针对连续读取进行优化
//             for (int i = 0; i < tileNum; i++) {
//                 CopyVars(i * varTileLen, 1, varTileLen);
//                 CopyUpdates(i * updateTileLen, 1, updateTileLen);
//                 CopyIndices(i * indiceTileLen, 1, indiceTileLen);
//                 Compute();
//                 CopyOut(i * varTileLen, 1, varTileLen);
//             }
//         }
//         else {
//             for (int i = 0; i < tileNum; i++) {
//                 CopyVars(i, 1, varTileLen);
//                 CopyUpdates(i, 1, updateTileLen);
//                 CopyIndices(i, 1, indiceTileLen);
//                 Compute();
//                 CopyOut(i, 1, varTileLen);
//             }
//         }
//         // if (tailTileLength) {
//         //     // TODO
//         // }
//     }

// private:
//     __aicore__ inline void CopyVars(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
//         uint32_t blockLen = dataLen * sizeof(half);
//         uint32_t srcStride = (varStride - 1) * sizeof(half);
//         LocalTensor<half> vars = varsQ.AllocTensor<half>();
//         DataCopyExtParams copyParams {repeatTime, blockLen, srcStride, 0, 0};
//         DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
//         DataCopyPad(vars, varsGm[offset], copyParams, padParams);
//         varsQ.EnQue<half>(vars);
//     }
//     __aicore__ inline void CopyUpdates(
//         uint64_t offset,
//         uint32_t dataLen,
//         uint16_t repeatTime
//     ) {
//         uint32_t blockLen = dataLen * sizeof(half);
//         uint32_t srcStride = (indiceStride - 1) * sizeof(half);
//         LocalTensor<half> updates = updatesQ.AllocTensor<half>();
//         DataCopyExtParams copyParams {repeatTime, blockLen, srcStride, 0, 0};
//         DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
//         DataCopyPad(updates, updatesGm[offset], copyParams, padParams);
//         updatesQ.EnQue<half>(updates);
//     }
//     __aicore__ inline void CopyIndices(
//         uint64_t offset,
//         uint32_t dataLen,
//         uint16_t repeatTime
//     ) {
//         uint32_t blockLen = dataLen * sizeof(int32_t);
//         uint32_t srcStride = (indiceStride - 1) * sizeof(int32_t);
//         // printf("blockLen: %d, srcStride, %d, second indice: %d\n", blockLen, srcStride, indicesGm.GetValue)
//         LocalTensor<int32_t> indices = indicesQ.AllocTensor<int32_t>();
//         DataCopyExtParams copyParams {repeatTime, blockLen, srcStride, 0, 0};
//         DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
//         DataCopyPad(indices, indicesGm[offset], copyParams, padParams);
//         indicesQ.EnQue<int32_t>(indices);
//     }

//     __aicore__ inline void Compute() {
//         LocalTensor<half> vars = varsQ.DeQue<half>();
//         LocalTensor<half> updates = updatesQ.DeQue<half>();
//         LocalTensor<int32_t> indices = indicesQ.DeQue<int32_t>();
//         LocalTensor<half> out = outQ.AllocTensor<half>();
//         LocalTensor<float> varsfp32 = varsfp32Buf.Get<float>();
//         LocalTensor<float> updatesfp32 = updatesfp32Buf.Get<float>();
//         LocalTensor<uint16_t> mask1 = mask1Buf.Get<uint16_t>();
//         LocalTensor<uint32_t> mask2 = mask2Buf.Get<uint32_t>();
//         // printf("[update copyed] ");
//         // for (int i = 0; i < updateTileLen; i++) {
//         //     printf("%f ,", updates.GetValue(i * 16));
//         // }
//         uint16_t repeatTime = varTileLenAlign32 / 8;
//         uint64_t rsvdCnt = 0;
//         Duplicate<uint16_t>(mask1, 0x8000u, 128);
//         GatherMask(vars, vars, mask1, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
//         repeatTime = updateTileLenAlign32 / 8;
//         GatherMask(updates, updates, mask1, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
//         Duplicate<uint16_t>(mask1, 0x8080u, 128);
//         GatherMask(indices, indices, mask1.ReinterpretCast<uint32_t>(), false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
//         Cast(varsfp32, vars, RoundMode::CAST_NONE, varTileLenAlign32);
//         Cast(updatesfp32, updates, RoundMode::CAST_NONE, updateTileLen);
//         // PipeBarrier<PIPE_V>();
//         // printf("[var] ");
//         // for (int i = 0; i < varTileLen; i++) {
//         //     printf("%f ,", vars.GetValue(i));
//         // }
//         printf("");
//         // printf("[indice] ");
//         // for (int i = 0; i < indiceTileLen; i++) {
//         //     printf("%d ,", indices.GetValue(i));
//         // }
//         // printf("\n");
//         // printf("[update] ");
//         // for (int i = 0; i < updateTileLen; i++) {
//         //     printf("%f ,", updates.GetValue(i));
//         // }
//         // printf("\n");
//         if (mode == 0) {
//             for (int i = 0; i < indiceTileLen; i++) {
//                 int32_t indice = indices.GetValue(i);
//                 float var = varsfp32.GetValue(indice);
//                 float update = updatesfp32.GetValue(i);
//                 varsfp32.SetValue(indice, update);
//             }
//         }
//         else if (mode == 1) {
//             for (int i = 0; i < indiceTileLen; i++) {
//                 int32_t indice = indices.GetValue(i);
//                 float var = varsfp32.GetValue(indice);
//                 float update = updatesfp32.GetValue(i);
//                 // printf("[%d] indice: %d, var: %f, update: %f\n", i, indice, var, update);
//                 var = var + update;
//                 varsfp32.SetValue(indice, var);
//             }
//         }
//         else {
//             for (int i = 0; i < indiceTileLen; i++) {
//                 int32_t indice = indices.GetValue(i);
//                 float var = varsfp32.GetValue(indice);
//                 float update = updatesfp32.GetValue(i);
//                 var = var * update;
//                 varsfp32.SetValue(indice, var);
//             }
//         }
//         Cast(vars, varsfp32, RoundMode::CAST_RINT, varTileLenAlign32);
//         // printf("[var] ");
//         // for (int i = 0; i < varTileLen; i++) {
//         //     printf("%f ,", vars.GetValue(i));
//         // }
//         // printf("\n");
//         // TODO: Brcb 一次只能处理 255 * 8 个元素
//         Brcb(out, vars, (uint8_t)varTileLenAlign32 / 8, {1, 8});
//         // printf("[out] ");
//         // for (int i = 0; i < varTileLen; i++) {
//         //     printf("%f ,", out.GetValue(i * 16));
//         // }
//         // printf("\n");
//         outQ.EnQue<half>(out);
//         varsQ.FreeTensor(vars);
//         updatesQ.FreeTensor(updates);
//         indicesQ.FreeTensor(indices);
//     }
    
//     __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
//         uint32_t blockLen = dataLen * sizeof(half);
//         uint32_t dstStride = (varStride - 1) * sizeof(half);
//         LocalTensor<half> out = outQ.DeQue<half>();
//         DataCopyExtParams copyParams {repeatTime, blockLen, 0, dstStride, 0};
//         DataCopyPad(varsGm[offset], out, copyParams);
//         outQ.FreeTensor(out);
//     }
// private:
//     TPipe pipe;
//     GlobalTensor<half> varsGm, updatesGm;
//     GlobalTensor<int32_t> indicesGm;

//     TQue<QuePosition::VECIN, BUFFER_NUM> varsQ, indicesQ, updatesQ;
//     TQue<QuePosition::VECIN, BUFFER_NUM> outQ;
//     TBuf<QuePosition::VECCALC> varsfp32Buf, updatesfp32Buf;
//     TBuf<QuePosition::VECCALC> mask1Buf, mask2Buf;

//     int64_t tileLength;
//     int64_t tileNum;
//     int64_t tailTileLength;
//     int64_t tileLengthAlign32;
//     int64_t varLen;
//     int64_t indicesLen;
//     int64_t varStride;
//     int64_t indiceStride;
//     int64_t mode;
//     int64_t varTileLen;
//     int64_t varTileLenAlign32;
//     int64_t indiceTileLen;
//     int64_t indiceTileLenAlign32;
//     int64_t updateTileLen;
//     int64_t updateTileLenAlign32;
// };

// template <>
// class KernelScatterElements<float> {
// public:
//     __aicore__ inline KernelScatterElements() {}
//     __aicore__ inline void Init(
//         GM_ADDR var,
//         GM_ADDR indices,
//         GM_ADDR updates,
//         int64_t tileLength,
//         int64_t tileNum,
//         int64_t tailTileLength,
//         int64_t indicesLen,
//         int64_t varStride,
//         int64_t updateStride,
//         int64_t mode,
//         int64_t varTileLen,
//         int64_t indiceTileLen,
//         int64_t updateTileLen
//     ) {}
//     __aicore__ inline void Process() {
//         for(;;);
//     }

// private:
// };


extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelScatterElements<DTYPE_VAR> op;
    op.Init(var, indices, updates, t.varShape, t.indicesShape, t.dimNum, t.mode, t.axis);
    op.Process();
}