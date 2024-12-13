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
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength,
        int64_t varLen,
        int64_t indicesLen,
        int64_t varStride,
        int64_t indiceStride,
        int64_t mode,
        int64_t varTileLen,
        int64_t indiceTileLen
    ) {
        varGm.SetGlobalBuffer((__gm__ T*)var);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
        updatesGm.SetGlobalBuffer((__gm__ T*)updates);

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
        this->varLen = varLen;
        this->indicesLen = indicesLen;
        this->varStride = varStride;
        this->indiceStride = indiceStride;
        this->varTileLen = varTileLen;
        this->indiceTileLen = indiceTileLen;
        this->mode = mode;
    }

    // [i][j][k]
    // 1, 2, 3, ... k
    // j * k + 1, j * 
    
    // i * k
    // [i][j][k][l]
    // dim = 1, ... k * l = indiceStride
    // j * k * l
    // num = i * j * l
    __aicore__ inline void Process() {
        // for (int i = 0; i < indicesLen)

        // for (int i = 0; i < tileNum; i++) {
        //     int32_t indiceStart = i / indiceStride * indiceStride * indiceTileLen + (i % indiceStride);
        //     int32_t varStart = i / indiceStride * varStride * varTileLen + (i % indiceStride);
        //     // int32_t indiceStart = i;
        //     // int32_t varStart = i;
        //     if (indiceStride == 1) {
        //         indiceStart = i * indiceTileLen;
        //         varStart = i * varTileLen;
        //     }
        //     printf("[indiceStart] %d\n", indiceStart);
        //     printf("[varStart] %d\n", varStart);
        //     printf("");
        //     for (int ii = 0; ii < indiceTileLen; ii++) {
        //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
        //         int32_t indiceOffset = indiceStart + (ii * indiceStride);
        //         int32_t indice = indicesGm.GetValue(indiceOffset);
        //         T update = updatesGm.GetValue(indiceOffset);
        //         int32_t varOffset = varStart + (indice * varStride);
        //         T var = varGm.GetValue(varOffset);
        //         T res = update;
        //         if (mode == 1)
        //             res = update + var;
        //         if (mode == 2)
        //             res = update * var;
        //         printf("[indiceOffset] %d, [varOffset] %d\n", indiceOffset, varOffset);
        //         printf("[var] %d, [indice] %d, [update] %d, [res] %d\n", var, indice, update, res);
        //         varGm.SetValue(varOffset, res);
        //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
        //     }
        // }


        // for (int i = 0; i < indicesLen; i++) {
        //     DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
        //     int32_t indice = indicesGm.GetValue(i);
        //     int32_t offset = (indice * varStride) + (i % indiceStride);
        //     // int32_t offset = (i / updateStride) * varStride + (indices % updateStride);
        //     T update = updatesGm.GetValue(i);
        //     T var = varGm.GetValue(offset);
        //     if constexpr (std::is_same_v<T, half>) {
        //         float res = static_cast<float>(update);
        //         if (mode == 1) {
        //             res = res + static_cast<float>(var);
        //         }
        //         else if (mode == 2) {
        //             res = res * static_cast<float>(var);
        //         }
        //         varGm.SetValue(offset, static_cast<half>(res));
        //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
        //     }
        //     else {
        //         T res = update;
        //         if (mode == 1) {
        //             res = update + var;
        //         }
        //         else if (mode == 2) {
        //             res = update * var;
        //         }
        //         // printf("[offset]: %d\n", offset);
        //         // printf("[var] %d, [update] %d, [res]: %d\n", var, update, res);
        //         varGm.SetValue(offset, res);
        //         DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(varGm);
        //     }
        // }
    }
private:
    GlobalTensor<T> varGm, updatesGm;
    GlobalTensor<int32_t> indicesGm;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailTileLength;
    int64_t varLen;
    int64_t indicesLen;
    int64_t varStride;
    int64_t indiceStride;
    int64_t mode;
    int64_t varTileLen;
    int64_t indiceTileLen;
};


template <>
class KernelScatterElements<half> {
public:
    __aicore__ inline KernelScatterElements() {}
    __aicore__ inline void Init(
        GM_ADDR var,
        GM_ADDR indices,
        GM_ADDR updates,
        int64_t tileLength,
        int64_t tileNum,
        int64_t tailTileLength,
        int64_t varLen,
        int64_t indicesLen,
        int64_t varStride,
        int64_t indiceStride,
        int64_t mode,
        int64_t varTileLen,
        int64_t indiceTileLen
    ) {
        varsGm.SetGlobalBuffer((__gm__ half*)var);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
        updatesGm.SetGlobalBuffer((__gm__ half*)updates);

        int64_t varTileLenAlign32 = (varTileLen + 15) / 16 * 16;
        int64_t indiceTileLenAlign32 = (indiceTileLen + 7) / 8 * 8;
        int64_t updateTileLenAlign32 = (indiceTileLen + 15) / 16 * 16;
        pipe.InitBuffer(varsQ, BUFFER_NUM, varTileLenAlign32 * 16 * sizeof(half));
        pipe.InitBuffer(indicesQ, BUFFER_NUM, indiceTileLenAlign32 * 8 * sizeof(int32_t));
        pipe.InitBuffer(updatesQ, BUFFER_NUM, updateTileLenAlign32 * 16 * sizeof(half));
        pipe.InitBuffer(outQ, BUFFER_NUM, varTileLenAlign32 * 16 * sizeof(half));
        pipe.InitBuffer(varsfp32Buf, varTileLenAlign32 * sizeof(float));
        pipe.InitBuffer(updatesfp32Buf, updateTileLenAlign32 * sizeof(float));
        pipe.InitBuffer(mask1Buf, 256);
        pipe.InitBuffer(mask2Buf, 256);

        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailTileLength = tailTileLength;
        this->varLen = varLen;
        this->indicesLen = indicesLen;
        this->varStride = varStride;
        this->indiceStride = indiceStride;
        this->mode = mode;
        // this->tileLengthAlign32 = tileLengthAlign32;
        this->varTileLen = varTileLen;
        this->varTileLenAlign32 = varTileLenAlign32;
        this->indiceTileLen = indiceTileLen;
        this->indiceTileLenAlign32 = indiceTileLenAlign32;
        this->updateTileLen = indiceTileLen;
        this->updateTileLenAlign32 = updateTileLenAlign32;
    }
    __aicore__ inline void Process() {
        if (indiceStride == 1) {
            // TODO: 针对连续读取进行优化
            for (int i = 0; i < tileNum; i++) {
                CopyVars(i * varTileLen, 1, varTileLen);
                CopyUpdates(i * updateTileLen, 1, updateTileLen);
                CopyIndices(i * indiceTileLen, 1, indiceTileLen);
                Compute();
                CopyOut(i * varTileLen, 1, varTileLen);
            }
        }
        else {
            for (int i = 0; i < tileNum; i++) {
                CopyVars(i, 1, varTileLen);
                CopyUpdates(i, 1, updateTileLen);
                CopyIndices(i, 1, indiceTileLen);
                Compute();
                CopyOut(i, 1, varTileLen);
            }
        }
        // if (tailTileLength) {
        //     // TODO
        // }
    }

private:
    __aicore__ inline void CopyVars(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(half);
        uint32_t srcStride = (varStride - 1) * sizeof(half);
        LocalTensor<half> vars = varsQ.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, blockLen, srcStride, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(vars, varsGm[offset], copyParams, padParams);
        varsQ.EnQue<half>(vars);
    }
    __aicore__ inline void CopyUpdates(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = dataLen * sizeof(half);
        uint32_t srcStride = (indiceStride - 1) * sizeof(half);
        LocalTensor<half> updates = updatesQ.AllocTensor<half>();
        DataCopyExtParams copyParams {repeatTime, blockLen, srcStride, 0, 0};
        DataCopyPadExtParams<half> padParams {false, 0, 0, 0};
        DataCopyPad(updates, updatesGm[offset], copyParams, padParams);
        updatesQ.EnQue<half>(updates);
    }
    __aicore__ inline void CopyIndices(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t blockLen = dataLen * sizeof(int32_t);
        uint32_t srcStride = (indiceStride - 1) * sizeof(int32_t);
        // printf("blockLen: %d, srcStride, %d, second indice: %d\n", blockLen, srcStride, indicesGm.GetValue)
        LocalTensor<int32_t> indices = indicesQ.AllocTensor<int32_t>();
        DataCopyExtParams copyParams {repeatTime, blockLen, srcStride, 0, 0};
        DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
        DataCopyPad(indices, indicesGm[offset], copyParams, padParams);
        indicesQ.EnQue<int32_t>(indices);
    }

    __aicore__ inline void Compute() {
        LocalTensor<half> vars = varsQ.DeQue<half>();
        LocalTensor<half> updates = updatesQ.DeQue<half>();
        LocalTensor<int32_t> indices = indicesQ.DeQue<int32_t>();
        LocalTensor<half> out = outQ.AllocTensor<half>();
        LocalTensor<float> varsfp32 = varsfp32Buf.Get<float>();
        LocalTensor<float> updatesfp32 = updatesfp32Buf.Get<float>();
        LocalTensor<uint16_t> mask1 = mask1Buf.Get<uint16_t>();
        LocalTensor<uint32_t> mask2 = mask2Buf.Get<uint32_t>();
        // printf("[update copyed] ");
        // for (int i = 0; i < updateTileLen; i++) {
        //     printf("%f ,", updates.GetValue(i * 16));
        // }
        uint16_t repeatTime = varTileLenAlign32 / 8;
        uint64_t rsvdCnt = 0;
        Duplicate<uint16_t>(mask1, 0x8000u, 128);
        GatherMask(vars, vars, mask1, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
        repeatTime = updateTileLenAlign32 / 8;
        GatherMask(updates, updates, mask1, false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
        Duplicate<uint16_t>(mask1, 0x8080u, 128);
        GatherMask(indices, indices, mask1.ReinterpretCast<uint32_t>(), false, 0, {1, repeatTime, 8, 0}, rsvdCnt);
        Cast(varsfp32, vars, RoundMode::CAST_NONE, varTileLenAlign32);
        Cast(updatesfp32, updates, RoundMode::CAST_NONE, updateTileLen);
        // PipeBarrier<PIPE_V>();
        // printf("[var] ");
        // for (int i = 0; i < varTileLen; i++) {
        //     printf("%f ,", vars.GetValue(i));
        // }
        printf("");
        // printf("[indice] ");
        // for (int i = 0; i < indiceTileLen; i++) {
        //     printf("%d ,", indices.GetValue(i));
        // }
        // printf("\n");
        // printf("[update] ");
        // for (int i = 0; i < updateTileLen; i++) {
        //     printf("%f ,", updates.GetValue(i));
        // }
        // printf("\n");
        if (mode == 0) {
            for (int i = 0; i < indiceTileLen; i++) {
                int32_t indice = indices.GetValue(i);
                float var = varsfp32.GetValue(indice);
                float update = updatesfp32.GetValue(i);
                varsfp32.SetValue(indice, update);
            }
        }
        else if (mode == 1) {
            for (int i = 0; i < indiceTileLen; i++) {
                int32_t indice = indices.GetValue(i);
                float var = varsfp32.GetValue(indice);
                float update = updatesfp32.GetValue(i);
                // printf("[%d] indice: %d, var: %f, update: %f\n", i, indice, var, update);
                var = var + update;
                varsfp32.SetValue(indice, var);
            }
        }
        else {
            for (int i = 0; i < indiceTileLen; i++) {
                int32_t indice = indices.GetValue(i);
                float var = varsfp32.GetValue(indice);
                float update = updatesfp32.GetValue(i);
                var = var * update;
                varsfp32.SetValue(indice, var);
            }
        }
        Cast(vars, varsfp32, RoundMode::CAST_RINT, varTileLenAlign32);
        // printf("[var] ");
        // for (int i = 0; i < varTileLen; i++) {
        //     printf("%f ,", vars.GetValue(i));
        // }
        // printf("\n");
        // TODO: Brcb 一次只能处理 255 * 8 个元素
        Brcb(out, vars, (uint8_t)varTileLenAlign32 / 8, {1, 8});
        // printf("[out] ");
        // for (int i = 0; i < varTileLen; i++) {
        //     printf("%f ,", out.GetValue(i * 16));
        // }
        // printf("\n");
        outQ.EnQue<half>(out);
        varsQ.FreeTensor(vars);
        updatesQ.FreeTensor(updates);
        indicesQ.FreeTensor(indices);
    }
    
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataLen, uint16_t repeatTime) {
        uint32_t blockLen = dataLen * sizeof(half);
        uint32_t dstStride = (varStride - 1) * sizeof(half);
        LocalTensor<half> out = outQ.DeQue<half>();
        DataCopyExtParams copyParams {repeatTime, blockLen, 0, dstStride, 0};
        DataCopyPad(varsGm[offset], out, copyParams);
        outQ.FreeTensor(out);
    }
private:
    TPipe pipe;
    GlobalTensor<half> varsGm, updatesGm;
    GlobalTensor<int32_t> indicesGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> varsQ, indicesQ, updatesQ;
    TQue<QuePosition::VECIN, BUFFER_NUM> outQ;
    TBuf<QuePosition::VECCALC> varsfp32Buf, updatesfp32Buf;
    TBuf<QuePosition::VECCALC> mask1Buf, mask2Buf;

    int64_t tileLength;
    int64_t tileNum;
    int64_t tailTileLength;
    int64_t tileLengthAlign32;
    int64_t varLen;
    int64_t indicesLen;
    int64_t varStride;
    int64_t indiceStride;
    int64_t mode;
    int64_t varTileLen;
    int64_t varTileLenAlign32;
    int64_t indiceTileLen;
    int64_t indiceTileLenAlign32;
    int64_t updateTileLen;
    int64_t updateTileLenAlign32;
};

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
    op.Init(
        var,
        indices,
        updates,
        t.tileLength,
        t.tileNum,
        t.tailTileLength,
        t.varLen,
        t.indicesLen,
        t.varStride,
        t.indiceStride,
        t.mode,
        t.varTileLen,
        t.indiceTileLen
    );
    op.Process();
}