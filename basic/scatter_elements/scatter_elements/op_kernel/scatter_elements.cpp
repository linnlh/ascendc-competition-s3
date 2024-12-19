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
        pipe.InitBuffer(varsfp32Buf, (shapes[0][axis] + 7) / 8 * 32);
        pipe.InitBuffer(updatesfp32Buf, (shapes[1][axis] + 7) / 8 * 32);
        this->dimNum = dimNum;
        this->mode = mode;
        this->axis = axis;
    }

    __aicore__ inline void Process() {
        // 优化连续存储的数据读取
        if (axis == dimNum - 1) {
            for (int i = 0; i < loopCount; i++) {
                int varOffset = i * shapes[0][axis];
                int indiceOffset = i * shapes[1][axis];
                CopyVars(varOffset, shapes[0][axis], 1);
                CopyUpdates(indiceOffset, shapes[1][axis], 1);
                CopyIndices(indiceOffset, shapes[1][axis], 1);
                ComputeContiuous();
                CopyOut(varOffset, shapes[0][axis], 1);
            }
        }
        else {
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
                CopyVars(varOffset, 1, shapes[0][axis]);
                CopyIndices(indiceOffset, 1, shapes[1][axis]);
                CopyUpdates(indiceOffset, 1, shapes[1][axis]);
                Compute();
                CopyOut(varOffset, 1, shapes[1][axis]);
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
        for (int i = 0; i < shapes[1][axis]; i++) {
            int32_t indice = indices.GetValue(i * 8);
            float updated = updatesfp32.GetValue(i * 16);
            float var = varsfp32.GetValue(indice * 16);
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

    __aicore__ inline void ComputeContiuous() {
        LocalTensor<half> vars = varQ.DeQue<half>();
        LocalTensor<half> updates = updatesQ.DeQue<half>();
        LocalTensor<int32_t> indices = indicesQ.DeQue<int32_t>();
        LocalTensor<half> out = outQ.AllocTensor<half>();
        LocalTensor<float> varsfp32 = varsfp32Buf.Get<float>();
        LocalTensor<float> updatesfp32 = updatesfp32Buf.Get<float>();
        Cast(varsfp32, vars, RoundMode::CAST_NONE, shapes[0][axis]);
        Cast(updatesfp32, updates, RoundMode::CAST_NONE, shapes[1][axis]);
        if (mode == 0) {
            for (int i = 0; i < shapes[1][axis]; i++) {
                int32_t indice = indices.GetValue(i);
                float var = varsfp32.GetValue(indice);
                float update = updatesfp32.GetValue(i);
                varsfp32.SetValue(indice, update);
            }
        }
        else if (mode == 1) {
            for (int i = 0; i < shapes[1][axis]; i++) {
                int32_t indice = indices.GetValue(i);
                float var = varsfp32.GetValue(indice);
                float update = updatesfp32.GetValue(i);
                var = var + update;
                varsfp32.SetValue(indice, var);
            }
        }
        else {
            for (int i = 0; i < shapes[1][axis]; i++) {
                int32_t indice = indices.GetValue(i);
                float var = varsfp32.GetValue(indice);
                float update = updatesfp32.GetValue(i);
                var = var * update;
                varsfp32.SetValue(indice, var);
            }
        }
        Cast(out, varsfp32, RoundMode::CAST_NONE, shapes[0][axis]);
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



extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelScatterElements<DTYPE_VAR> op;
    op.Init(var, indices, updates, t.varShape, t.indicesShape, t.dimNum, t.mode, t.axis);
    op.Process();
}