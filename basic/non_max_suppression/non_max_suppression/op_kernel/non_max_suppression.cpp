#include <iostream>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelNonMaxSuppression {
public:
    __aicore__ inline KernelNonMaxSuppression() {}
    __aicore__ inline void Init(
        GM_ADDR boxes,
        GM_ADDR scores,
        GM_ADDR max_output_boxes_per_class,
        GM_ADDR iou_threshold,
        GM_ADDR score_threshold,
        GM_ADDR selected_indices,
        int64_t batchSize,
        int64_t classNum,
        int64_t boxNum
    ) {
        boxesGm.SetGlobalBuffer((__gm__ float*)boxes);
        scoresGm.SetGlobalBuffer((__gm__ float*)scores);
        maxOutputBoxesPerClassGm.SetGlobalBuffer((__gm__ int32_t*)max_output_boxes_per_class);
        scoreThreGm.SetGlobalBuffer((__gm__ float*)score_threshold);
        iouThreGm.SetGlobalBuffer((__gm__ float*)iou_threshold);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)selected_indices);

        int64_t boxNumAlign2 = (boxNum + 1) / 2 * 2;
        int64_t boxNumAlign32 = (boxNum + 31) / 32 * 32;
        pipe.InitBuffer(boxesQ, BUFFER_NUM, boxNumAlign2 * 4 * sizeof(float));
        pipe.InitBuffer(scoresQ, BUFFER_NUM, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(indicesQ, BUFFER_NUM, 8 * sizeof(int32_t));
        pipe.InitBuffer(maxValueBuf, 8 * sizeof(float));
        pipe.InitBuffer(workBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(maskBuf, boxNumAlign32 * sizeof(uint16_t));
        pipe.InitBuffer(flagBuf, boxNumAlign32 * sizeof(uint16_t));        

        pipe.InitBuffer(temp1Buf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(temp2Buf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(x1MinBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(x1MaxBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(x2MinBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(x2MaxBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(y1MinBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(y1MaxBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(y2MinBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(y2MaxBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(intersectXMinBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(intersectXMaxBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(intersectYMinBuf, boxNumAlign32 * sizeof(float));
        pipe.InitBuffer(intersectYMaxBuf, boxNumAlign32 * sizeof(float));

        this->scoreThreshold = scoreThreGm.GetValue(0);
        this->iouThreshold = iouThreGm.GetValue(0);
        this->maxOutputBoxesPerClass = maxOutputBoxesPerClassGm.GetValue(0);
        this->batchSize = batchSize;
        this->classNum = classNum;
        this->boxNum = boxNum;
        this->boxNumAlign32 = boxNumAlign32;
        this->boxNumAlign2 = boxNumAlign2;
        this->indiceOffset = 0;
    }

    __aicore__ inline void Process() {
        for (int batch = 0; batch < batchSize; batch++) {
            CopyBoxes(batch);
            LocalTensor<float> boxes = boxesQ.DeQue<float>();
            for (int cls = 0; cls < classNum; cls++) {
                CopyScores(batch * classNum + cls);
                Compute(boxes, batch, cls);
            }
            boxesQ.FreeTensor(boxes);
        }
    }

private:
    __aicore__ inline void CopyScores(int progress) {
        uint32_t blockLen = boxNum * sizeof(float);
        LocalTensor<float> scores = scoresQ.AllocTensor<float>();
        DataCopyExtParams copyParams {1, blockLen, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(scores, scoresGm[progress * boxNum], copyParams, padParams);
        scoresQ.EnQue<float>(scores);
    }

    __aicore__ inline void CopyBoxes(int progress) {
        LocalTensor<float> boxes = boxesQ.AllocTensor<float>();
        DataCopy(boxes, boxesGm[progress * boxNum * 4], boxNumAlign2 * 4);
        boxesQ.EnQue<float>(boxes);
    }

    __aicore__ inline void Compute(LocalTensor<float>& boxes, int32_t batch, int32_t cls) {
        LocalTensor<float> scores = scoresQ.DeQue<float>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
        CompareScalar(mask, scores, scoreThreshold, CMPMODE::GT, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Select(scores, mask.ReinterpretCast<uint8_t>(), scores, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < maxOutputBoxesPerClass; i++) {
            ReduceMax(work, scores, work, boxNum, true);
            PipeBarrier<PIPE_V>();
            if (work.GetValue(0) < .0f) break;
            float idxFp = work.GetValue(1);
            uint32_t idx = *reinterpret_cast<uint32_t*>(&idxFp);
            uint32_t offset = idx * 4;
            SuppressByIOU(
                boxes.GetValue(offset),
                boxes.GetValue(offset + 1),
                boxes.GetValue(offset + 2),
                boxes.GetValue(offset + 3),
                boxes,
                mask
            );
            Select(scores, mask.ReinterpretCast<uint8_t>(), scores, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, boxNumAlign32);
            scores.SetValue(idx, -1.0f);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> indices = indicesQ.AllocTensor<int32_t>();
            indices.SetValue(0, batch);
            indices.SetValue(1, cls);
            indices.SetValue(2, idx);
            indicesQ.EnQue<int32_t>(indices);
            CopyOut(indiceOffset);
            indiceOffset += 1;
        }
        scoresQ.FreeTensor(scores);
    }

    __aicore__ inline void CopyOut(int progress) {
        LocalTensor<int32_t> indices = indicesQ.DeQue<int32_t>();
        DataCopy(indicesGm[progress * 3], indices, 8);
        indicesQ.FreeTensor(indices);
    }

    __aicore__ inline void SuppressByIOU(
        float y1,
        float x1,
        float y2,
        float x2,
        LocalTensor<float>& boxes,
        LocalTensor<uint16_t>& mask
    ) {
        LocalTensor<float> temp1 = temp1Buf.Get<float>();
        LocalTensor<float> temp2 = temp2Buf.Get<float>();
        LocalTensor<float> x1Min = x1MinBuf.Get<float>();
        LocalTensor<float> x1Max = x1MaxBuf.Get<float>();
        LocalTensor<float> y1Min = y1MinBuf.Get<float>();
        LocalTensor<float> y1Max = y1MaxBuf.Get<float>();
        LocalTensor<float> x2Min = x2MinBuf.Get<float>();
        LocalTensor<float> x2Max = x2MaxBuf.Get<float>();
        LocalTensor<float> y2Min = y2MinBuf.Get<float>();
        LocalTensor<float> y2Max = y2MaxBuf.Get<float>();
        LocalTensor<float> intersectXMin = intersectXMinBuf.Get<float>();
        LocalTensor<float> intersectXMax = intersectXMaxBuf.Get<float>();
        LocalTensor<float> intersectYMin = intersectYMinBuf.Get<float>();
        LocalTensor<float> intersectYMax = intersectYMaxBuf.Get<float>();
        // 将 flag 进行清零
        LocalTensor<uint16_t> flag = flagBuf.Get<uint16_t>();
        Not(mask, flag, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        And(flag, flag, mask, boxNumAlign32);
        
        float y1_min, y1_max, x1_min, x1_max;
        MaxMin(y1, y2, y1_min, y1_max);
        MaxMin(x1, x2, x1_min, x1_max);
        Duplicate(x1Min, x1_min, boxNumAlign32);
        Duplicate(x1Max, x1_max, boxNumAlign32);
        Duplicate(y1Min, y1_min, boxNumAlign32);
        Duplicate(y1Max, y1_max, boxNumAlign32);

        uint64_t rsvdCnt = 0;
        uint16_t repeatTime = (boxNumAlign2 * 4 + 63) / 64;
        GatherMask(temp1, boxes, 3, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
        GatherMask(temp2, boxes, 5, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Min(y2Min, temp1, temp2, boxNumAlign32);
        Max(y2Max, temp1, temp2, boxNumAlign32);
        GatherMask(temp1, boxes, 4, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
        GatherMask(temp2, boxes, 6, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Min(x2Min, temp1, temp2, boxNumAlign32);
        Max(x2Max, temp1, temp2, boxNumAlign32);
        PipeBarrier<PIPE_V>();

        Max(intersectXMin, x1Min, x2Min, boxNumAlign32);
        Min(intersectXMax, x1Max, x2Max, boxNumAlign32);
        Max(intersectYMin, y1Min, y2Min, boxNumAlign32);
        Min(intersectYMax, y1Max, y2Max, boxNumAlign32);
        PipeBarrier<PIPE_V>();

        Compare(mask, intersectXMax, intersectXMin, CMPMODE::LE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Or(flag, flag, mask, boxNumAlign32);
        Compare(mask, intersectYMax, intersectYMin, CMPMODE::LE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Or(flag, flag, mask, boxNumAlign32);

        // intersection area -> intersectXMax
        Sub(intersectXMax, intersectXMax, intersectXMin, boxNumAlign32);
        Sub(intersectYMax, intersectYMax, intersectYMin, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Mul(intersectXMax, intersectXMax, intersectYMax, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, intersectXMax, .0f, CMPMODE::LE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Or(flag, flag, mask, boxNumAlign32);
        
        // area1 -> x1Max
        Sub(x1Max, x1Max, x1Min, boxNumAlign32);
        Sub(y1Max, y1Max, y1Min, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Mul(x1Max, x1Max, y1Max, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, x1Max, .0f, CMPMODE::LE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Or(flag, flag, mask, boxNumAlign32);

        // area2 -> x2Max
        Sub(x2Max, x2Max, x2Min, boxNumAlign32);
        Sub(y2Max, y2Max, y2Min, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Mul(x2Max, x2Max, y2Max, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, x2Max, .0f, CMPMODE::LE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Or(flag, flag, mask, boxNumAlign32);

        // union = area1 + area2 - intersection area -> x1Max
        Add(x1Max, x1Max, x2Max, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Sub(x1Max, x1Max, intersectXMax, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, x1Max, .0f, CMPMODE::LE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Or(flag, flag, mask, boxNumAlign32);

        // iou = intersection / union -> x1Max
        Div(x1Max, intersectXMax, x1Max, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, x1Max, iouThreshold, CMPMODE::LE, boxNumAlign32);
        PipeBarrier<PIPE_V>();
        Or(mask, mask, flag, boxNumAlign32);
        PipeBarrier<PIPE_V>();
    }


    __aicore__ inline bool SuppressByIOUNaive(
        const LocalTensor<float>& box1,
        const LocalTensor<float>& box2
    ) {
        float y1_1 = box1.GetValue(0);
        float x1_1 = box1.GetValue(1);
        float y2_1 = box1.GetValue(2);
        float x2_1 = box1.GetValue(3);
        float y1_2 = box2.GetValue(0);
        float x1_2 = box2.GetValue(1);
        float y2_2 = box2.GetValue(2);
        float x2_2 = box2.GetValue(3);
        float x1_min, y1_min, x1_max, y1_max;
        float x2_min, y2_min, x2_max, y2_max;
        float intersection_x_min, intersection_x_max, intersection_y_min, intersection_y_max;
        MaxMin(x1_1, x2_1, x1_min, x1_max);
        MaxMin(x1_2, x2_2, x2_min, x2_max);
        intersection_x_min = HelperMax(x1_min, x2_min);
        intersection_x_max = HelperMin(x1_max, x2_max);
        if (intersection_x_max <= intersection_x_min) {
            return false;
        }
        MaxMin(y1_1, y2_1, y1_min, y1_max);
        MaxMin(y1_2, y2_2, y2_min, y2_max);
        intersection_y_min = HelperMax(y1_min, y2_min);
        intersection_y_max = HelperMin(y1_max, y2_max);
        if (intersection_y_max <= intersection_y_min) {
            return false;
        }
        float intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min);
        if (intersection_area <= .0f) {
            return false;
        }
        float area1 = (x1_max - x1_min) * (y1_max - y1_min);
        float area2 = (x2_max - x2_min) * (y2_max - y2_min);
        float union_area = area1 + area2 - intersection_area;
        if (area1 <= .0f || area2 <= .0f || union_area <= .0f) {
            return false;
        }
        float iou = intersection_area / union_area;
        return iou > iouThreshold;
    }

    __aicore__ inline void MaxMin(float lhs, float rhs, float& min, float& max) {
        if (lhs >= rhs) {
            min = rhs;
            max = lhs;
        } else {
            min = lhs;
            max = rhs;
        }
    }

    __aicore__ inline float HelperMin(float a, float b) {
        if (b < a) {
            return b;
        }
        return a;
    }

    __aicore__ inline float HelperMax(float a, float b) {
        if (b > a) {
            return b;
        }
        return a;
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> scoresQ, boxesQ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> indicesQ;
    TBuf<QuePosition::VECCALC> maxValueBuf, workBuf;
    TBuf<QuePosition::VECCALC> maskBuf, flagBuf;

    TBuf<QuePosition::VECCALC> temp1Buf, temp2Buf;
    TBuf<QuePosition::VECCALC> x1MinBuf, x1MaxBuf, x2MinBuf, x2MaxBuf;
    TBuf<QuePosition::VECCALC> y1MinBuf, y1MaxBuf, y2MinBuf, y2MaxBuf;
    TBuf<QuePosition::VECCALC> intersectXMinBuf, intersectXMaxBuf;
    TBuf<QuePosition::VECCALC> intersectYMinBuf, intersectYMaxBuf;

    GlobalTensor<float> boxesGm, scoresGm, scoreThreGm, iouThreGm;
    GlobalTensor<int32_t> indicesGm, maxOutputBoxesPerClassGm;

    int64_t batchSize;
    int64_t classNum;
    int64_t boxNum;
    int64_t boxNumAlign32;
    int64_t boxNumAlign2;
    int64_t indiceOffset;
    int64_t maxOutputBoxesPerClass;
    float scoreThreshold;
    float iouThreshold;
};

extern "C" __global__ __aicore__ void non_max_suppression(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold, GM_ADDR selected_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    KernelNonMaxSuppression op;
    op.Init(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        selected_indices,
        t.batchSize,
        t.classNum,
        t.boxNum
    );
    op.Process();
}