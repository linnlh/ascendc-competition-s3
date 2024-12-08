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

        pipe.InitBuffer(scoresQ, BUFFER_NUM, 32);
        pipe.InitBuffer(indicesQ, BUFFER_NUM, 32);
        this->scoreThreshold = scoreThreGm.GetValue(0);
        this->iouThreshold = iouThreGm.GetValue(0);
        this->maxOutputBoxesPerClass = maxOutputBoxesPerClassGm.GetValue(0);
        this->batchSize = batchSize;
        this->classNum = classNum;
        this->boxNum = boxNum;
    }

    __aicore__ inline void Process() {
        // int loopCount = batchSize * classNum;
        // for (int i = 0; i < loopCount; i++) {
        //     CopyIn(i);
        //     Compute();
        //     CopyOut(i);
        // }
        Compute();
    }

private:
    __aicore__ inline void CopyIn(int progress) {
        LocalTensor<float> scores = scoresQ.AllocTensor<float>();
        DataCopy(scores, scoresGm[progress * boxNum], boxNum);
        scoresQ.EnQue<float>(scores);
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

    __aicore__ inline void Compute() {
        for (int batch = 0; batch < batchSize; batch++) {
            for (int cls = 0; cls < classNum; cls++) {
                int scoreOffset = (batch * classNum + cls) * boxNum;
                int boxOffset = batch * boxNum * 4;
                for (int k = 0; k < maxOutputBoxesPerClass; k++) {
                    float maxScore = .0f;
                    int maxIdx = -1;
                    for (int idx = 0; idx < boxNum; idx++) {
                        float score = scoresGm.GetValue(scoreOffset + idx);
                        if (score > scoreThreshold && score > maxScore) {
                            maxScore = score;
                            maxIdx = idx;
                        }
                    }
                    if (maxIdx == -1)
                        break;

                    float y1_1 = boxesGm.GetValue(boxOffset + maxIdx * 4);
                    float x1_1 = boxesGm.GetValue(boxOffset + maxIdx * 4 + 1);
                    float y2_1 = boxesGm.GetValue(boxOffset + maxIdx * 4 + 2);
                    float x2_1 = boxesGm.GetValue(boxOffset + maxIdx * 4 + 3);
                    float x1_min;
                    float y1_min;
                    float x1_max;
                    float y1_max;
                    float x2_min;
                    float y2_min;
                    float x2_max;
                    float y2_max;
                    float intersection_x_min;
                    float intersection_x_max;
                    float intersection_y_min;
                    float intersection_y_max;
                    for (int idx = 0; idx < boxNum; idx++) {
                        if (idx == maxIdx)
                            continue;
                        float y1_2 = boxesGm.GetValue(boxOffset + idx * 4);
                        float x1_2 = boxesGm.GetValue(boxOffset + idx * 4 + 1);
                        float y2_2 = boxesGm.GetValue(boxOffset + idx * 4 + 2);
                        float x2_2 = boxesGm.GetValue(boxOffset + idx * 4 + 3);
                        MaxMin(x1_1, x2_1, x1_min, x1_max);
                        MaxMin(x1_2, x2_2, x2_min, x2_max);
                        intersection_x_min = HelperMax(x1_min, x2_min);
                        intersection_x_max = HelperMin(x1_max, x2_max);
                        if (intersection_x_max <= intersection_x_min) {
                            // scoresGm.SetValue(scoreOffset + idx, -1);
                            continue;
                        }

                        MaxMin(y1_1, y2_1, y1_min, y1_max);
                        MaxMin(y1_2, y2_2, y2_min, y2_max);
                        intersection_y_min = HelperMax(y1_min, y2_min);
                        intersection_y_max = HelperMin(y1_max, y2_max);
                        if (intersection_y_max <= intersection_y_min) {
                            // scoresGm.SetValue(scoreOffset + idx, -1);
                            continue;
                        }
                        // printf("max idx: %d, cur idx: %d\n", maxIdx, idx);
                        // printf("[x min]: %f, [x max]: %f [y min]: %f [y max]: %f\n", intersection_x_min, intersection_x_max, intersection_y_min, intersection_y_max);
                        float intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min);
                        // printf("[intersection area]: %f\n", intersection_area);
                        if (intersection_area <= .0f) {
                            // scoresGm.SetValue(scoreOffset + idx, -1);
                            continue;
                        }
                        float area1 = (x1_max - x1_min) * (y1_max - y1_min);
                        float area2 = (x2_max - x2_min) * (y2_max - y2_min);
                        float union_area = area1 + area2 - intersection_area;
                        // printf("[area1]: %f, [area2]: %f, [union area]: %f\n", area1, area2, union_area);
                        if (area1 <= .0f || area2 <= .0f || union_area <= .0f) {
                            // scoresGm.SetValue(scoreOffset + idx, -1);
                            continue;
                        }
                        float iou = intersection_area / union_area;
                        // printf("[iou]: %f\n", iou);
                        if (iou > iouThreshold) {
                            scoresGm.SetValue(scoreOffset + idx, -1);
                        }
                    }
                    int indiceOffset = ((batch * classNum + cls) * maxOutputBoxesPerClass + k) * 3;
                    scoresGm.SetValue(scoreOffset + maxIdx, -1);
                    indicesGm.SetValue(indiceOffset, batch);
                    indicesGm.SetValue(indiceOffset + 1, cls);
                    indicesGm.SetValue(indiceOffset + 2, maxIdx);
                }
            }
        }



        // LocalTensor<float> scores = scoresQ.DeQue<float>();
        // LocalTensor<int32_t> indices = indicesQ.AllocTensor<int32_t>();
        // float maxScore = .0f;
        // int32_t maxIdx = 0;
        // for (int i = 0; i < maxOutputBoxesPerClass; i++) {
        //     for (int j = 0; j < boxNum; j++) {
        //         float score = scores.GetValue(j);
        //         if (score > maxScore) {
        //             maxScore = score;
        //             maxIdx = j;
        //         }
        //     }
        //     scores.SetValue(maxIdx, 0);
        //     indices.SetValue(i, maxIdx);
        // }
        // printf("[indice]: ");
        // for (int i = 0; i < maxOutputBoxesPerClass; i++) {
        //     printf("%f ", indices.GetValue(i));
        // }
        // printf("\n");
        // scoresQ.FreeTensor(scores);
    }

    __aicore__ inline void CopyOut(int progress) {
        LocalTensor<int32_t> indices = indicesQ.DeQue<int32_t>();
        DataCopy(indicesGm[progress * maxOutputBoxesPerClass], indices, boxNumAlign32);
        indicesQ.FreeTensor(indices);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> scoresQ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> indicesQ;

    GlobalTensor<float> boxesGm, scoresGm, scoreThreGm, iouThreGm;
    GlobalTensor<int32_t> indicesGm, maxOutputBoxesPerClassGm;

    int64_t batchSize;
    int64_t classNum;
    int64_t boxNum;
    int64_t boxNumAlign32 = 32;
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