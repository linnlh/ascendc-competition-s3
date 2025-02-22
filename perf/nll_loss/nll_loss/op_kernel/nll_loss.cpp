#include "kernel_operator.h"

using namespace AscendC;

#include "kernel_operator.h"

using namespace AscendC;
constexpr int BUFFER_NUM = 2;
constexpr int BLOCK_SIZE = 32;

class KernelNLLLoss {

constexpr static int elemPerBlk = 8;

public:
    __aicore__ inline KernelNLLLoss() {}

    __aicore__ inline void Init(
        TPipe* pipe,
        GM_ADDR x,
        GM_ADDR target,
        GM_ADDR weight,
        GM_ADDR y,
        GM_ADDR workspace,
        NLLLossTilingData* tiling
    ) {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t bigCoreNum = tiling->bigCoreNum;
        uint32_t bigCoreBatchSize = tiling->bigCoreBatchSize;
        uint32_t batchIdx = bigCoreBatchSize * coreIdx;
        // 小核
        if (coreIdx >= bigCoreNum) {
            uint32_t smallCoreBatchSize = tiling->smallCoreBatchSize;
            batchIdx -= (coreIdx - bigCoreNum) * (bigCoreBatchSize - smallCoreBatchSize);
            this->tileLen = tiling->smallCoreTileLen;
            this->tileNum = tiling->smallCoreTileNum;
            this->tailTileLen = tiling->smallCoreTailTileLen;
            this->batchSize = smallCoreBatchSize;
        }
        else {
            this->tileLen = tiling->bigCoreTileLen;
            this->tileNum = tiling->bigCoreTileNum;
            this->tailTileLen = tiling->bigCoreTailTileLen;
            this->batchSize = bigCoreBatchSize;
        }
        this->clsNum = tiling->clsNum;
        this->reduce = tiling->reduce;
        this->pipe = pipe;
        
        xGm.SetGlobalBuffer((__gm__ float *)x + batchIdx * clsNum, batchSize * clsNum);
        targetGm.SetGlobalBuffer((__gm__ int32_t *)target + batchIdx, batchSize);
        weightGm.SetGlobalBuffer((__gm__ float *)weight);
        yGm.SetGlobalBuffer((__gm__ float *)y);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace + coreIdx);
        pipe->InitBuffer(xQueue, BUFFER_NUM, (tileLen * clsNum + 7) / 8 * 8 * sizeof(float));
        pipe->InitBuffer(yQueue, BUFFER_NUM, GetBlockNum() * sizeof(float));
        pipe->InitBuffer(weightSumQ, BUFFER_NUM, GetBlockNum() * sizeof(float));
        pipe->InitBuffer(targetQ, BUFFER_NUM, (tileLen + 7) / 8 * 8 * sizeof(float));
        pipe->InitBuffer(weightQ, 1, (clsNum + 7) / 8 * 8 * sizeof(float));
        pipe->InitBuffer(indicesBuf, tileLen * sizeof(int32_t));
        pipe->InitBuffer(buf1, tileLen * sizeof(float));
        pipe->InitBuffer(buf2, tileLen * sizeof(float));

        LocalTensor<int32_t> indices = indicesBuf.Get<int32_t>();
        CreateVecIndex(indices, 0, tileLen);
        Muls(indices, indices, int32_t(4 * clsNum), tileLen);
    }

    __aicore__ inline void Process() {
        uint64_t xOffset = 0;
        uint64_t targetOffset = 0;
        CopyWeight();
        weight = weightQ.DeQue<float>();
        for (int i = 0; i < tileNum; i++) {
            CopyX(xOffset, tileLen * clsNum, 1);
            CopyTarget(targetOffset, tileLen, 1);
            Compute(tileLen);
            xOffset += tileLen * clsNum;
            targetOffset += tileLen;
        }
        if (tailTileLen > 0) {
            CopyX(xOffset, tailTileLen * clsNum, 1);
            CopyTarget(targetOffset, tailTileLen, 1);
            Compute(tailTileLen);
        }
        weightQ.FreeTensor(weight);
        LocalTensor<float> score = yQueue.AllocTensor<float>();
        score.SetValue(0, -scoreVal);
        yQueue.EnQue<float>(score);
        if (reduce == 1) {
            LocalTensor<float> weightSum = weightSumQ.AllocTensor<float>();
            weightSum.SetValue(0, weightVal);
            weightSumQ.EnQue<float>(weightSum);
        }
        CopyOut();
        SyncAll();
        if (GetBlockIdx() == 0) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm);
            float scoreSum = .0f;
            uint64_t coreNum = GetBlockNum();
            for (int i = 0;  i < coreNum; i++) {
                scoreSum += workspaceGm.GetValue(i);
            }
            if (reduce == 1) {
                float weightSum = .0f;
                for (int i = 0;  i < coreNum; i++) {
                    weightSum += workspaceGm.GetValue(coreNum + i);
                }
                scoreSum /= weightSum;
            }
            yGm.SetValue(0, scoreSum);
        }
    }

private:
    __aicore__ inline void CopyWeight() {
        uint32_t dataSize = clsNum * 4;
        LocalTensor<float> weight = weightQ.AllocTensor<float>();
        DataCopyExtParams copyParams {1, dataSize, 0, 0, 0};
        DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
        DataCopyPad(weight, weightGm, copyParams, padParams);
        weightQ.EnQue<float>(weight);
    }

    __aicore__ inline void CopyX(uint64_t offset, int dataLen, uint16_t repeatTime) {
        uint32_t dataSize = dataLen * 4;
        LocalTensor<float> x = xQueue.AllocTensor<float>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        xQueue.EnQue<float>(x);
    }

    __aicore__ inline void CopyTarget(uint64_t offset, int dataLen, uint16_t repeatTime) {
        uint32_t dataSize = dataLen * 4;
        LocalTensor<int32_t> target = targetQ.AllocTensor<int32_t>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
        DataCopyPad(target, targetGm[offset], copyParams, padParams);
        targetQ.EnQue<int32_t>(target);
    }

    __aicore__ inline void Compute(int computeLen) {
        LocalTensor<float> x = xQueue.DeQue<float>();
        LocalTensor<int32_t> target = targetQ.DeQue<int32_t>();
        LocalTensor<int32_t> indices = indicesBuf.Get<int32_t>();
        LocalTensor<float> work1 = buf1.Get<float>();
        // 收集权重值
        Muls(target, target, 4, computeLen);
        Gather(work1, weight, target.ReinterpretCast<uint32_t>(), uint32_t(0), computeLen);
        if (reduce == 1) {
            LocalTensor<float> work2 = buf2.Get<float>();
            ReduceSum(work2, work1, work2, computeLen);
            weightVal += work2.GetValue(0);
        }
        // 计算索引
        Add(target, target, indices, computeLen);
        // 收集分数值
        Gather(x, x, target.ReinterpretCast<uint32_t>(), uint32_t(0), computeLen);
        Mul(x, x, work1, computeLen);
        ReduceSum(x, x, work1, computeLen);
        scoreVal += x.GetValue(0);

        xQueue.FreeTensor(x);
        targetQ.FreeTensor(target);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> y = yQueue.DeQue<float>();
        DataCopyExtParams copyParams {(uint16_t)1, (uint32_t)4, 0, 0, 0};
        DataCopyPad(workspaceGm, y, copyParams);
        yQueue.FreeTensor(y);
        if (reduce == 1) {
            LocalTensor<float> weightSum = weightSumQ.DeQue<float>();
            DataCopyPad(workspaceGm[GetBlockNum()], weightSum, copyParams);
            weightSumQ.FreeTensor(weightSum);
        }
    }

private:
    TPipe* pipe;

    GlobalTensor<float> xGm, weightGm, yGm, workspaceGm;
    GlobalTensor<int32_t> targetGm;

    LocalTensor<float> weight;

    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue, targetQ, weightQ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue, weightSumQ;
    TBuf<QuePosition::VECCALC> indicesBuf, buf1, buf2;

    uint32_t batchSize, tileLen, tileNum, tailTileLen;
    uint32_t clsNum, reduce;

    float weightVal = .0f;
    float scoreVal = .0f;
};


extern "C" __global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR work = GetUserWorkspace(workspace);
    KernelNLLLoss op;
    TPipe pipe;
    op.Init(&pipe, x, target, weight, y, work, &tiling_data);
    op.Process();
}