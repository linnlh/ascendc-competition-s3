# ascendc-competition-s3
昇腾AI原生创新算子挑战赛（S3赛季）

## 基础赛道

| 序号 | Op | (case 5) 性能测试 | benchmark | 状态 |
| ----- | ----- | ----- | ----- | ----- |
| 1 | [Softmax](basic/softmax/README.md) | 6897.02 | 530.4 | 4/5 |
| 2 | [Div](basic/div/README.md) | 1592.04 | 1760.2752 | Passed |
| 3 | [NotEqual](basic/not_equal/README.md) | 1272.23 | 2799.0744 | Passed |
| 4 | [Asinh](basic/asinh/README.md) | 3207.77 | 5625.8748 | Passed |
| 5 | [AsinhGrad](basic/asinh_grad/README.md) | 1104.12 | 1031.4504 | 4/5 |
| 6 | [ScatterElements](basic/scatter_elements/README.md) | 4841.79 | 8993.2152 | Passed |
| 7 | [NonMaxSuppression](basic/non_max_suppression/README.md) | 9025.15 | 40353.77016 | Passed |
| 8 | [IsClose](basic/is_close/README.md) | 3511.83 | 32354.4 | Passed |
| 9 | [LogSumExp](basic/logsum_exp/README.md) | 321611.44 | 43584.424 | 4/5 |
| 10 | [ReplicationPad2d](basic/replication_pad2d/README.md) | 3462.94 | 4594.56 | Passed |

## 性能赛道

| 序号 | Op | (case 5) 性能测试 (us) | 状态 |
| ----- | ----- | ----- | ----- |
| 3 | [NLLLoss](perf/nll_loss/README.md) | 15.28 | Passed |