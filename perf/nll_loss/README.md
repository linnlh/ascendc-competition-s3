# NLLLoss

## 生成工程

```bash
msopgen gen -i nll_loss.json -c ai_core-ascend910b -lan cpp -out nll_loss/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径，在 `modelart` 下对应的路径是 `/home/ma-user/Ascend/ascend-toolkit/latest`

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## Note

- case5: 使用 `mean` reduction