# LogSumExp

## 生成工程

```bash
msopgen gen -i logsum_exp.json -f pytorch -c ai_core-ascend310B -lan cpp -out logsum_exp/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## Note

性能测试失败
- case4: axis dim > 1, 并且是连续的
- case5 用例：dim 长度为 2，并且不包含最后一维