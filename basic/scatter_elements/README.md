# ScatterElements

## 生成工程

```bash
msopgen gen -i scatter_elements.json -f pytorch -c ai_core-ascend310B -lan cpp -out scatter_elements/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## Note

- case1: fp16, Passed
- case2: Passed
- case3: Passed/Wrong answer
- case4: fp16? Run failed
- case5: fp32 Passed/性能测试失败 wrong answer