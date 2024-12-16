# NotEqual

## 生成工程

```bash
msopgen gen -i not_equal.json -f pytorch -c ai_core-ascend310B -lan cpp -out not_equal/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## TODO

- [ ] min(max(x1, x2) - min(x1, x2), 1) 可以减少 buffer 的使用

## Note
- case1: fp16
- case2: fp32
- case3: int8
- case5: int32