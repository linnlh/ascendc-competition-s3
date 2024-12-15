# Div

## 生成工程

```bash
msopgen gen -i div.json -f pytorch -c ai_core-ascend310B -lan cpp -out div/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## Note

- case1: fp16
- case2: fp32
- case3: int8
- case4: fp16
- case5: int32

## TODO

- [ ] 整理 `div` 整体代码