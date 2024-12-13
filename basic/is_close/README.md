# IsClose

## 生成工程

```bash
msopgen gen -i is_close.json -f pytorch -c ai_core-ascend310B -lan cpp -out is_close/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## Note

- case1: fp16 pass
- case2: fp32 pass
- case3: int32 pass
- case4: 
- case5: fp32 pass