# NonMaxSuppression

## 生成工程

```bash
msopgen gen -i non_max_suppression.json -f pytorch -c ai_core-ascend310B -lan cpp -out non_max_suppression/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## TODO

- [ ] 根据 `UB` 调整单次循环的 `tiling`
- [ ] 设置 `DOUBLE BUFFER`