# Softmax

## 生成工程

```bash
msopgen gen -i softmax.json -f pytorch -c ai_core-ascend310B -lan cpp -out softmax/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## TODO
- [ ] Case 5 性能测试失败，修改 `Tile` 的长度