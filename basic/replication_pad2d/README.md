# ReplicationPad2d

## 生成工程

```bash
msopgen gen -i replication_pad2d.json -f pytorch -c ai_core-ascend310B -lan cpp -out replication_pad2d/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## Note

性能测试失败