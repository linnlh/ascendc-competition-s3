# Asinh

## 生成工程

```bash
msopgen gen -i asinh.json -f pytorch -c ai_core-ascend310B -lan cpp -out asinh/
```

修改 `CMakePresets.json` 下 `ASCEND_CANN_PACKAGE_PATH` 为 `CANN` 的实际安装路径

> -f 指定为 aclnn 时框架代码只生成 .so 文件，不生成 .run 文件

## TODO

- [ ] 优化 `tileLength` 的分配
- [ ] 实现 `double buffer`
 
## Notes

- case1: fp32, Passed
- case2: fp16, Passed
- case3: fp32, Passed
- case4: fp16, Passed
- case5: fp32, Passed