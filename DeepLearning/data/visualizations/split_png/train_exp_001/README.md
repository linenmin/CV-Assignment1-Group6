# train 导出 PNG

- 来源: `data\splits\exp_001\train.csv`
- 共 64 行 CSV，写出成功 64 张
- 文件名格式: `序号_idXXX_classC.png`（序号与 CSV 行顺序一致）
- PNG 经 Pillow 以标准 RGB 写出，兼容性优于部分 OpenCV 直写
- 若某张在相册里显示空白/裂图：先确认百度网盘已同步到本机，或改用 `--upscale 2` 重导
