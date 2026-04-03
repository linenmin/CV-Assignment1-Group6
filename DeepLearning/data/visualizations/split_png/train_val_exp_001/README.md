# 导出 PNG

- `data\splits\exp_001\train.csv` → 成功 64 / 64 行
- `data\splits\exp_001\val.csv` → 成功 16 / 16 行

- 文件名: `{split}_{行序}_idXXX_classC.png`（行序为该 split 的 CSV 顺序）
- PNG 经 Pillow RGB 写出
- 相册裂图时请确认网盘已同步本地，或 `--upscale 2` 重导
