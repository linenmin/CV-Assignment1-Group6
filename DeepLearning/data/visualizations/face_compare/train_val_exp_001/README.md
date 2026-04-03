# Face Comparison

- `data\splits\exp_001\train.csv` → 导出 64 张
- `data\splits\exp_001\val.csv` → 导出 16 张

- 每张图包含 `raw / HAAR / MTCNN` 三列
- `raw` 为原始 `.npy`，已按比赛数据实际存储格式从 BGR 转成 RGB
- `HAAR` 使用当前主线 `exp_001_faces_224`
- `MTCNN` 使用现有 `exp_008_mtcnn_faces_112`，可直接肉眼比较裁剪稳定性
