# DeepLearning 2.3 Pipeline

这个子项目用于独立完成 Group Assignment 1 的 2.3 部分。

当前工作流采用 `script-first`：

1. 下载 Kaggle 数据
2. 裁脸并生成处理后数据
3. 固定训练/验证划分
4. 训练第一版预训练分类器
5. 导出 `submission.csv`

常用命令：

```powershell
conda activate gpu_env
python scripts\download_data.py --competition kul-computer-vision-ga-1-2026
python scripts\prepare_faces.py --config configs\experiments\exp_001_resnet18_haar.yaml
python scripts\make_split.py --config configs\experiments\exp_001_resnet18_haar.yaml
python scripts\train.py --config configs\experiments\exp_001_resnet18_haar.yaml
python scripts\predict.py --config configs\experiments\exp_001_resnet18_haar.yaml
python scripts\run_first_submission.py --config configs\experiments\exp_001_resnet18_haar.yaml
```
