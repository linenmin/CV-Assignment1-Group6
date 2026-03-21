# DeepLearning 子项目说明

这个文件夹用于独立完成 `CV_GA1(1).pdf` 第 `2.3 Improve performance` 部分。

我们当前采用的是一条完全独立的深度学习流水线：

`Kaggle 数据下载 -> 裁脸 -> 固定划分 -> 训练 -> 预测 -> 生成 submission.csv -> 手动上传到 Kaggle`

当前已经跑通的第一版实验是：

- 实验名：`exp_001_resnet18_haar`
- 预处理：`HAAR` 裁脸，检测失败时自动使用中心裁剪兜底
- 模型：`timm` 的 `resnet18` 预训练模型
- 框架：`PyTorch Lightning`
- 第一版验证集最佳准确率：`0.6944444179534912`
- 第一版 Kaggle public score：`0.61178`

## 1. 目录结构

```text
DeepLearning/
├─ configs/                  # 实验配置
├─ data/                     # 原始数据、处理后数据、划分文件、submission
├─ outputs/                  # checkpoint、训练日志、metrics
├─ reports/                  # 实验登记表
├─ scripts/                  # 命令行入口
├─ src/dl_pipeline/          # 真实业务代码
├─ tests/                    # 基础单元测试
├─ task_plan.md              # 项目计划
├─ findings.md               # 发现与结论
├─ progress.md               # 进度日志
├─ requirements.txt          # Python 依赖
└─ README.md                 # 本说明
```

## 2. 你最需要知道的几个文件

如果你是第一次接手这个项目，优先看下面这些：

- [configs/experiments/exp_001_resnet18_haar.yaml](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\configs\experiments\exp_001_resnet18_haar.yaml)
  - 当前第一版实验的配置文件
- [scripts/run_first_submission.py](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\scripts\run_first_submission.py)
  - 一键顺序执行整条流程
- [scripts/train.py](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\scripts\train.py)
  - 训练入口
- [scripts/predict.py](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\scripts\predict.py)
  - 生成 submission 的入口
- [reports/experiments/registry.csv](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\reports\experiments\registry.csv)
  - 每轮实验的登记表

## 3. 环境准备

### 3.1 推荐环境

默认使用你本机已有的 conda 环境：

```powershell
conda activate gpu_env
```

### 3.2 安装依赖

如果你的队友也要在自己的环境中跑，可以在激活环境后执行：

```powershell
pip install -r requirements.txt
```

说明：

- `torch / torchvision` 最好和本机 CUDA 版本匹配。
- 当前我们已经在 `gpu_env` 中验证过这个项目能跑。

## 4. Kaggle 认证

### 4.1 推荐方式

当前项目使用 Kaggle 2.0 CLI，可以通过环境变量 `KAGGLE_API_TOKEN` 工作。

如果你只是在自己机器上运行，推荐在当前终端临时设置：

```powershell
$env:KAGGLE_API_TOKEN = "你的 token"
```

或者使用 `cmd` 风格：

```cmd
set KAGGLE_API_TOKEN=你的token
```

### 4.2 注意

- 不要把 token 写进仓库文件。
- 不要把 `kaggle.json` 提交到 git。

## 5. 一次完整运行怎么做

### 5.1 最简单的方式：一键跑完整流程

在 `DeepLearning` 目录下：

```powershell
conda activate gpu_env
$env:KAGGLE_API_TOKEN = "你的 token"
cd "d:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group\DeepLearning"
$env:PYTHONPATH = "src"
python scripts\run_first_submission.py --config configs\experiments\exp_001_resnet18_haar.yaml
```

这个命令会顺序执行：

1. 下载 Kaggle 数据
2. 裁脸
3. 生成训练/验证划分
4. 训练模型
5. 生成 submission.csv

### 5.2 如果数据已经下载过

可以跳过下载：

```powershell
python scripts\run_first_submission.py --config configs\experiments\exp_001_resnet18_haar.yaml --skip-download
```

## 6. 分步骤运行

如果你不想一次全跑，也可以按步骤执行。

### 6.1 下载数据

```powershell
python scripts\download_data.py --competition kul-computer-vision-ga-1-2026
```

下载后的默认目录：

- `data/raw/kul-computer-vision-ga-1-2026/`

### 6.2 裁脸并生成处理后图像

```powershell
python scripts\prepare_faces.py --config configs\experiments\exp_001_resnet18_haar.yaml
```

输出目录：

- `data/processed/exp_001_faces_224/`

关键文件：

- `train_metadata.csv`
- `test_metadata.csv`

### 6.3 生成固定训练/验证划分

```powershell
python scripts\make_split.py --config configs\experiments\exp_001_resnet18_haar.yaml
```

输出目录：

- `data/splits/exp_001/`

关键文件：

- `train.csv`
- `val.csv`
- `test.csv`

### 6.4 训练模型

```powershell
python scripts\train.py --config configs\experiments\exp_001_resnet18_haar.yaml
```

输出目录：

- `outputs/exp_001_resnet18_haar/`

关键文件：

- `checkpoints/best-v1.ckpt`
- `metrics.json`
- `logs/`

### 6.5 生成 submission.csv

```powershell
python scripts\predict.py --config configs\experiments\exp_001_resnet18_haar.yaml
```

输出目录：

- `data/submissions/`

当前第一版生成文件：

- [20260321_123354_exp_001_resnet18_haar_submission.csv](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\data\submissions\20260321_123354_exp_001_resnet18_haar_submission.csv)

## 7. 生成了 submission 之后怎么做

我们当前的约定是：**手动上传到 Kaggle**，不在项目中自动提交。

步骤如下：

1. 找到 `data/submissions/` 里的最新 `submission.csv`
2. 打开 Kaggle 比赛页面
3. 手动上传这个文件
4. 记录返回的 public score
5. 把这个分数写回 `reports/experiments/registry.csv`

## 8. 如何开始做下一轮实验

推荐方式不是直接改 `exp_001`，而是复制一份新配置。

例如：

1. 复制
   - `configs/experiments/exp_001_resnet18_haar.yaml`
2. 改成
   - `configs/experiments/exp_002_xxx.yaml`
3. 只改一个主变量
   - 例如 backbone
   - 或 batch size
   - 或 face size
   - 或数据增强
4. 重新跑
   - `prepare_faces`（如果预处理变了）
   - `make_split`（通常不用重复）
   - `train`
   - `predict`

这样做的好处是实验之间更容易比较。

## 9. 当前最适合继续优化的方向

第一版是一个能跑通的 baseline，不是最终形态。后续优先级建议：

1. 把 `HAAR` 检测器换成更稳的人脸检测器
2. 把 `resnet18` 换成更强的 backbone
3. 调整数据增强强度
4. 试 `batch size / lr / epoch`
5. 看是否需要 class weighting 或 sampler
6. 做 TTA 或简单 ensemble

## 10. 实验记录怎么维护

每次有新实验，都要更新：

- [reports/experiments/registry.csv](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\reports\experiments\registry.csv)
- [progress.md](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\progress.md)
- [findings.md](d:\BaiduNetdiskWorkspace\Leuven\8th\Computer%20Vision\assignment\Group\DeepLearning\findings.md)

最低限度要记：

- 配置文件名
- 改了什么
- 验证集结果
- Kaggle public score
- 结论

## 11. 常见问题

### 11.1 `make_split.py` 报 `class` 列不存在

先确认你有没有重新跑过：

```powershell
python scripts\prepare_faces.py --config configs\experiments\exp_001_resnet18_haar.yaml
```

因为 `train_metadata.csv` 是预处理阶段生成的。

### 11.2 Windows 终端乱码或 Lightning rich progress 报错

项目里已经在训练和预测脚本中关闭了 rich progress bar。
如果你又改回去，Windows `gbk` 终端可能再次报编码错误。

### 11.3 数据下载失败

先检查：

- `KAGGLE_API_TOKEN` 是否设置
- 是否已经加入比赛
- Kaggle 账号是否完成验证

## 12. 当前提交记录

当前第一版完整实现已经在 `DL` 分支提交。

如果你只是想复现实验，不需要重新搭项目骨架，直接按本 README 运行即可。
