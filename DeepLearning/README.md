# DeepLearning 子项目说明

这个文件夹用于独立完成 `CV_GA1(1).pdf` 中的 `2.3 Improve performance` 部分。

当前项目是一条完全独立的深度学习流水线：

`Kaggle 数据下载 -> 裁脸 -> 固定划分 -> 训练 -> 预测 -> 生成 submission.csv -> 手动上传到 Kaggle`

当前已经跑通的第一版实验：

- 实验名：`exp_001_resnet18_haar`
- 预处理：`HAAR` 裁脸，检测失败时自动使用中心裁剪兜底
- 模型：`timm` 的 `resnet18` 预训练模型
- 框架：`PyTorch Lightning`
- 第一版验证集最佳准确率：`0.6944444179534912`
- 第一版 Kaggle public score：`0.61178`

## 1. 从 Git 仓库开始

项目仓库：

```bash
git@github.com:linenmin/CV-Assignment1-Group6.git
```

如果你的队友是第一次在自己电脑上开始，推荐从下面这套流程开始。

### 1.1 克隆仓库

```bash
git clone git@github.com:linenmin/CV-Assignment1-Group6.git
cd CV-Assignment1-Group6
```

### 1.2 切换到 `DL` 分支

先更新远端信息：

```bash
git fetch origin
```

然后切到我们当前工作的分支：

```bash
git checkout DL
```

如果本地还没有这个分支，可以用：

```bash
git checkout -b DL origin/DL
```

### 1.3 进入 DeepLearning 子目录

从仓库根目录进入：

```bash
cd "Computer Vision/assignment/Group/DeepLearning"
```

## 2. 目录结构

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
├─ requirement.txt           # requirements.txt 的兼容入口
└─ README.md                 # 本说明
```

## 3. 创建你自己的 conda 环境

不要直接依赖别人的环境名。每个人在自己机器上创建一个新的环境即可。

例如：

```bash
conda create -n cvga1_dl python=3.12 -y
conda activate cvga1_dl
```

环境名 `cvga1_dl` 只是一个示例，你也可以换成自己喜欢的名字，比如：

- `cv_dl`
- `ga1_dl`
- `cv_group6`

重点不是名字，而是环境要独立。

## 4. 安装依赖

激活环境后，在 `DeepLearning` 目录下执行：

```bash
pip install -r requirements.txt
```

如果你的队友习惯用单数文件名，也可以执行：

```bash
pip install -r requirement.txt
```

说明：

- `torch / torchvision` 最好和自己机器的 CUDA 环境匹配。
- 如果你的机器没有 GPU，也能跑，但训练会更慢。

## 5. Kaggle 认证

### 5.1 推荐方式

当前项目使用 Kaggle 2.0 CLI，可以通过环境变量 `KAGGLE_API_TOKEN` 工作。

在当前终端临时设置：

```bash
export KAGGLE_API_TOKEN="你的 token"
```

如果是在 Windows `cmd`：

```cmd
set KAGGLE_API_TOKEN=你的token
```

如果是在 PowerShell：

```powershell
$env:KAGGLE_API_TOKEN = "你的 token"
```

### 5.2 注意

- 不要把 token 写进仓库文件。
- 不要把认证文件提交到 git。

## 6. 一次完整运行怎么做

### 6.1 最简单的方式：一键跑完整流程

在 `DeepLearning` 目录下：

```bash
conda activate cvga1_dl
export PYTHONPATH=src
python scripts/run_first_submission.py --config configs/experiments/exp_001_resnet18_haar.yaml
```

这个命令会顺序执行：

1. 下载 Kaggle 数据
2. 裁脸
3. 生成训练/验证划分
4. 训练模型
5. 生成 `submission.csv`

### 6.2 如果数据已经下载过

可以跳过下载：

```bash
python scripts/run_first_submission.py --config configs/experiments/exp_001_resnet18_haar.yaml --skip-download
```

## 7. 分步骤运行

如果不想一次全跑，也可以按步骤执行。

### 7.1 下载数据

```bash
python scripts/download_data.py --competition kul-computer-vision-ga-1-2026
```

下载后的默认目录：

```text
data/raw/kul-computer-vision-ga-1-2026/
```

### 7.2 裁脸并生成处理后图像

```bash
python scripts/prepare_faces.py --config configs/experiments/exp_001_resnet18_haar.yaml
```

输出目录：

```text
data/processed/exp_001_faces_224/
```

关键文件：

- `train_metadata.csv`
- `test_metadata.csv`

### 7.3 生成固定训练/验证划分

```bash
python scripts/make_split.py --config configs/experiments/exp_001_resnet18_haar.yaml
```

输出目录：

```text
data/splits/exp_001/
```

关键文件：

- `train.csv`
- `val.csv`
- `test.csv`

### 7.4 训练模型

```bash
python scripts/train.py --config configs/experiments/exp_001_resnet18_haar.yaml
```

输出目录：

```text
outputs/exp_001_resnet18_haar/
```

关键文件：

- `checkpoints/best-v1.ckpt`
- `metrics.json`
- `logs/`

### 7.5 生成 submission.csv

```bash
python scripts/predict.py --config configs/experiments/exp_001_resnet18_haar.yaml
```

输出目录：

```text
data/submissions/
```

## 8. 生成 submission 之后怎么做

我们当前的约定是：**手动上传到 Kaggle**，不在项目中自动提交。

步骤如下：

1. 找到 `data/submissions/` 里的最新 `submission.csv`
2. 打开 Kaggle 比赛页面
3. 手动上传这个文件
4. 记录返回的 public score
5. 把这个分数写回 `reports/experiments/registry.csv`

## 9. 如何开始做下一轮实验

推荐方式不是直接改 `exp_001`，而是复制一份新配置。

例如：

1. 复制：
   - `configs/experiments/exp_001_resnet18_haar.yaml`
2. 改成：
   - `configs/experiments/exp_002_xxx.yaml`
3. 只改一个主变量：
   - backbone
   - batch size
   - face size
   - 数据增强
   - 学习率
4. 重新跑：
   - `prepare_faces`（如果预处理变了）
   - `make_split`
   - `train`
   - `predict`

这样实验之间更容易比较，也更适合两个人轮流迭代。

## 10. 当前优先推荐的优化方向

第一版是 baseline，不是最终形态。后续优先级建议：

1. 把 `HAAR` 检测器换成更稳的人脸检测器
2. 把 `resnet18` 换成更强的 backbone
3. 调整数据增强强度
4. 试 `batch size / lr / epoch`
5. 看是否需要 class weighting 或 sampler
6. 做 TTA 或简单 ensemble

## 11. 实验记录怎么维护

每次有新实验，都要更新：

- `reports/experiments/registry.csv`
- `progress.md`
- `findings.md`

最低限度要记：

- 配置文件名
- 改了什么
- 验证集结果
- Kaggle public score
- 结论

## 12. 常见问题

### 12.1 `make_split.py` 报 `class` 列不存在

先确认你有没有重新跑过：

```bash
python scripts/prepare_faces.py --config configs/experiments/exp_001_resnet18_haar.yaml
```

因为 `train_metadata.csv` 是预处理阶段生成的。

### 12.2 Windows 终端乱码或 Lightning rich progress 报错

项目里已经在训练和预测脚本中关闭了 rich progress bar。
如果你又改回去，Windows 控制台可能再次报编码错误。

### 12.3 数据下载失败

先检查：

- `KAGGLE_API_TOKEN` 是否设置
- 是否已经加入比赛
- Kaggle 账号是否完成验证

## 13. 当前状态

当前第一版完整实现已经在 `DL` 分支中。

如果只是想复现实验，不需要重新搭项目骨架，直接按本 README 的步骤执行即可。
