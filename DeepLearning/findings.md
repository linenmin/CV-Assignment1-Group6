# DeepLearning 子项目发现记录

## 2026-03-21

### 仓库与目录现状

- `Computer Vision/assignment/Group` 是一个独立 git 仓库。
- `DeepLearning` 子目录已经存在，可以作为 2.3 部分的独立工作区。
- 当前仓库还没有提交记录，分支基线较干净，但整个仓库处于未提交状态。

### 流程与工具约束

- 当前子目录 `AGENTS.md` 明确可用的核心技能是 `pi-planning-with-files`。
- 用户希望通过该方式记录系统如何逐步迭代。

### 文档读取情况

- `CV_GA1(1).pdf` 已确认可读取，共 7 页。
- 终端编码导致直接输出 PDF 文本失败，不影响后续通过 UTF-8 文件提取关键内容。
- 已将 PDF 文本提取到 `DeepLearning/_cv_ga1_extracted.txt`。
- 在提取文本中已定位到 `2.3`，并确认文档正文包含 `deep learning`、`neural network`、`submission` 等关键词，说明用户对 2.3 适合独立拆分为深度学习子项目的判断是有依据的。

### 2.3 章节关键要求

- 2.3 的正式标题是 `Improve performance`。
- 该部分允许改进整个 pipeline 中任何最有影响力的部分：`face detection`、`feature extraction`、`classification model`。
- 文档明确写明 `There are no rules for this section, you can try anything you please`，但同时强调 `data and compute are limited`。
- 最关键的交付要求之一不是单次最好分数，而是要在最终提交 notebook 中清楚记录 `zero-to-hero journey`，也就是每轮迭代做了什么、有没有带来提升。
- 如果实验很多，最终 Kaggle 提交 notebook 可以只运行最后最佳 pipeline，较早实验可以用文字和离线图表总结。
- Kaggle 排名只占总成绩的小部分，更重要的是 `ingenuity`、`methodological correctness` 和 `insight`。

### 当前设计判断

- 用户认为 2.3 若采用预训练神经网络/深度学习主线，则更合理的定位是“独立增强线”，而不是继续与 2.1/2.2 的传统 baseline 共用大量实现。
- 这个判断与文档要求一致：2.3 需要的是对已有 pipeline 的提升与对比，不要求和 baseline 在实现层面强耦合。
- 因此后续工程结构应优先保证：
  1. 能独立训练/验证/推理；
  2. 能与 baseline 在相同数据划分和指标上做对比；
  3. 能把实验迭代过程清楚汇总进最终 Kaggle notebook。

### 已确认的项目管理策略

- 开发模式选 `script-first`，暂不设计 notebook 优先的工作流。
- Kaggle 日常迭代提交采用 `csv` 提交思路，最终 notebook 等方案收敛后再补。
- 项目协作不是固定模块分工，而是在同一条 pipeline 上轮流迭代。
- 因此工程文件需要强调：
  1. 命令行入口统一；
  2. 配置和实验记录可追踪；
  3. 同一条 pipeline 易于交接和复现。

### 提交与工程约束

- 最终评分依据是 Kaggle 上截止前最后一次提交的 notebook。
- 只接受 Kaggle 提交，不接受本地额外提交。
- 最终 notebook 必须自包含、可运行、文档充分，能够让助教看懂设计选择与实验洞察。
- 文档明确建议大量训练与实验可以放在离线环境完成，最终 notebook 只保留总结、必要图表和最终预测流程。

### 当前实现状态

- 第一版 pipeline 已按 `Lightning + timm + yaml` 落成基础骨架。
- 当前默认首版实验是 `exp_001_resnet18_haar`：
  - 预处理：HAAR 裁脸，失败时中心裁剪兜底；
  - 模型：`resnet18` 预训练；
  - 训练：Lightning + `AdamW` + `CosineAnnealingLR`；
  - 输出：最佳 checkpoint、metrics、实验 registry、规范命名的 `submission.csv`。
- 已增加一键入口 `scripts/run_first_submission.py`，用于顺序执行下载、裁脸、划分、训练、预测。
- 当前最大的非代码风险不是实现，而是 Kaggle 凭证缺失导致无法拿到真实比赛数据。

### 第一版真实运行结果

- 已在 `gpu_env` 中跑通真实比赛数据。
- 第一版验证集最佳准确率为 `0.6944444179534912`。
- 第一版 Kaggle public score 为 `0.61178`。
- 第一版 submission 已生成：
  - `data/submissions/20260321_123354_exp_001_resnet18_haar_submission.csv`
- 训练与预测登记已经追加到：
  - `reports/experiments/registry.csv`

### 运行中暴露的问题

- `class` 列在 `itertuples()` 中会因关键字改名，导致预处理元数据最初丢失标签；已通过测试修复为 `to_dict(orient="records")` 流程。
- Windows `gbk` 控制台与 Lightning 默认 `rich` 进度条不兼容；已关闭 rich progress/model summary，改为稳定模式运行。
