# DeepLearning 子项目进度日志

## 2026-03-21

### Session 1

- 读取用户需求，确认目标是为第 2.3 部分建立独立深度学习项目。
- 读取 `Group/AGENTS.md`，确认该目录强调使用 `pi-planning-with-files`。
- 检查仓库状态，确认 `Group` 目录是独立 git 仓库，`DeepLearning` 目录已存在。
- 尝试抽取 `CV_GA1(1).pdf` 文本，发现控制台编码问题，已切换为后续文件中转方案。
- 初始化 `task_plan.md`、`findings.md`、`progress.md`。
- 已完成 PDF 全文提取，并初步定位到 2.3 节所在文本位置。
- 已读完 2.3、3.1、3.2 相关正文，确认 2.3 核心是自由改进 pipeline + 记录完整迭代过程，而不仅仅是追求榜单分数。
- 已确认一个重要设计方向：2.3 若走预训练深度学习路线，更适合做成独立增强子项目，与前两节保持评估可比但不强耦合实现。
- 用户已确认采用 `script-first`，先不处理 notebook，Kaggle 迭代以 `csv` 提交为主。
- 用户已确认协作方式为共享同一 pipeline 后轮流迭代。
- 已创建第一版工程骨架：`configs/`、`src/`、`scripts/`、`tests/`、`README.md`、`requirements.txt`。
- 已按 TDD 先写基础测试，再实现配置合并、分层划分、submission 生成等核心能力。
- 已完成 `train.py`、`predict.py`、`prepare_faces.py`、`make_split.py`、`download_data.py`、`run_first_submission.py`。
- 已在 `gpu_env` 中安装 `lightning` 与 `albumentations`，并确认 `torch 2.5.1` 可用且 CUDA 可见。
- 当前唯一实质性阻塞：本机缺少 `~/.kaggle/kaggle.json`，无法下载比赛数据，因此还不能跑出真实 `submission.csv`。
- 已通过 `KAGGLE_API_TOKEN` 完成 Kaggle 2.0 CLI 认证，并成功下载 `kul-computer-vision-ga-1-2026` 数据。
- 已完成第一版完整闭环：下载 -> 裁脸 -> 划分 -> 训练 -> 预测 -> 导出 `submission.csv`。
- 第一版结果：
  - 实验名：`exp_001_resnet18_haar`
  - 最佳 checkpoint：`outputs/exp_001_resnet18_haar/checkpoints/best-v1.ckpt`
  - 验证集最佳准确率：`0.6944444179534912`
  - submission 文件：`data/submissions/20260321_123354_exp_001_resnet18_haar_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.61178`。

## 2026-04-01

### Session 2

- 已尝试基于 `exp_002_resnet18_haar_10ep` 的最佳 checkpoint 引入推理阶段 `horizontal flip TTA`。
- 本次改动没有修改训练流程，只修改了预测逻辑，因此变量控制较干净。
- 已生成 `exp_004_resnet18_haar_10ep_tta` 的 submission：
  - `data/submissions/20260401_132435_exp_004_resnet18_haar_10ep_tta_submission.csv`
- 该 submission 的 Kaggle public score 为 `0.51266`，明显低于当前基线 `0.61178`。
- 当前结论：`horizontal flip TTA` 不适合作为这条任务线的默认推理策略，后续不再继续沿这个方向扩展更复杂的 TTA 组合。

### Session 3

- 已将策略切回训练增强，并明确把实验变量收缩到单项：关闭训练时的 `HorizontalFlip`。
- 为此新增 `augmentation.use_horizontal_flip` 配置项，并建立实验：
  - `exp_005_resnet18_haar_10ep_nohflip`
- 该实验已经完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_005_resnet18_haar_10ep_nohflip/checkpoints/best.ckpt`
  - 本地验证集最佳准确率：`0.7222222089767456`
  - 新 submission：`data/submissions/20260401_135008_exp_005_resnet18_haar_10ep_nohflip_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.56607`。
- 结论：去掉训练时 `HorizontalFlip` 相比 `exp_004` 的错误 TTA 方向有明显回升，但仍未超过当前最好结果 `0.61178`。

### Session 4

- 已继续沿“训练增强单变量迭代”路线，新增实验：
  - `exp_006_resnet18_haar_10ep_nohflip_affine`
- 本次唯一新增变量是在 `no HorizontalFlip` 的基础上，把几何增强从 `ShiftScaleRotate` 替换为 `Affine`。
- 该实验已经完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_006_resnet18_haar_10ep_nohflip_affine/checkpoints/best.ckpt`
  - 本地验证集最佳准确率：`0.6111111044883728`
  - 新 submission：`data/submissions/20260401_140958_exp_006_resnet18_haar_10ep_nohflip_affine_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.48898`。
- 从本地验证结果和 Kaggle 分数看，`Affine` 替换后明显弱于 `exp_005`，因此当前应把它视为一个明确的负例实验，用于说明几何增强在当前小样本人脸任务上不能加得太激进。

### Session 5

- 已将模型升级方向从通用 `timm` backbone 切到人脸识别专用预训练模型，新增 `CVLFace / AdaFace` 支持。
- 当前采用的具体配置为：
  - 实验名：`exp_007_ir101_adaface_haar_10ep`
  - backbone：`IR101 + AdaFace`
  - 预训练仓库：`minchul/cvlface_adaface_ir101_webface4m`
  - 输入尺寸：`112x112`
  - 归一化：`mean/std = 0.5`
  - 训练策略：冻结 backbone，只训练分类头
- 为接入该模型，本轮工程改动包括：
  - 新增 `cvlface` 模型加载器；
  - 支持 `model.family` / `pretrained_repo_id` / `freeze_backbone`；
  - 支持 `face` 归一化模式；
  - 补充了对应单元测试。
- 该实验已经完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_007_ir101_adaface_haar_10ep/checkpoints/best.ckpt`
  - 本地验证集最佳准确率：`1.0`
  - 新 submission：`data/submissions/20260401_144656_exp_007_ir101_adaface_haar_10ep_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.62720`。
- 由于当前验证集仅有 16 张，本地 `1.0` 不能直接视为真实泛化结论；但 `0.62720` 已经超过此前最好结果 `0.61178`，说明“人脸识别专用预训练 backbone” 这条主线是有效的。

### Session 6

- 已沿 `exp_007` 的主线继续做单变量升级，把预处理从 `HAAR` 裁框切到 `MTCNN + 5 点对齐`。
- 当前实验配置为：
  - 实验名：`exp_008_ir101_adaface_mtcnn_10ep`
  - backbone：`IR101 + AdaFace`
  - 训练策略：继续冻结 backbone，只训练分类头
  - 输入尺寸：`112x112`
  - 预处理：`MTCNN` 检测 + 5 点 landmark 仿射对齐
- 本轮新增工程能力：
  - 新增 `MTCNNFaceDetector`
  - 支持 ArcFace 风格的 5 点参考模板对齐
  - 预处理构建器支持 `detector=mtcnn`
  - 增加 `mtcnn` 相关单元测试
- 该实验已经完成预处理、训练与预测：
  - 最佳 checkpoint：`outputs/exp_008_ir101_adaface_mtcnn_10ep/checkpoints/best.ckpt`
  - 本地验证集最佳准确率：`1.0`
  - 新 submission：`data/submissions/20260401_151920_exp_008_ir101_adaface_mtcnn_10ep_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.60627`。
- 结论：
  - `MTCNN + 5 点对齐` 没有超过 `exp_007` 的 `0.62720`；
  - 它也没有明显劣化到失效，而是回到了接近早期 `resnet18` 基线的区间；
  - 这说明当前主瓶颈已经不再是“检测器够不够新”，而更可能是任务建模方式本身。

### Session 7

- 已在 `exp_007` 主线上继续做低风险微调升级，新增实验：
  - `exp_009_ir101_adaface_haar_10ep_laststage_ft`
- 本次唯一核心变量是训练策略：
  - 继续使用 `IR101 + AdaFace`
  - 继续使用 `HAAR`
  - 继续使用 `112x112`
  - 从“冻结 backbone，只训练分类头”升级为“仅解冻最后一个 stage，并为 backbone 使用更小学习率”
- 为支持该实验，本轮工程改动包括：
  - `cvlface` 分类器支持只解冻最后一个 stage
  - 优化器支持 backbone 与分类头使用不同学习率
  - 新增对应单元测试
- 训练过程中发现部分微调对资源更敏感：
  - 初始 `batch_size=16` 会导致本机资源压力过高
  - 调整为 `batch_size=8` 后训练稳定完成
- 该实验已经完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_009_ir101_adaface_haar_10ep_laststage_ft/checkpoints/best.ckpt`
  - 本地验证集最佳准确率：`0.9444444179534912`
  - 新 submission：`data/submissions/20260401_164220_exp_009_ir101_adaface_haar_10ep_laststage_ft_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.78799`。
- 结论：
  - 这次从 `0.62720` 提升到 `0.78799`，说明“人脸专用预训练 backbone + 轻量部分微调”是当前最有效的主线；
  - 提升的关键不是继续换检测器，而是让强预训练人脸 backbone 对比赛分布做适度适配；
  - 这一结果也说明，开发阶段仍应保留验证集作为护栏，但真正决定方向的仍然是 Kaggle 分数。

### Session 8

- 已在 `exp_009` 主线上继续做单变量细调，新增实验：
  - `exp_010_ir101_adaface_haar_10ep_laststage_ft_lr1e5`
- 本次唯一核心变量是把 backbone 学习率从 `3e-5` 继续降低到 `1e-5`：
  - 继续使用 `IR101 + AdaFace`
  - 继续使用 `HAAR`
  - 继续使用 `112x112`
  - 继续只解冻最后一个 stage
  - 分类头学习率保持 `3e-4`
- 该实验已经完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_010_ir101_adaface_haar_10ep_laststage_ft_lr1e5/checkpoints/best.ckpt`
  - 本地验证集最佳准确率：`1.0`
  - 新 submission：`data/submissions/20260401_170932_exp_010_ir101_adaface_haar_10ep_laststage_ft_lr1e5_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.87885`。
- 结论：
  - 从 `0.78799` 提升到 `0.87885`，说明 backbone 微调幅度进一步收敛后，测试集表现继续明显改善；
  - 当前最优方向已经非常明确：不是“更强更猛”的微调，而是“更温和、更受控”的微调；
  - 下一步应继续围绕这条主线优化模型选择标准，而不是回到 detector、增强或更大 backbone 的大改路线。
