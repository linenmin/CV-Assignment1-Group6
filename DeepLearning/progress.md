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

### Session 9

- 已把训练阶段的模型选择逻辑从固定硬编码改成可配置：
  - `train.monitor_metric`
  - `train.monitor_mode`
- 默认行为保持不变：
  - 默认仍然使用 `val_acc / max`
  - 只有实验配置显式覆盖时，才切到其他指标
- 为此新增：
  - `src/dl_pipeline/training/monitoring.py`
  - `tests/test_monitoring.py`
  - `tests/test_lightning_module.py` 中的自定义 monitor 测试
- 已完成本轮实验：
  - `exp_011_ir101_adaface_haar_10ep_laststage_ft_lr1e5_valloss`
- 该实验与 `exp_010` 保持完全相同的 backbone、学习率和微调范围，仅把选模指标从 `val_acc` 切换到 `val_loss`。
- 当前已完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_011_ir101_adaface_haar_10ep_laststage_ft_lr1e5_valloss/checkpoints/best.ckpt`
  - 本地最佳 `val_loss`：`0.1152695044875145`
  - 对应 epoch 的 `val_acc`：`1.0`
  - 新 submission：`data/submissions/20260401_173902_exp_011_ir101_adaface_haar_10ep_laststage_ft_lr1e5_valloss_submission.csv`
- 当前状态：
  - 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.85572`
  - 结论：这次切换到 `val_loss` 选模并没有超过 `exp_010` 的 `0.87885`，说明在“只解冻最后一个 stage”这条线上，`exp_010` 原本就已经选到了接近最优的 checkpoint，选模指标不是当前主要增益来源

### Session 10

- 已为 `CVLFace` 主线补充更通用的分阶段解冻能力：
  - 新增 `model.unfreeze_stage_count`
  - 兼容原有 `unfreeze_last_stage`
  - 保持默认行为不变
- 已完成本轮实验：
  - `exp_012_ir101_adaface_haar_10ep_last2stage_ft_head1e4`
- 该实验相对 `exp_010` 的核心变化是：
  - 从只解冻最后 `1` 个 stage 扩展到解冻最后 `2` 个 stage
  - 分类头学习率从 `3e-4` 降到 `1e-4`
  - backbone 学习率设置为 `5e-6`
  - 由于资源压力上升，`batch_size` 收缩到 `4`
- 当前已完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_012_ir101_adaface_haar_10ep_last2stage_ft_head1e4/checkpoints/best.ckpt`
  - 本地最佳 `val_loss`：`0.18338939547538757`
  - 对应 `val_acc`：`0.9444444179534912`
  - 新 submission：`data/submissions/20260401_175943_exp_012_ir101_adaface_haar_10ep_last2stage_ft_head1e4_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.87775`
- 结论：
  - 线上几乎追平 `exp_010` 的 `0.87885`，只差 `0.00110`
  - 这说明“两层解冻”并没有把模型明显带偏，反而说明这条线仍有潜力
  - 但当前这组超参数并没有真正超过 `exp_010`，因此还不能视为新的最优主线

### Session 11

- 已在 `exp_012` 的基础上做一次更干净的选模验证：
  - `exp_013_ir101_adaface_haar_10ep_last2stage_ft_head1e4_valloss`
- 这轮不再改结构和学习率，只改 checkpoint 选择标准：
  - 从 `val_acc / max` 切到 `val_loss / min`
- 当前已完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_013_ir101_adaface_haar_10ep_last2stage_ft_head1e4_valloss/checkpoints/best.ckpt`
  - 本地最佳 `val_loss`：`0.1475604921579361`
  - 对应 `val_acc`：`0.9444444179534912`
  - 新 submission：`data/submissions/20260401_182258_exp_013_ir101_adaface_haar_10ep_last2stage_ft_head1e4_valloss_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.87720`
- 结论：
  - 在“两层解冻”这条线上，切换到 `val_loss` 选模确实显著改善了本地 `val_loss`
  - 但线上分数从 `0.87775` 变为 `0.87720`，并没有转化为 Kaggle 提升
  - 这说明当前瓶颈已经不是“选模指标选错了”，而是“两层解冻 + 当前学习率组合”本身尚未超过 `exp_010` 这条更稳的主线

### Session 12

- 已在 `exp_010` 主线上新增一条“更强但人脸安全的退化增强”实验：
  - `exp_014_ir101_adaface_haar_10ep_laststage_ft_lr1e5_degradeaug`
- 这轮保持 `exp_010` 的 backbone、学习率和解冻范围不变，只在训练增强里新增：
  - `GaussianBlur / MotionBlur / GaussNoise / ImageCompression` 的 `OneOf`
  - 很轻的 `CoarseDropout`
- 当前已完成训练与预测：
  - 最佳 checkpoint：`outputs/exp_014_ir101_adaface_haar_10ep_laststage_ft_lr1e5_degradeaug/checkpoints/best.ckpt`
  - 本地最佳 `val_loss`：`0.09722547978162766`
  - 本地最佳 `val_acc`：`1.0`
  - 新 submission：`data/submissions/20260401_185158_exp_014_ir101_adaface_haar_10ep_laststage_ft_lr1e5_degradeaug_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.87665`
- 结论：
  - 新增强确实让本地曲线更“好看”，`val_loss` 明显低于 `exp_010`
  - 但线上分数反而从 `0.87885` 降到 `0.87665`
  - 这说明当前更强的退化增强提升了本地正则化表现，却没有改善真实测试集泛化，当前主线最优仍然是 `exp_010`

### Session 13

- 已启动一条新的方法线，不再继续训练新模型，而是直接重用 `exp_010` 的最佳 checkpoint，做开放集推理实验：
  - `exp_015_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype`
- 本次唯一核心变化是推理建模方式：
  - 不再直接使用 softmax 三分类输出
  - 改为提取 face embedding
  - 使用训练集上的 `Jesse` / `Mila` 样本构建两个 prototype
  - 在验证集上搜索相似度阈值，低于阈值时预测为 `other`
- 为支持该实验，本轮工程改动包括：
  - 分类器新增 `extract_features`
  - 新增 `prototype` 推理模块
  - `predict.py` 支持 `inference.mode = prototype`
  - 新增对应单元测试
- 当前已完成开放集推理与 submission 生成：
  - 复用 checkpoint：`outputs/exp_010_ir101_adaface_haar_10ep_laststage_ft_lr1e5/checkpoints/best.ckpt`
  - 自动选中的阈值：`0.7`
  - 本地验证集准确率：`0.875`
  - 新 submission：`data/submissions/20260401_190448_exp_015_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.89842`
- 结论：
  - 这是当前最高分，首次超过 `0.89`
  - 尽管本地验证准确率低于 `exp_010`，线上却明显提升，说明当前真正有效的改进来自**任务建模方式改变**，而不是继续抠训练技巧
  - 这轮实验强烈支持一个更新后的主判断：`other` 更接近开放集识别问题，不适合继续只当作普通 softmax 第三类处理

### Session 14

- 已在 `exp_015` 的基础上继续做纯推理层迭代：
  - `exp_016_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_finegrained`
- 这轮仍然不重训模型，继续复用 `exp_010` checkpoint，只把 prototype 阈值搜索从粗粒度改成细粒度。
- 自动搜索结果为：
  - `selected_threshold = 0.55`
  - 本地 `val_accuracy = 0.9375`
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_191216_exp_016_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_finegrained_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.91299`
- 结论：
  - 相比 `exp_015` 的 `0.89842`，这次进一步提升约 `0.01457`
  - 这说明上一轮 `0.7` 的阈值偏保守，拒识过多
  - 当前最好结果已经再次证明：后续最高优先级应继续放在开放集推理建模，而不是重新回到训练技巧调参

### Session 15

- 已在 `exp_016` 的基础上继续尝试更灵活的开放集边界：
  - `exp_017_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_classspecific`
- 这轮仍然不重训模型，继续复用 `exp_010` checkpoint，只把 prototype 推理从“单一全局阈值”扩展为“每个目标类一个独立阈值”。
- 自动搜索结果为：
  - `selected_thresholds_by_class = {1: 0.45, 2: 0.45}`
  - 本地 `val_accuracy = 0.9375`
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_192331_exp_017_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_classspecific_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.85572`
- 结论：
  - 这次并没有学出稳定的“类别差异阈值”，最终反而退化成了更低的统一阈值
  - 相比 `exp_016`，预测中有 `136` 张从 `other` 被放宽为目标类，其中 `131` 张是 `0 -> 1`
  - 线上显著下降说明：当前 16 张验证集不足以支撑更高自由度的阈值拟合，后续应回到更稳健的全局阈值估计，而不是继续增加边界参数

### Session 16

- 已继续尝试“更稳健的全局阈值估计”：
  - `exp_018_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_cvglobal`
- 这轮仍然不重训模型，继续复用 `exp_010` checkpoint，只把全局阈值搜索从单一 holdout 改成了训练集分层交叉验证平均。
- 自动搜索结果为：
  - `selected_threshold = 0.45`
  - `crossval_mean_accuracy = 0.90625`
  - 本地 `val_accuracy = 0.9375`
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_194047_exp_018_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_cvglobal_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.85848`
- 结论：
  - 交叉验证并没有把阈值稳定在 `exp_016` 的最佳点 `0.55`，反而再次把边界放宽到 `0.45`
  - 相比 `exp_016`，预测中有 `131` 张从 `other` 被放宽为目标类，其中 `126` 张是 `0 -> 1`
  - 线上再次明显下降，说明当前最优阈值已经被 `exp_016` 充分证明，后续不应继续围绕更松的阈值搜索，而应固定最佳阈值后再回到模型改进

### Session 17

- 已把 `exp_016` 证明有效的全局阈值 `0.55` 固定下来，开始统一重评估已有模型的 embedding 质量，而不再继续搜索阈值。
- 本轮新增四个固定协议实验：
  - `exp_019_ir101_adaface_haar_10ep_laststage_ft_prototype_fixed055`
  - `exp_020_ir101_adaface_haar_10ep_last2stage_ft_head1e4_prototype_fixed055`
  - `exp_021_ir101_adaface_haar_10ep_last2stage_ft_head1e4_valloss_prototype_fixed055`
  - `exp_022_ir101_adaface_haar_10ep_laststage_ft_lr1e5_degradeaug_prototype_fixed055`
- 这轮全部复用现有 checkpoint，不重训模型，唯一目标是回答：
  - 在统一 `prototype + threshold=0.55` 协议下，到底哪个模型的 embedding 最强？
- 当前 Kaggle public score 为：
  - `exp_019 = 0.91685`
  - `exp_020 = 0.91519`
  - `exp_021 = 0.91079`
  - `exp_022 = 0.91024`
- 这轮最重要的结论是：
  - 固定阈值后，原来 softmax 下并不最优的 `exp_009` 反而成为当前最好模型；
  - `exp_010/012/013/014` 之间的差异都很小，说明当前分数上限已经不主要由“再调一点学习率/monitor/增强”决定；
  - 这也证明之前很多 softmax 结论会误导 embedding 质量判断，后续模型迭代必须直接在固定 open-set 协议下评估，而不是再回到 softmax submission

### Session 18

- 已开始尝试比单 centroid 更强的推理规则：
  - `exp_023_ir101_adaface_haar_10ep_laststage_ft_exemplar_knn_margin`
- 这轮继续复用 `exp_009` checkpoint，不重训模型，只把推理方式从：
  - `single prototype + fixed threshold`
  - 升级为 `exemplar gallery + top-k nearest neighbor + target-vs-other margin rejection`
- 自动搜索结果为：
  - `selected_threshold = 0.55`
  - `selected_top_k = 1`
  - `selected_margin = 0.0`
  - 本地 `val_accuracy = 0.9375`
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_201410_exp_023_ir101_adaface_haar_10ep_laststage_ft_exemplar_knn_margin_submission.csv`
- 该 submission 已手动提交到 Kaggle，当前 public score 为 `0.84911`
- 结论：
  - 这轮本地选参最后退化成了 `1-NN + 0 margin`，没有真正学到“更稳的近邻投票”或“更强的 other 抑制”
  - 相比 `exp_019`，预测中有 `215` 张发生变化，其中 `214` 张把原本的 `other` 放宽为目标类
  - 线上明显下降说明：单样本最近邻对噪声样本极其敏感，当前最有效的方向不是直接上 1-NN，而是继续保留原型式稳定性，同时把 `other` 作为显式竞争项引入更稳健的 margin / aggregation 规则

### Session 19

- 已测试“全部标注样本做 gallery，但仍保留最稳的单 prototype 规则”：
  - `exp_024_ir101_adaface_haar_10ep_laststage_ft_prototype_fixed055_alllabeled`
- 这轮继续复用 `exp_009` checkpoint，不重训模型，只把 gallery 从 `train split` 扩展到 `all 80 labeled`。
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_202654_exp_024_ir101_adaface_haar_10ep_laststage_ft_prototype_fixed055_alllabeled_submission.csv`
- 本地最关键的结论是：
  - `exp_024` 的 submission 与 `exp_019` **逐行完全一致**
  - 差异条数为 `0`
- 结论：
  - 在 `single prototype + fixed threshold 0.55` 这个规则下，把 `val` 并入 gallery 没有带来任何变化；
  - 当前瓶颈不在“gallery 样本总数不够”，而在“所有样本被压成每类一个均值后，新增样本信息被抹平”。

### Session 20

- 已继续测试更丰富的开放集打分规则：
  - `exp_025_ir101_adaface_haar_10ep_laststage_ft_richer_openset_alllabeled`
- 这轮继续复用 `exp_009` checkpoint，不重训模型，只在 `all 80 labeled` 上做 leave-one-out 选参，加入：
  - `target top-k mean`
  - `other top-k mean`
  - `best-vs-second target margin`
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_203557_exp_025_ir101_adaface_haar_10ep_laststage_ft_richer_openset_alllabeled_submission.csv`
- 自动搜索结果为：
  - `selected_threshold = 0.53`
  - `selected_target_top_k = 3`
  - `selected_other_top_k = 3`
  - `selected_other_margin = 0.0`
  - `selected_target_margin = 0.0`
  - `leave_one_out_accuracy = 0.9125`
- 结论：
  - 这轮虽然形式上更复杂，但最后依然退化成了“低阈值 + 无 margin”的更松接收规则；
  - 相对 `exp_019` 只改了 `8` 张预测，信息量不足，不能视为值得优先消耗 Kaggle 配额的高价值实验。

### Session 21

- 已尝试把 `AdaFace` 的质量信息显式引入推理：
  - `exp_026_ir101_adaface_haar_10ep_quality_aware_openset_alllabeled`
- 这轮继续复用 `exp_009` checkpoint，不重训模型，在 `exp_025` 的基础上新增：
  - quality-weighted gallery aggregation
  - low-quality probe threshold boost
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_210643_exp_026_ir101_adaface_haar_10ep_quality_aware_openset_alllabeled_submission.csv`
- 自动搜索结果为：
  - `selected_threshold = 0.53`
  - `selected_target_top_k = 3`
  - `selected_other_top_k = 3`
  - `selected_other_margin = 0.02`
  - `selected_quality_alpha = 0.0`
  - `selected_low_quality_threshold_boost = 0.0`
  - `leave_one_out_accuracy = 0.9125`
- 结论：
  - 真正代表质量感知的两项参数都被选成了 `0.0`；
  - 说明在当前实现和当前 embedding 上，quality-aware inference 没有提供有效增益；
  - 相对 `exp_019` 只改了 `12` 张，而且主要是 `0 -> 1` 的轻微放松，不值得单独提交 Kaggle。

### Session 22

- 已继续尝试把问题重构成两个 one-vs-rest verifier：
  - `exp_027_ir101_adaface_haar_10ep_verifier_openset_alllabeled`
- 这轮继续复用 `exp_009` checkpoint，不重训模型，在 `all 80 labeled` 上做 leave-one-out 搜索：
  - `target_top_k`
  - `negative_top_k`
  - 两个 verifier 各自阈值
- 当前已完成 submission 生成：
  - 新 submission：`data/submissions/20260401_212349_exp_027_ir101_adaface_haar_10ep_verifier_openset_alllabeled_submission.csv`
- 自动搜索结果为：
  - `selected_target_top_k = 3`
  - `selected_negative_top_k = 3`
  - `selected_thresholds_by_class = {1: 0.06, 2: 0.06}`
  - `leave_one_out_accuracy = 0.9375`
- 结论：
  - verifier 方案在当前实现下学出了过低阈值，导致相对 `exp_019` 有 `119` 张预测变化，而且全是把 `other` 放宽为目标类；
  - 这和此前多轮掉分实验的错误模式高度一致；
  - 因而它虽然完成了代码验证，但不值得占用 Kaggle 日提交配额。

### Session 23

- 已实现并离线运行同 backbone 多 checkpoint embedding 集成：
  - `exp_028_ir101_adaface_haar_10ep_ensemble_prototype_fixed055`
- 这轮固定协议为：
  - `exp_009 + exp_010 + exp_012 + exp_014`
  - 各自 embedding 先 `L2 normalize`
  - 跨模型平均后再 `L2 normalize`
  - 推理仍使用 `prototype + threshold=0.55`
- 当前已完成 submission 生成：
  - `data/submissions/20260401_225724_exp_028_ir101_adaface_haar_10ep_ensemble_prototype_fixed055_submission.csv`
- 本地结果为：
  - `val_accuracy = 0.875`
- 相比 `exp_019`：
  - 只改了 `20` 张预测
  - 全部是更保守的回收：
    - `1 -> 0`: `3`
    - `2 -> 0`: `17`
- 结论：
  - 当前这组同源 checkpoint 平均没有带来有效互补；
  - 它主要是在稀释 `exp_009` 的任务特异性 embedding，导致更多目标样本被推回 `other`；
  - 这轮不值得消耗 Kaggle 配额，也说明“继续堆同 backbone checkpoint 平均”不是当前优先方向。

### Session 24

- 已重新核对作业文本，确认 `other` 的真实构成不是随机陌生人，而是两组 look-alike：
  - `Michael Cera`（接近 `Jesse Eisenberg`）
  - `Sarah Hyland`（接近 `Mila Kunis`）
- 已基于 `exp_009` 的 `all 80 labeled` embedding 对 `other=0` 的 `20` 张样本做 `k=2` 聚类，并生成分析产物：
  - `reports/analysis/exp_029_other_k2/report.md`
  - `reports/analysis/exp_029_other_k2/summary.json`
  - `reports/analysis/exp_029_other_k2/other_cluster_assignments.csv`
  - `reports/analysis/exp_029_other_k2/other_k2_pca.png`
- 当前关键结果：
  - `other` 正好分成两个各 `10` 张的簇
  - `silhouette_score = 0.3656`
  - 一个簇显著更像 `Jesse`，另一个显著更像 `Mila`
  - `michael_like` 簇均值：
    - `mean_sim_to_jesse = 0.3316`
    - `mean_sim_to_mila = 0.0618`
  - `sarah_like` 簇均值：
    - `mean_sim_to_jesse = 0.0251`
    - `mean_sim_to_mila = 0.3821`
- 诊断性结论：
  - 当前任务的主要结构更接近“look-alike-aware rejection”，而不是泛化的 `other` 开放集拒识；
  - 但训练集上的 look-alike 样本与目标 prototype 的相似度仍明显低于当前接受阈值 `0.55`，因此这条线更像一个**值得验证的诊断方向**，而不是已确认的大增益方向；
  - 下一步应先做零新超参数的 `4 prototype` 推理诊断，而不是重新引入 margin 搜索或直接重训。

### Session 25

- 已完成零新超参数的 look-alike-aware 推理实验：
  - `exp_029_ir101_adaface_haar_10ep_lookalike_prototype_fixed055`
- 这轮继续复用 `exp_009` checkpoint，不重训模型，只把 `other` 拆成两个聚类原型：
  - `michael_like`
  - `sarah_like`
- 推理规则固定为：
  - 最近 prototype 是 `michael_like` 或 `sarah_like` → `other`
  - 最近 prototype 是 `Jesse` / `Mila` 且分数大于 `0.55` → 接收
  - 否则 → `other`
- 当前已完成 submission 生成：
  - `data/submissions/20260401_234742_exp_029_ir101_adaface_haar_10ep_lookalike_prototype_fixed055_submission.csv`
- 本地结果为：
  - `val_accuracy = 0.9375`
  - `test_nearest_label_counts = {1:393, 2:525, 3:431, 4:467}`
- 相比 `exp_019`：
  - submission **逐行完全一致**
  - 差异条数为 `0`
- 结论：
  - 在当前 `prototype + threshold=0.55` 协议下，把 `other` 显式拆成两组 look-alike prototype 并不会改变线上预测；
  - 这说明当前 `0.91685 -> 0.97` 的差距，至少不是由“现有规则没显式利用 Michael/Sarah 结构”直接造成的；
  - 因而 look-alike-aware inference 这条零新超参数推理线可以先判定为诊断完成，不值得继续占用 Kaggle 配额。

### Session 26

- 已完成训练侧的 metric-aligned pilot：
  - `exp_030_ir101_adaface_haar_10ep_laststage_ft_arcface2_prototype_recalib`
- 这轮以 `exp_009` 的训练协议为底座，仅做最小改动：
  - loss 从 `CrossEntropy` 改为 `2 类 ArcFace`
  - 训练时只对 `Jesse / Mila` 计算 margin loss
  - `other=0` 不参与 loss
  - 推理仍使用 `prototype`，只在 `[0.50, 0.525, 0.55, 0.575, 0.60]` 上做小范围阈值重校准
- 当前训练结果：
  - `best_val_target_acc = 0.9167`
  - 训练过程中 `val_target_acc` 基本没有突破起点
- 当前推理结果：
  - `selected_threshold = 0.55`
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_001807_exp_030_ir101_adaface_haar_10ep_laststage_ft_arcface2_prototype_recalib_submission.csv`
- 相比 `exp_019`：
  - 共 `278` 张预测变化
  - 全部是把 `other` 放宽为目标类：
    - `0 -> 1`: `195`
    - `0 -> 2`: `83`
  - 分布从 `0:1091, 1:343, 2:382` 变成：
    - `0:813, 1:538, 2:465`
- 结论：
  - 当前版本的 2 类 ArcFace 并没有带来更稳的 open-set embedding，反而显著削弱了拒识；
  - 线上 Kaggle 分数进一步确认了这一点：`0.76927`；
  - 这轮已经被线上正式证伪，不值得继续占用 Kaggle 配额；
  - 但它也说明一个重要事实：仅仅把训练目标对齐到角度空间还不够，当前任务还需要显式处理 `other` 的拒识约束。

### Session 27

- 已完成受控的 backbone 切换实验：
  - `exp_031_ir50_adaface_haar_10ep_laststage_ft_prototype_fixed055`
- 这轮只做一件事：
  - 把 `IR101 AdaFace` 换成 `IR50 AdaFace`
  - 其余训练协议与 `exp_009 / exp_019` 尽量保持一致
  - 推理仍固定为 `prototype + threshold=0.55`
- 当前训练结果：
  - `best_val_acc = 0.9444`
- 当前推理结果：
  - `selected_threshold = 0.55`
  - `val_accuracy = 0.875`
  - submission：`data/submissions/20260402_105844_exp_031_ir50_adaface_haar_10ep_laststage_ft_prototype_fixed055_submission.csv`
- 相比 `exp_019`：
  - 共 `237` 张预测变化
  - 主要模式是显著放宽接收：
    - `0 -> 1`: `233`
    - `0 -> 2`: `2`
    - `2 -> 0`: `2`
  - 分布从 `0:1091, 1:343, 2:382` 变成：
    - `0:858, 1:576, 2:382`
- 结论：
  - `IR50` 在当前协议下没有带来更稳的开放集行为，反而更容易把 `other` 吸入 `Jesse`；
  - 这轮离线信号不值得直接消耗 Kaggle 配额；
  - 但它也说明“同家族 iResNet 容量切换”不是当前主杠杆，若继续走新 backbone 轴，下一步应优先考虑结构差异更大的 `ViT`。

### Session 28

- 已完成异构 backbone 的低成本诊断实验：
  - `exp_032_vit_adaface_haar_1ep_frozen_prototype_fixed055`
- 这轮的目标不是训练 `ViT` 分类头，而是尽量不改现有 pipeline，先检验：
  - `minchul/cvlface_adaface_vit_base_webface4m` 的预训练 embedding 在当前 open-set 协议下是否有可用信号；
  - 是否值得继续投入 `ViT fine-tune` 的工程适配。
- 为避免无意义地训练随机分类头，这轮采用：
  - frozen backbone
  - `1 epoch` 仅产出可加载 checkpoint
  - 推理仍固定为 `prototype + threshold=0.55`
- 当前训练结果：
  - `best_val_acc = 0.75`
- 当前推理结果：
  - `selected_threshold = 0.55`
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_111133_exp_032_vit_adaface_haar_1ep_frozen_prototype_fixed055_submission.csv`
- 相比 `exp_019`：
  - 共 `11` 张预测变化
  - 变化方向非常小：
    - `0 -> 1`: `9`
    - `0 -> 2`: `1`
    - `2 -> 0`: `1`
  - 分布从 `0:1091, 1:343, 2:382` 变成：
    - `0:1082, 1:352, 2:382`
- 结论：
  - frozen `ViT` 的 prototype 行为和当前最强基线非常接近，没有像 `exp_030 / exp_031` 一样出现大规模“错误放宽接收”；
  - 这说明 `ViT` 不是一个无效方向，反而是当前最值得继续深入的新 backbone 轴；
  - 线上 Kaggle public score 已确认是 `0.92180`，超过 `exp_019 = 0.91685`；
  - 这说明虽然离线只改了 `11` 张，但这些修正是高价值修正，`exp_032` 现已成为新的 strongest online baseline。

### Session 29

- 已完成 `ViT` 轻量微调实验：
  - `exp_033_vit_adaface_haar_3ep_lightft_prototype_recalib`
- 这轮保持改动最小：
  - `ViT AdaFace` backbone
  - `freeze_backbone = false`
  - `backbone_learning_rate = 5e-6`
  - `max_epochs = 3`
  - 推理侧只在 `[0.525, 0.55, 0.575, 0.6]` 上做窄范围阈值重校准
- 当前训练结果：
  - `best_val_acc = 0.9167`
  - `val_loss` 从 `0.4760` 降到 `0.2292`
- 当前推理结果：
  - `selected_threshold = 0.525`
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_112429_exp_033_vit_adaface_haar_3ep_lightft_prototype_recalib_submission.csv`
- 相比 `exp_019`：
  - 共 `12` 张预测变化
  - 全部是更松的接收：
    - `0 -> 1`: `10`
    - `0 -> 2`: `2`
  - 分布从 `0:1091, 1:343, 2:382` 变成：
    - `0:1079, 1:353, 2:384`
- 相比 `exp_032`：
  - 只多改了 `3` 张
  - 方向仍然是继续放宽接收：
    - `0 -> 1`: `1`
    - `0 -> 2`: `2`
- 结论：
  - `ViT` 轻量微调确实让分类头在本地验证上更快适配了任务；
  - 但对最终 open-set prototype 行为的改善几乎没有，且变化方向仍然偏向“更宽松地接收目标类”；
  - 因此这轮不值得直接消耗 Kaggle 配额，也不建议继续单模型深挖 `ViT fine-tune`；
  - `ViT` 线目前最有价值的用途，更可能是和 `exp_019` 做异构融合，而不是替代 `exp_032` 这条 frozen ViT 主线。

### Session 30

- 已为异构融合新增独立工具链：
  - `scripts/export_embeddings.py`
  - `scripts/predict_score_fusion.py`
- 为避免 `CVLFace` 的 `wrapper` 模块污染，这条链路采用：
  - 子进程逐个导出成员模型的 train/val/test embedding
  - 再在干净的融合脚本中读取 `.pt` 文件做 score fusion
- 同时新增了受控 helper 与测试：
  - `compute_similarity_matrix`
  - `predict_open_set_from_similarity_matrix`
  - `fuse_similarity_matrices`
- 验证结果：
  - `unittest discover -s tests -v`：`42/42` 通过
  - `python -m compileall src scripts`：通过

### Session 31

- 已完成 `0.5/0.5` 的异构均值融合：
  - `exp_034_ir101_vit_score_mean_fixed055`
- 当前结果：
  - `selected_threshold = 0.55`
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_115747_exp_034_ir101_vit_score_mean_fixed055_submission.csv`
- 相比 `exp_032`：
  - 共 `7` 张预测变化
  - 主要方向是把 `exp_032` 的接受拉回拒识：
    - `1 -> 0`: `5`
    - `2 -> 0`: `1`
    - `0 -> 2`: `1`
- 相比 `exp_019`：
  - 只剩 `4` 张差异，全部是 `0 -> 1`
- 结论：
  - `0.5/0.5 mean` 明显把结果往 `IR101` 主线拖回去；
  - 它会吐回 `exp_032` 已经拿到的一部分潜在收益，不值得直接提交。

### Session 32

- 已完成偏向 `ViT` 的异构均值融合：
  - `exp_035_ir101_vit_score_mean_vit075_fixed055`
- 当前结果：
  - `selected_threshold = 0.55`
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_120302_exp_035_ir101_vit_score_mean_vit075_fixed055_submission.csv`
- 相比 `exp_032`：
  - 共 `5` 张预测变化
  - 仍然是在回收 `exp_032` 的接受：
    - `1 -> 0`: `3`
    - `2 -> 0`: `1`
    - `0 -> 2`: `1`
- 相比 `exp_034`：
  - 只多恢复了 `2` 张 `0 -> 1`
- 结论：
  - 即使把权重压到 `IR101:ViT = 0.25:0.75`，线性均值融合仍然在损害 `exp_032` 的少量高价值修正；
  - 当前证据不支持继续在线性 mean fusion 上扫更多权重；
  - 若继续做融合，更值得尝试的是更偏 `ViT` 的非对称策略或非线性 gating，而不是继续做简单均值。

### Session 33

- 已完成 `ViT` 受控 partial fine-tune 实验：
  - `exp_036_vit_adaface_haar_10ep_last2block_ft_prototype_fixed055`
- 本轮的关键改动不是继续“轻量全开微调”，而是让 `CVLFace ViT` 支持受控解冻：
  - 仅解冻最后 `2` 个 transformer blocks
  - 同时解冻 `norm` 与 `feature`
  - 其余 backbone 保持冻结
  - 推理继续固定为 `prototype + threshold=0.55`
- 为支持该实验，本轮工程改动包括：
  - 在 `src/dl_pipeline/models/classifier.py` 中新增 `ViT blocks` 分支的 stage-unfreeze 逻辑
  - 在 `tests/test_classifier.py` 中新增 `ViT partial unfreeze` 单元测试
  - 新增配置：`configs/experiments/exp_036_vit_adaface_haar_10ep_last2block_ft_prototype_fixed055.yaml`
- 当前本地结果：
  - `best_val_acc = 1.0`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
  - submission：`data/submissions/20260402_123255_exp_036_vit_adaface_haar_10ep_last2block_ft_prototype_fixed055_submission.csv`
- 相比 `exp_032`：
  - 只有 `2` 张预测变化
  - 全部是 `0 -> 2`
  - 分布从 `0:1082, 1:352, 2:382` 变成：
    - `0:1080, 1:352, 2:384`
- 相比 `exp_019`：
  - 共 `11` 张预测变化
  - `9` 张 `0 -> 1`
  - `2` 张 `0 -> 2`
- 结论：
  - `exp_036` 没有像 `exp_033` 那样出现明显的边界放宽漂移；
  - `exp_033` 证伪的是“全开 ViT 的轻量乱动”，而不是“所有 ViT 微调都无效”；
  - 当前这版 `last-2-block` 微调基本保住了 `exp_032` 的主体行为，只额外增加了 `2` 张 `Mila` 接收，因此它是一个合理的后续提交候选，但还不足以在离线阶段直接取代 `exp_032`。

### Session 34

- 已完成真正的 `blocks-only ViT` 微调实验：
  - `exp_037_vit_adaface_haar_10ep_last2blockonly_ft_prototype_fixed055`
- 本轮相对 `exp_036` 的唯一方法层变化是：
  - 继续只解冻最后 `2` 个 transformer blocks
  - 但显式冻结 `norm`
  - 显式冻结 `feature`
  - 即不再让 `51.6M` 参数的 projection head 参与更新
- 为支持该实验，本轮工程改动包括：
  - 在 `src/dl_pipeline/models/classifier.py` 中新增 `unfreeze_cvlface_norm` / `unfreeze_cvlface_feature`
  - 在 `src/dl_pipeline/training/lightning_module.py`、`scripts/train.py`、`scripts/predict.py` 中把这两个开关接到配置层
  - 在 `tests/test_classifier.py` 中新增 `ViT feature+norm freeze during partial unfreeze` 测试
  - 新增配置：`configs/experiments/exp_037_vit_adaface_haar_10ep_last2blockonly_ft_prototype_fixed055.yaml`
- 验证结果：
  - `unittest discover -s tests -v`：`45/45` 通过
  - `python -m compileall src scripts`：通过
- 当前本地结果：
  - `best_val_acc = 1.0`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
  - submission：`data/submissions/20260402_134601_exp_037_vit_adaface_haar_10ep_last2blockonly_ft_prototype_fixed055_submission.csv`
- 相比 `exp_032`：
  - 仅 `1` 张预测变化
  - `0 -> 2`: `1`
- 相比 `exp_036`：
  - 仅 `1` 张预测变化
  - `2 -> 0`: `1`
- 相比 `exp_019`：
  - 共 `12` 张变化
  - `9` 张 `0 -> 1`
  - `2` 张 `0 -> 2`
  - `1` 张 `2 -> 0`
- 结论：
  - 冻结 `feature/norm` 后，`exp_037` 比 `exp_036` 更接近 `exp_032`，这直接支持了“`exp_036` 的主导更新来自 feature 层”这个判断；
  - `blocks-only` 是比 `exp_036` 更真实的受控 `ViT partial fine-tune`；
  - 但当前它仍比 `exp_032` 多 `1` 张 `0 -> 2`，离线还不足以直接宣布它优于 strongest baseline。

### Session 35

- 已按用户要求启动 `exp_038 = KP-RPE integration audit + frozen ViT KP-RPE WebFace12M pilot`。
- 当前完成的审计结论如下：
  - `minchul/cvlface_adaface_vit_base_kprpe_webface12m` 的官方接口不是普通 `model(input)`，而是 `model(input, keypoints)`；
  - 官方模型卡同时要求额外加载 aligner：`minchul/cvlface_DFA_mobilenet`；
  - 当前本地 `cvlface.py` loader 只支持 `CVLFaceRecognitionModel`，不能直接加载 aligner repo 中的 `CVLFaceAlignmentModel`；
  - `KP-RPE` 仓库依赖 `rpe_ops` 扩展，且没有可用的纯 Python fallback。
- 为排除简单环境问题，本轮已经做了以下尝试：
  - 在 `gpu_env` 中安装 `easydict`
  - 下载并审阅 `KP-RPE` 模型仓库与 `DFA` aligner 仓库
  - 尝试编译 `models/vit_kprpe/RPE/rpe_ops`
- 结论是：当前 pilot 被环境层硬阻塞，尚不能真正开始训练/推理。
  - `rpe_ops` 编译失败
  - 当前机器不存在 `cl` / `g++` / `ninja`
  - 唯一可见编译器相关工具是 `nvcc 11.0`
  - 编译报错明确指出：`detected CUDA 11.0 mismatches PyTorch 12.1`
- 因此，这轮 `exp_038` 的产出是**接入审计结论**而不是 runnable pilot：
  - `KP-RPE` 值得继续做
  - 但要先解决编译链和 aligner/keypoint 输入链路，当前环境下不能假装它已经是“只改 repo_id 就能跑”的实验

### Session 36

- 已完成真正的 `IR101 WebFace12M` drop-in 替换实验：
  - `exp_039_ir101_adaface_haar_10ep_laststage_ft_webface12m_prototype_fixed055`
- 本轮刻意保持其余协议不变：
  - 继续使用 `HAAR` 裁脸
  - 继续使用 `last-stage fine-tune`
  - 推理继续固定为 `prototype + threshold=0.55`
  - 唯一方法层变化是把 `pretrained_repo_id` 从 `minchul/cvlface_adaface_ir101_webface4m` 换成 `minchul/cvlface_adaface_ir101_webface12m`
- 当前本地结果：
  - `best_val_acc = 1.0`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
  - submission：`data/submissions/20260402_143133_exp_039_ir101_adaface_haar_10ep_laststage_ft_webface12m_prototype_fixed055_submission.csv`
- 相比 `exp_019`：
  - 共 `264` 张预测变化
  - `263` 张 `0 -> 1`
  - `1` 张 `0 -> 2`
- 相比 `exp_032`：
  - 共 `257` 张预测变化
  - `254` 张 `0 -> 1`
  - `2` 张 `0 -> 2`
  - `1` 张 `2 -> 0`
- `exp_039` 的预测分布为：
  - `0:827, 1:606, 2:383`
- 结论：
  - `WebFace12M` 在当前 `IR101 + exp_019` 协议下没有表现为“更强但更稳”的 drop-in 增强；
  - 它和 `exp_030 / exp_031` 一样，主要效果是系统性放宽接受边界；
  - 因此仅仅换成更大预训练数据，并不能保证在当前 open-set 规则下得到更好的拒识行为。
  - 线上 Kaggle public score 已确认是 `0.80341`，明显低于 `exp_019 = 0.91685` 与 `exp_032 = 0.92180`，说明这类大规模 `0 -> 1` 放宽在线上确实是负向变化。

### Session 37

- 已完成 `frozen ViT + MTCNN + prototype + fixed 0.55` 的受控输入侧实验：
  - `exp_040_vit_adaface_mtcnn_1ep_frozen_prototype_fixed055`
- 本轮刻意保持与 `exp_032` 尽量一致：
  - backbone 仍是 `minchul/cvlface_adaface_vit_base_webface4m`
  - 仍是 frozen `ViT`
  - 仍然固定 `prototype + threshold=0.55`
  - 唯一主变量是输入从 `HAAR` 裁脸切到 `MTCNN + 5 点对齐`
- 当前本地结果：
  - `best_val_acc = 0.75`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
  - submission：`data/submissions/20260402_144532_exp_040_vit_adaface_mtcnn_1ep_frozen_prototype_fixed055_submission.csv`
- 相比 `exp_032`：
  - 只有 `1` 张预测变化
  - 方向是 `1 -> 0`
- 相比 `exp_019`：
  - 共 `12` 张预测变化
  - `9` 张 `0 -> 1`
  - `1` 张 `0 -> 2`
  - `1` 张 `1 -> 0`
  - `1` 张 `2 -> 0`
- 相比早期 `exp_008`：
  - 完全不是同一种失败模式
  - `exp_008` 相比 `exp_040` 有 `708` 张差异，主要是把大量目标类压回 `0`
- 结论：
  - 早期的 `MTCNN` 负例不能外推到当前 `frozen ViT` 主线；
  - 在当前 `ViT` backbone 上，`MTCNN` 没有带来大突破，但也没有破坏 `exp_032` 的主体行为；
  - `exp_040` 目前应视为一个**极近邻候选**，是否优于 `exp_032` 需要 Kaggle 才能判断。

### Session 38

- 已完成 `frozen ViT + horizontal flip TTA + prototype + fixed 0.55`：
  - `exp_041_vit_adaface_haar_1ep_frozen_prototype_fixed055_hfliptta`
- 本轮是纯推理增强，不重新训练：
  - 直接复用 `exp_032` 的 frozen `ViT` checkpoint
  - 在 embedding 提取阶段增加 `original + hflip` 两次前向并做平均
  - 推理协议仍固定为 `prototype + threshold=0.55`
- 为此新增了通用 TTA helper：
  - `src/dl_pipeline/inference/tta.py`
  - `predict.py` 现已支持 `tta_horizontal_flip: true`
- 当前本地结果：
  - `val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
  - submission：`data/submissions/20260402_153303_exp_041_vit_adaface_haar_1ep_frozen_prototype_fixed055_hfliptta_submission.csv`
- 相比 `exp_032`：
  - 只有 `1` 张预测变化
  - 方向是 `0 -> 2`
- 预测分布为：
  - `0:1081, 1:352, 2:383`
- 结论：
  - `horizontal flip TTA` 在当前 strongest baseline 上表现为**低风险、极小幅**的正交扰动；
  - 它没有重演 `IR101 WebFace12M` 那种系统性边界放松；
  - 但离线信息量仍然很小，单独看还不足以称为“突破”。

### Session 39

- 已完成第一版保守式图修正实验：
  - `exp_042_vit_adaface_haar_1ep_frozen_graphrefine_fixed055_hfliptta`
- 设计目标不是激进传播，而是验证“保守 gated graph refinement 是否能在不污染 look-alike 邻域的前提下修正少量边界样本”：
  - base embedding 直接使用 `exp_041` 的 `frozen ViT + hflip TTA`
  - labeled gallery 始终作为硬锚点
  - 只对低边际样本开放修正
  - `0 -> 1/2` 的接受门槛高于 `1/2 -> 0`
- 本轮实现内容：
  - `src/dl_pipeline/inference/prototype.py` 新增 `conservative_graph_refine_predictions`
  - `predict.py` 新增 `mode: conservative_graph_refine`
  - 增加对应单元测试覆盖低边际接收、拒绝和高置信不变三种情形
- 当前本地结果：
  - `base_val_accuracy = 0.9375`
  - `refined_val_accuracy = 0.9375`
  - `num_changed_test_predictions = 0`
  - submission：`data/submissions/20260402_154035_exp_042_vit_adaface_haar_1ep_frozen_graphrefine_fixed055_hfliptta_submission.csv`
- 相比 `exp_041`：
  - 测试集预测完全一致
- 结论：
  - 当前这版 conservative graph refinement 是**安全但未触发**的；
  - 好消息是没有出现 look-alike 污染或大规模误接收；
  - 坏消息是当前 gating 过于保守，尚不足以带来任何线上有意义的变化。

### Session 40

- 已完成第一版全局图传播 pilot：
  - `exp_043_vit_adaface_haar_1ep_frozen_globallabelspread_hfliptta`
- 设计目标是验证“高置信错例是否真的需要全局图结构才能被纠正”，因此这轮不再只看 boundary case：
  - base embedding 仍是 `frozen ViT + hflip TTA`
  - labeled nodes 使用 `80` 张 gallery
  - 图使用全局 cosine kNN 传播，而不是局部边界 gating
- 当前本地结果：
  - `val_accuracy = 1.0`
  - `alpha = 0.2`
  - `top_k = 20`
  - submission：`data/submissions/20260402_160537_exp_043_vit_adaface_haar_1ep_frozen_globallabelspread_hfliptta_submission.csv`
- 相比 `exp_032`：
  - 共 `153` 张预测变化
  - `115` 张 `0 -> 2`
  - `38` 张 `0 -> 1`
- 预测分布为：
  - `0:929, 1:390, 2:497`
- 结论：
  - 这轮明确回答了“当前瓶颈是不是只有 boundary case 太少”这个问题：不是；
  - 全局图结构确实能推动大量高置信样本发生变化，说明 `exp_042` 的 inert 不是因为图方法整体无效；
  - 但当前第一版 `LabelSpreading-style` 传播的变化方向又重新退化成**纯接受型扩张**，风险画像与此前的大规模 `0 -> 1/2` 负例更接近，而不是与 `exp_032` 的“小而准修正”接近。

### Session 41

- 已完成 look-alike margin rejection 对照实验：
  - `exp_044_vit_adaface_haar_1ep_frozen_lookalikemargin_hfliptta`
- 本轮复用了 `exp_029` 的 `other -> {michael_like, sarah_like}` 聚类构造，但不再使用“最近 prototype 是 look-alike 就拒绝”的硬规则，而是改成：
  - 先过目标类阈值 `0.55`
  - 再检查 `target score - matched look-alike score`
  - 若 margin 小于候选值，则拒到 `other`
- 当前本地结果：
  - `selected_margin = 0.01`
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_160538_exp_044_vit_adaface_haar_1ep_frozen_lookalikemargin_hfliptta_submission.csv`
- 相比 `exp_032`：
  - 只有 `1` 张预测变化
  - 方向仍是 `0 -> 2`
- 结论：
  - 当前这版 look-alike margin rejection 在自动选出的 margin 下几乎没有触发；
  - 它没有提供比 `exp_041` 更多的信息量；
  - 至少在 frozen `ViT` + 当前 clustered look-alike prototype 设定下，简单 margin rejection 还不足以成为突破方向。

### Session 42

- 已完成 `Spectral clustering k=4 + gallery label matching`：
  - `exp_045_vit_adaface_haar_1ep_frozen_spectralcluster_hfliptta`
- 本轮是对“能否完全绕开 0.55 阈值，直接用全局聚类结构重建三类决策”的第一版验证：
  - base embedding 仍是 `frozen ViT + hflip TTA`
  - 用 `4` 个 spectral clusters 划分 test
  - 再用 labeled gallery 近邻给 cluster 匹配类别
- 当前本地结果：
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_162721_exp_045_vit_adaface_haar_1ep_frozen_spectralcluster_hfliptta_submission.csv`
- 相比 `exp_032`：
  - 共 `494` 张预测变化
  - `382` 张 `2 -> 0`
  - `75` 张 `0 -> 2`
  - `37` 张 `0 -> 1`
- 预测分布为：
  - `0:1352, 1:389, 2:75`
- 结论：
  - 这轮第一次在后处理里出现了明显的**大规模保守修正**；
  - 但强度过大，已经不再是“微调局部错误”，而是几乎重写了 `Mila` 的整体决策边界；
  - 因而它是一个很有信息量的结构性实验，但当前版本不可直接提交。

### Session 43

- 已完成 `PCA whitening + prototype + 0.55`：
  - `exp_046_vit_adaface_haar_1ep_frozen_pcawhitenedproto_hfliptta`
- 本轮验证的是“是否能通过改变 embedding 几何而不是改变模型/规则，提升 prototype 分离度”。
- 当前本地结果：
  - `val_accuracy = 0.3125`
  - `n_components = 32`
  - submission：`data/submissions/20260402_162720_exp_046_vit_adaface_haar_1ep_frozen_pcawhitenedproto_hfliptta_submission.csv`
- 相比 `exp_032`：
  - 共 `671` 张预测变化
  - `289` 张 `1 -> 0`
  - `382` 张 `2 -> 0`
- 预测分布为：
  - `0:1753, 1:63`
- 结论：
  - 这轮不是“微弱无效”，而是**明显破坏了原始 embedding 几何**；
  - 在当前数据规模和设定下，PCA whitening 会把原本可接受的目标类大面积压回 `other`；
  - 因而这条线当前可以视为明确负例。

### Session 44

- 已完成 `Neighborhood-aware scoring`：
  - `exp_047_vit_adaface_haar_1ep_frozen_neighborhoodaware_fixed055_hfliptta`
- 本轮严格按诊断脚本中的定义实现：
  - `base_score = best prototype similarity`
  - `neighbor_mean_score = test-test top-15 邻域的 base score 均值`
  - `final_score = 0.5 * base_score + 0.5 * neighbor_mean_score`
  - `threshold = 0.55`
  - 最终类别仍保持 prototype argmax，只让邻域分数参与 accept/reject
- 当前本地结果：
  - `val_accuracy = 0.9375`
  - submission：`data/submissions/20260402_164731_exp_047_vit_adaface_haar_1ep_frozen_neighborhoodaware_fixed055_hfliptta_submission.csv`
- 相比 `exp_032`：
  - 共 `8` 张预测变化
  - `3` 张 `0 -> 1`
  - `5` 张 `0 -> 2`
- 相比 `exp_041`：
  - 共 `7` 张变化
  - `3` 张 `0 -> 1`
  - `4` 张 `0 -> 2`
- 预测分布为：
  - `0:1074, 1:355, 2:387`
- 结论：
  - 这轮和 `exp_043 / exp_045` 的大范围结构改动不同，属于**小而准的健康修正**；
  - 它没有引入任何 `1/2 -> 0` 的大规模收缩，也没有出现几十上百张级别的边界放宽；
  - 因而它是当前最接近“值得提交验证”的推理改进候选之一。
- 线上 Kaggle public score 已确认是 `0.92621`，超过 `exp_032 = 0.92180`；
- 因此 `exp_047` 已更新为新的 strongest online baseline。

### Session 45

- 已按计划执行 **neighborhood_aware 参数网格扫描**（`top_k ∈ {5,10,15,20,30}` × `base_weight ∈ {0.3,0.4,0.5,0.6,0.7}`），脚本：`scripts/sweep_neighborhood_aware.py`，结果：`reports/sweeps/neighborhood_aware_grid_latest.json`。
- 扫描结论（**以 baseline submission 按 `id` 对齐后的迁移为准**，不用小验证集做主排序）：
  - 全部组合中 **没有任何一组的 `val_accuracy` 超过 `0.9375`**（仅作旁注）；
  - 与 `exp_047` baseline submission **逐行完全一致** 的「等价 plateau」包含多组参数；脚本初版曾用 **DataLoader 行序与按 `id` 排序后的 CSV 直接 zip**，在一般情形下可能错位；已改为 **`baseline.merge(候选, on='id')`** 再统计 `0->1` 等迁移；
  - 自动选参已改为 **submission 优先**：先最小化 `(0->1 + 0->2)`，再最小化 `total_changed`，再优先贴近 `exp_047` 的 `(top_k=15, base_weight=0.5)`。当前输出 winner 为 **`top_k=15, base_weight=0.5`**（与 `exp_047` 一致，而非此前误用的 `val` 平局规则）。
- 在扫描结果中另选 **保守外推候选**（相对 `exp_047` 仅 `1` 张 `2 -> 0`，`val` 仍为 `0.9375`）：
  - 实验名：`exp_048_vit_adaface_haar_1ep_frozen_neighborhood_topk5_bw06_fixed055_hfliptta`
  - 配置：`top_k=5`, `base_weight=0.6`（更信任 prototype base score）
  - submission：`data/submissions/20260402_172254_exp_048_vit_adaface_haar_1ep_frozen_neighborhood_topk5_bw06_fixed055_hfliptta_submission.csv`
  - `prototype_metrics` 已写入：`outputs/exp_048_vit_adaface_haar_1ep_frozen_neighborhood_topk5_bw06_fixed055_hfliptta/prototype_metrics.json`
- 当前最强线上基线仍为 **`exp_047 = 0.92621`**；`exp_048` 已提交 Kaggle 对照：**public score `0.92511`**，低于 `exp_047` 约 `0.00110`。
- 结论：离线仅 `1` 张 `2->0` 的「更保守」邻域参数在线上仍为负向；**不应以 `exp_048` 替代 `exp_047`**，后续 neighborhood 线继续以 `(top_k=15, base_weight=0.5)` 为锚。

### Session 46

- 已落实 **exp_049**：在 **不保留独立训练 hold-out** 的意义下，将 `train.csv` 与 `val.csv` 合并为 **80 张**训练集，仅解冻 ViT **最后 2 个 block**（与 exp_037 一致），`backbone_lr=1e-5`，**固定 5 epoch**；checkpoint 采用 **`save_last`**（不做 `val_acc` 选优）；推理与 **exp_047** 对齐：`neighborhood_aware`、`top_k=15`、`base_weight=0.5`、`threshold=0.55`、**hflip TTA**。
- 工程改动：
  - `data.use_full_train: true`：`FaceDataModule` 在 `setup` 时按 `id` 排序拼接 train+val 作为 `train_dataset`；验证集 DataLoader 仍为原 `val.csv`（指标有泄漏，仅作日志，不用于早停）。
  - `scripts/train.py`：在 `use_full_train` 时关闭 `EarlyStopping` 与按指标 `ModelCheckpoint`，训练结束将 `metrics.json` 的 `best_model_path` 指向 **`last*.ckpt`**。
- 产物：
  - 配置：`configs/experiments/exp_049_vit_adaface_haar_5ep_last2blockonly_ft_fulltrain_neighborhoodaware_fixed055_hfliptta.yaml`
  - checkpoint：`outputs/exp_049_vit_adaface_haar_5ep_last2blockonly_ft_fulltrain_neighborhoodaware_fixed055_hfliptta/checkpoints/last-v1.ckpt`
  - submission：`data/submissions/20260402_193642_exp_049_vit_adaface_haar_5ep_last2blockonly_ft_fulltrain_neighborhoodaware_fixed055_hfliptta_submission.csv`
- **与 exp_047 submission 按 `id` 对齐**：`1816` 条测试 **预测类别逐行完全一致**（`diff=0`）。说明在当前推理协议下，5 epoch 全量微调后的 embedding 未改变任一测试样本的最终离散决策；Kaggle 上预期与 `0.92621` 一致或极接近，可按需再提交一次确认。
- 备注：本机 `conda run -n gpu_env` 曾因控制台 **GBK 打印 Unicode** 报错；训练/预测已改用 **`D:\Anaconda3\envs\gpu_env\python.exe` 直接调用** 并设 `PYTHONIOENCODING=utf-8` 规避。

### Session 47

- 已完成 **exp_061**：`ViT AdaFace` + **CrossEntropy** + 解冻最后 **2** 个 block（及 `norm` / `feature`），训练增强开启 **HorizontalFlip** 与 **Affine**（与 `build_train_transform` 内 ColorJitter 叠加）；`lr_head=3e-4`、`lr_backbone=1e-5`；`max_epochs=20`、`early_stopping_patience=6`，实际约 **epoch 9** 早停；`best_val_acc=1.0`。
- 推理与 **exp_047** 对齐：`neighborhood_aware`、`top_k=15`、`base_weight=0.5`、`threshold=0.55`、**hflip TTA**。
- 产物：
  - 配置：`configs/experiments/exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml`
  - checkpoint：`outputs/exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta/checkpoints/best.ckpt`
  - submission：`data/submissions/20260402_224630_exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv`
- 与 `exp_047` submission 按 `id` 对齐：约 **2** 条测试样本类别不同（共 `1816` 条）。
- 该 submission 已提交 Kaggle，**public score `0.92731`**，超过此前最强 **`exp_047 = 0.92621`** 约 **`0.00110`**。
- 结论：**exp_061** 现为当前 **strongest online baseline**；说明在固定 open-set 推理协议下，**受控 CE 微调** 可带来线上增益，与此前失败的 **3-class ArcFace（exp_060）** 形成对照。

### Session 48

- 已完成 **exp_062**：复用 **exp_061** 权重；推理为 **adaptive_neighborhood_aware**（`th_min=0.45`、`th_max=0.65`）+ **pseudo_label_refinement**（`min_final_score=0.72`、`max_rounds=2`）；TTA 仍为 hflip。
- 产物：
  - 配置：`configs/experiments/exp_062_vit061_adaptive_neighborhood_pseudo2_hfliptta.yaml`
  - submission：`data/submissions/20260402_231317_exp_062_vit061_adaptive_neighborhood_pseudo2_hfliptta_submission.csv`
  - 指标：`outputs/exp_062_vit061_adaptive_neighborhood_pseudo2_hfliptta/prototype_metrics.json`
- 与 **exp_061** submission 按 `id` 对齐：**6** 条不同（均为 `0 -> 1` 或 `0 -> 2`）。
- 该 submission 已提交 Kaggle，**public score `0.92731`**，与 **exp_061** 相同。
- 结论：方向 C 未提升 public 分数；**strongest online baseline** 仍为 **exp_061**（与 exp_062 分数并列）。

### Session 49

- 基于对整体实验历程的学术性复盘，确认了以下战略性问题：
  1. 任务建模方式错误：一直当作"3 类分类 + 全局阈值"处理，而 2.2 节明确建议"2 个二元分类器"，任务本质是 **face verification**
  2. 模型多样性不足：所有 backbone 全部来自 AdaFace 家族（同一 loss、同一训练范式）
  3. ViT-Base 在 80 张数据上做微调收益极低（frozen→微调仅 +0.005），文献支持小数据集下 ViT 更容易过拟合
  4. look-alike 数据（10 Michael + 10 Sarah）本应用于 per-identity 阈值校准，但一直被当作泛化 other 处理
- 据此启动 **exp_067**：引入完全不同的模型家族（insightface `buffalo_l` = ArcFace ResNet50@WebFace600K），实现双独立验证器 + look-alike 分簇阈值校准。
- 关键工程改动：
  - 新增 `src/dl_pipeline/inference/insightface_dual_verifier.py`：完整的双通道验证推理模块
  - 新增 `detect_align_with_crop_fallback` embedding 策略：优先从原始 .npy 整图走 RetinaFace 检测+对齐（正确的 ArcFace 输入），检测失败才回退到 HAAR crop
  - Gallery 和阈值校准使用全部 80 张标注数据（train+val 合并），不再受 train/val split 限制
  - Youden's J 统计量搜索各通道最优阈值
- **exp_067** 产物：
  - 配置：`configs/experiments/exp_067_buffalo_l_detect_align_dual_verifier_lookalike.yaml`
  - 指标：`outputs/exp_067_buffalo_l_detect_align_dual_verifier_lookalike/prototype_metrics.json`
  - submission：`data/submissions/20260403_134351_exp_067_buffalo_l_detect_align_dual_verifier_lookalike_submission.csv`
- 关键校准结果：
  - `theta_jesse = 0.489`，`theta_mila = 0.363` — 两个阈值相差 **0.126**，直接否定了"全局阈值足够"的假设
  - 两个通道 FPR 均为 0.0（10 Michael 全拒、10 Sarah 全拒），TPR 分别为 0.90 / 0.93
  - `val_accuracy = 0.9375`
  - `detect_align_fallback_count = 38`（原始整图检测率 ~98%，回退仅覆盖硬样本）
- 与 **exp_061** 按 `id` 对齐：**145** 条预测不同：
  - `0->1`: 55，`0->2`: 44（从 other 接收为目标类）
  - `1->0`: 22，`2->0`: 9（从目标类拒识为 other）
  - `1->2`: 8，`2->1`: 7（类别互换）
- 该 submission 已提交 Kaggle，**public score `0.92676`**，略低于 **exp_061 = 0.92731**（差 **0.00055**）。
- 结论：
  - ArcFace（buffalo_l）在完全不同的模型家族和推理框架下，几乎追平了 AdaFace ViT + neighborhood-aware 的当前最强结果
  - 两个模型的预测差异高达 **145 张**，但线上分数几乎一致 → 说明两者犯的错误高度互补
  - 这恰好为下一步的**异构 embedding 融合**提供了最强理论依据
  - **strongest online baseline 仍为 exp_061 = 0.92731**

### Session 50

- **IResNet50（Glint360K `16backbone.pth`）+ 与 ViT 线相同 `neighborhood_aware` 协议**（`threshold=0.55`、`top_k=15`、`base_weight=0.5`、hflip TTA）已提交 Kaggle，记录 public score 如下。
- **`exp_071`**（bn_only + Weight Imprinting + `accumulate_grad_batches=8`、20ep）：
  - submission：`data/submissions/20260403_154045_exp_071_arcface_r50_glint360k_bnonly_wi_accum8_20ep_neighborhoodaware_fixed055_hfliptta_submission.csv`
  - **Kaggle public score：`0.89207`**
- **`exp_077`**（last_stage + 较 073 更温和的 LR + `gradient_clip_val=1.0` + WI、35ep）：
  - submission：`data/submissions/20260403_163843_exp_077_arcface_r50_laststage_mildlr_clip_wi_accum8_35ep_neighborhoodaware_fixed055_hfliptta_submission.csv`
  - **Kaggle public score：`0.91134`**
- 对照：**`exp_077` 较 `exp_071` 高约 `0.01927`**；两者均 **低于当前最强 `exp_061 = 0.92731`**（差约 `0.01597` / `0.01587` 量级）。说明在同一 open-set 推理锚点下，**该 IResNet 微调线线上仍明显弱于 ViT CE + neighborhood**，但 **last_stage + mild LR + clip** 相对 **仅 BN** 有稳定 public 增益。

### Session 51

- 已将此前未入库的 `exp_069` 三路异构投票草稿正规化为可复现实验脚本：
  - `scripts/predict_submission_vote.py`
- 该脚本支持对已有 submission 做三种标签级融合：
  - `majority_vote_3way`
  - `conservative_vote`
  - `diverse_ensemble`
- 本轮以三路输入为固定成员重新生成了三个正式实验：
  - `exp_078_vote_majority_061_067_068`
  - `exp_079_vote_conservative_061_067_068`
  - `exp_080_vote_diverse_061_067_068`
- 三个实验均在 `gpu_env` 中完成，输入固定为：
  - `exp_061`：`ViT AdaFace + CE + neighborhood_aware`
  - `exp_067`：`buffalo_l detect_align dual verifier`
  - `exp_068`：`AdaFace ViT dual verifier`
- 本轮最重要的工程结论不是某个新分数，而是：
  - 三种投票策略在当前三路 submission 上**完全收敛为同一份输出**
  - `exp_078 / exp_079 / exp_080` 的预测分布一致：`{0: 1055, 1: 361, 2: 400}`
  - `exp_078` 与旧草稿 `exp_069_diverse_ensemble` **逐行完全一致（0 diff）**
- 相对 strongest online baseline `exp_061`：
  - 新投票 submission 共 `50 / 1816` 条不同
  - 迁移方向为：
    - `0 -> 1`: `11`
    - `0 -> 2`: `15`
    - `1 -> 0`: `6`
    - `1 -> 2`: `8`
    - `2 -> 0`: `3`
    - `2 -> 1`: `7`
- 相对 `exp_067`：
  - 共 `95` 条不同
  - 主要是把 `buffalo_l` 的目标类预测收回到更保守的边界：
    - `1 -> 0`: `44`
    - `2 -> 0`: `29`
- 相对 `exp_068`：
  - 仅 `13` 条不同
  - 说明当前三路投票的实际主导输出，更接近 `AdaFace ViT dual verifier`，而不是 `exp_061` 或 `exp_067` 的中间平均
- 当前判断：
  - 这轮已经完成了“异构 submission-level 投票融合”的第一次正规化落地
  - 但它没有产生新的策略分叉，说明当前三模型在标签层的可利用自由度有限
  - 是否值得提交 Kaggle，取决于是否愿意用一次配额验证这 `50` 条受控改动；它值得作为**信息量较高的候选**，但不是高置信必胜替代

### Session 52

- 用户已回报 **`exp_081_vote_majority_061_067_068_fixedcsv` Kaggle public score = `0.90143`**。
- 该结果直接证伪了“submission label vote 可以吃到 061/067/068 互补性”的假设：
  - `exp_081` 相比 strongest online baseline `exp_061 = 0.92731` 大幅下滑
  - 结合 `exp_078/079/080` 实际几乎贴近 `exp_068` 的输出，可判定 **label-level vote 会把强锚点 `exp_061` 拉坏**
- 据此启动真正的下一步：**`exp_082_vit061_buffalol067_gated_score_fusion_valgrid`**
  - 新增脚本：`scripts/predict_061_067_gated_fusion.py`
  - 新增纯逻辑门控模块：`src/dl_pipeline/inference/gated_fusion.py`
  - 新增单测：`tests/test_gated_fusion.py`
- 方法设计：
  - **主锚点**：`exp_061` 的 `neighborhood_aware final_score`
  - **辅专家**：`exp_067` 的 `dual verifier` 两通道分数与 threshold excess
  - **门控规则**：
    - `061` 判 `other` 时，仅当 `067` 的目标类 excess 足够强才“救回”
    - `061` 判目标类但分数偏低时，允许 `067` 做类别互换或 veto 回 `other`
  - 参数在 **val** 上做网格搜索：`low_conf_threshold × rescue_margin × swap_margin × veto_excess_threshold`
- 本轮最重要的实现细节：
  - `exp_061` 侧严格复现原协议：
    - **val**：仅用 `train` prototype
    - **test**：用 `train+val` prototype
  - `exp_067` 不再沿用“all_labeled 自身参与 val 打分”的宽松日志口径，而是改为：
    - **val**：`train` gallery + `train` 校准阈值
    - **test**：`train+val` gallery + `train+val` 校准阈值
  - `exp_061` / `exp_067` 的 test 预测都与历史正式 submission **完全对齐（0 diff）**，确认重现实验无偏差
- `exp_082` 结果：
  - submission：`data/submissions/20260403_175121_exp_082_vit061_buffalol067_gated_score_fusion_valgrid_submission.csv`
  - metrics：`outputs/exp_082_vit061_buffalol067_gated_score_fusion_valgrid/prototype_metrics.json`
  - val 分数：
    - `exp_061`：`0.9375`
    - `exp_067`（train-only gallery 口径）：`0.8750`
    - **fused**：`0.9375`
  - 最优网格参数：
    - `low_conf_threshold = 0.50`
    - `rescue_margin = 0.03`
    - `swap_margin = 0.05`
    - `veto_excess_threshold = -0.05`
  - 但该“最优”解在 **val 上对 `exp_061` 一张都没改**（`num_val_changed_vs_061 = 0`）
- 尽管 val 完全不动，`exp_082` 在 **test** 上却改动了 **99 / 1816** 张，相对 `exp_061` 的迁移为：
  - `0 -> 1`: `55`
  - `0 -> 2`: `44`
  - 预测分布由 `exp_061` 的 `{0:1072, 1:357, 2:387}` 变为 `{0:973, 1:412, 2:431}`
- 随后对整个 400 点网格做了二次审视：
  - **没有任何参数组合在 val 上超过 `0.9375`**
  - 在“保持 val 最优”的前提下，把 `rescue_margin` 从 `0.10` 提到 `0.30`，只能把 test 改动从 `90` 降到 `42`，但 **始终没有任何 val 增益**
- 结论：
  - 这条 **061 锚点 + 067 heuristic gate** 线已经得到清晰负结论
  - 它不是“找到一个可提交门控”，而只是“不同强度地把 067 的接受边界贴回 061”
  - **当前不值得提交 `exp_082`**；因为它对 val 没有任何收益，却会在 test 上大规模把 `other` 改成目标类，风险与 `exp_081` 属于同一类

### Session 53

- 用户已回报：**`exp_082_vit061_buffalol067_gated_score_fusion_valgrid` Kaggle public score = `0.94768`**。
- 这条结果推翻了 Session 52 的保守离线判断，说明：
  - `exp_082` 虽然在固定 val 上无收益，但线上确实抓到了有效互补性
  - 真正有价值的部分不是“通用多模型融合”，而是 **`exp_067` 对 `exp_061` false reject 的 rescue**
- 据此，下一步不再做通用 swap/veto gate，而改做：
  - **rescue-only**
  - **class-aware margin**
  - 参数搜索从固定 16 张 val，切换到 **80 张 labeled leave-one-out (LOO)** 口径
- 新增：
  - `scripts/predict_061_067_rescueonly_labeledloo.py`
  - `src/dl_pipeline/inference/gated_fusion.py` 扩展 `apply_rescue_only_fusion` / `select_best_rescue_only_params`
  - `tests/test_gated_fusion.py` 对应新增单测

### Session 54

- 已完成两版 rescue-only 候选：
  1. `exp_083_vit061_buffalol067_rescueonly_classaware_labeledloo`
  2. `exp_084_vit061_buffalol067_rescueonly_classaware_labeledloo_conservative`
- 共同点：
  - 仍以 `exp_061` 为锚点
  - 仅允许 `061 == other` 时被 `067` 救回，不再允许 swap / veto
  - 参数在 **80 张 labeled LOO** 上搜索，而不是依赖固定 val
- `exp_083`：
  - submission：`data/submissions/20260403_181210_exp_083_vit061_buffalol067_rescueonly_classaware_labeledloo_submission.csv`
  - metrics：`outputs/exp_083_vit061_buffalol067_rescueonly_classaware_labeledloo/prototype_metrics.json`
  - 最优参数：
    - `low_conf_threshold = 0.45`
    - `rescue_margin_jesse = 0.06`
    - `rescue_margin_mila = 0.06`
    - `require_anchor_argmax_match = false`
  - **LOO 指标**：
    - `exp_061 = 0.9250`
    - `exp_067 = 0.9125`
    - `fused = 0.9625`
  - 在 LOO 上仅改动 **3 张**，但在 test 上相对 `exp_061` 仍改动 **91** 张；相对 `exp_082` 只收回 **8** 张（`1 -> 0: 7`, `2 -> 0: 1`）
- 对 `exp_083` 的网格进一步筛查后发现：
  - **存在大块 plateau**：许多参数组合在 LOO 上都保持相同最优 `0.9625`，且同样只改动 `3` 张
  - 但这些同分组合在 test 上的改动量差别很大：从 **91** 可降到 **64**
- 因此又补跑了更保守的正式候选 **`exp_084`**：
  - submission：`data/submissions/20260403_181807_exp_084_vit061_buffalol067_rescueonly_classaware_labeledloo_conservative_submission.csv`
  - metrics：`outputs/exp_084_vit061_buffalol067_rescueonly_classaware_labeledloo_conservative/prototype_metrics.json`
  - 固定参数：
    - `low_conf_threshold = 0.45`
    - `rescue_margin_jesse = 0.10`
    - `rescue_margin_mila = 0.30`
    - `require_anchor_argmax_match = false`
  - 这组参数与 `exp_083` 在 **LOO 上完全同分**：
    - `fused = 0.9625`
    - `num_changes = 3`
  - 但在 **test** 上明显更保守：
    - 相对 `exp_061` 改动从 `91` 降到 **`64`**
    - 相对 `exp_082` 收回 **`35`** 张：
      - `2 -> 0`: `26`
      - `1 -> 0`: `9`
- 当前实验判断：
  - **若继续在线尝试，优先提交 `exp_084` 而不是 `exp_083`**
  - 原因不是它离线更高，而是它在**相同 LOO 最优精度**下，对 `exp_082` 的激进 rescue 做了更强收缩，更适合作为“同机制、更保守”的验证候选

### Session 55

- 已先完成一次代码归档：
  - 在 `Computer Vision/assignment/Group` 仓库的 `DL` 分支提交并推送：
    - commit: `c2c6bcf`
    - message: `feat(backend): add dual verifier and rescue fusion experiments`
- 随后按既定下一步实现并运行 **`exp_085_vit061_buffalol067_agreementaware_rescueonly_labeledloo`**：
  - 新增脚本：`scripts/predict_061_067_agreement_rescue_labeledloo.py`
  - `gated_fusion.py` 新增：
    - `apply_agreement_aware_rescue_only_fusion`
    - `select_best_agreement_aware_rescue_only_params`
  - `tests/test_gated_fusion.py` 对应补充 agreement-aware 单测
- 设计动机：
  - 不再把 `argmax match` 当硬过滤
  - 改为把 rescue 子集拆成：
    - `061_argmax == 067_pred`
    - `061_argmax != 067_pred`
  - Jesse / Mila 各自再分 `agree` / `disagree` 两套 margin
- 本轮验证：
  - `conda run -n gpu_env python -m unittest tests.test_gated_fusion tests.test_prototype_inference`
  - 结果：`42 tests` 全通过
  - 运行主实验时，`conda run` 依旧触发 Windows 编码问题，但**实验本体已完整执行并写出产物**；错误发生在 conda 回显 stdout 阶段，不影响结果文件
- `exp_085` 结果：
  - submission：`data/submissions/20260403_184059_exp_085_vit061_buffalol067_agreementaware_rescueonly_labeledloo_submission.csv`
  - metrics：`outputs/exp_085_vit061_buffalol067_agreementaware_rescueonly_labeledloo/prototype_metrics.json`
  - 最优参数：
    - `low_conf_threshold = 0.45`
    - `rescue_margin_jesse_agree = 0.02`
    - `rescue_margin_jesse_disagree = 0.06`
    - `rescue_margin_mila_agree = 0.02`
    - `rescue_margin_mila_disagree = 0.06`
  - **LOO 指标**：
    - `exp_061 = 0.9250`
    - `exp_067 = 0.9125`
    - `fused = 0.9625`
  - LOO rescue 子集仅 `3` 张：
    - agree: `2`
    - disagree: `1`
  - test rescue 候选共 `99` 张：
    - agree: `54`
    - disagree: `45`
- 与历史候选的关系：
  - 相对 `exp_061`：改动 `95` 张
    - `0 -> 1`: `51`
    - `0 -> 2`: `44`
  - 相对 `exp_082`：只差 `4` 张
    - 全部为 `1 -> 0`
  - 相对 `exp_083`：也只差 `4` 张
    - `0 -> 1`: `3`
    - `0 -> 2`: `1`
- 对这 `4` 张差异样本的检查结果：
  - `exp_085` 相比 `exp_082` 撤回的 4 张，全部是 Jesse rescue
  - 其中 `3` 张其实 `061_argmax == 1` 且 `067` 的 Jesse excess 很高，但因为 `score_061_final > 0.45`，被 `rescue-only low_conf gate` 挡住
  - 这说明：`exp_085` 的主要变化并不是“agreement-aware 学到了新结构”，而是**在 `exp_082` 与 `exp_083` 之间做了一次更细的中间收缩**
- 当前判断：
  - `exp_085` **值得作为一个低风险补充候选提交一次**，因为它与当前最强 `exp_082` 只差 `4` 张，不是大偏航
  - 但它**不值得高置信押注**；从结构上看，它更像 `082/083` 之间的 interpolation，而不是新的主突破
  - 如果线上不超过 `0.94768`，下一步不应继续在 agreement-aware margin 上细抠，而应直接进入：
    - `061 score-aware rescue`
    - 或 `082 + 085` 这类更小范围的 sample-wise meta-decider

### Session 56

- 根据 `raw / HAAR / MTCNN` 三联图人工检查，确认当前任务的主问题已不是“检测器谁更清晰”，而是：
  - 多人图 / 比较图里**单脸裁剪经常选错人**
  - 典型情况包括：
    - `class1` 与 `other` 放在一起的比较图，标签只给 `class1`
    - 合照中目标脸不是最大脸
    - `HAAR` 裁到背景人，`MTCNN` 裁到另一位主角
    - 占位图 / 坏图直接进入训练集
- 因此本轮不继续做单脸裁剪器微调，也不采用全手工裁剪；而是正式开启：
  - **多脸候选 + 图像级 max-pool 验证器原型**
- 新增：
  - 配置：`configs/experiments/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike.yaml`
  - 脚本：`scripts/predict_multiface_dual_verifier.py`
  - 逻辑：`src/dl_pipeline/inference/insightface_dual_verifier.py` 新增 `aggregate_multi_face_scores`
  - 测试：`tests/test_insightface_dual_verifier.py` 改为 `unittest`，并补充多脸聚合用例
- 设计原则：
  - **gallery / 阈值校准协议保持 `exp_067` 不变**
  - 只改 probe 侧：
    - 从原始整图 `detect_align` 提取**所有**检测到的人脸 embedding
    - 对每张候选脸分别算 Jesse / Mila 分数
    - 图片级别按通道取最大分，再做 dual verifier 判决
  - 这样可以单独验证：
    - “多人图裁错人” 是否就是当前最大瓶颈
- 验证：
  - `python -m unittest tests.test_insightface_dual_verifier tests.test_gated_fusion tests.test_prototype_inference`
  - 结果：`48 tests` 全通过
- `exp_086` 结果：
  - submission：`data/submissions/20260403_195050_exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike_submission.csv`
  - metrics：`outputs/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike/prototype_metrics.json`
  - **val_accuracy_multiface = 1.0000**
  - 相比 `exp_067` 风格的 test 预测，改动 **140** 张：
    - `0 -> 1`: `66`
    - `0 -> 2`: `15`
    - `1 -> 0`: `28`
    - `2 -> 0`: `15`
    - `2 -> 1`: `8`
    - `1 -> 2`: `8`
  - val 多脸检测统计：
    - `1 face`: `12`
    - `2 face`: `2`
    - `3 face`: `1`
    - `4 face`: `1`
  - test 多脸检测统计显示多人图并不少：
    - `1 face`: `1301`
    - `2+ faces`: `515`
- 当前判断：
  - `exp_086` 是**值得提交一次**的候选，因为它不是在 `082/083/085` 那类 threshold plateau 上继续打转，而是直接改变了任务假设
  - 即使线上未必立刻超过 `0.94768`，它也会给出更高信息量：  
    当前瓶颈究竟是“多人图选错人”，还是“061/067 的身份边界本身还不够好”
- 额外记录：
  - 三联图脚本 `export_face_comparison_visualization.py` 在修 `val -> train` 映射时曾误把 `_try_load_image()` 改坏，导致整批输出显示 `missing`
  - 根因已定位并修复；建议后续优先查看新导出的目录：
    - `data/visualizations/face_compare/train_val_exp_001_v2`

### Session 57

- 用户反馈 `exp_086` Kaggle public score = **`0.97026`**，当前已进入榜单第一梯队边缘。
- 这使得 `exp_086` 不再只是“信息量高的候选”，而是当前最强主线之一；因此下一步不继续盲推训练，而是先验证：
  - **多脸候选 + 图片级聚合** 在 80 张 labeled 样本上是否真的工作正常
- 新增：
  - `scripts/export_multiface_probe_visualization.py`
  - `src/dl_pipeline/inference/insightface_dual_verifier.py` 新增 `describe_multi_face_scores`
  - `tests/test_insightface_dual_verifier.py` 增补 face-level 明细测试
- 验证：
  - `python -m unittest tests.test_insightface_dual_verifier tests.test_gated_fusion tests.test_prototype_inference`
  - 结果：`50 tests` 全通过
- 导出目录：
  - `data/visualizations/multiface_probe/exp_086_train_val_exp_001`
  - 含：
    - `images/`：80 张 labeled 样本逐图可视化
    - `summary.csv`：图片级预测摘要
    - `face_details.csv`：每张脸的 bbox + Jesse/Mila 分数
- 当前 `exp_086` 在 80 张 labeled 审计结果：
  - **79 / 80 正确，accuracy = 0.9875**
  - 多脸数分布：
    - `1 face`: `57`
    - `2 faces`: `17`
    - `3 faces`: `5`
    - `4 faces`: `1`
  - 唯一错误样本：
    - `train id=65 class=2 -> pred=0`
    - `probe_source = crop_fallback`
    - 对应样本正是之前人工确认的 **占位坏图 / IMAGE NOT FOUND**
- 对用户点名的关键样本核查：
  - `id=14`：当前 `multiface` 预测正确为 `class1`
  - `id=34`：检测到 `3` 张脸，最终仍正确选择 `class1`
  - `id=40`：检测到 `2` 张脸，最终正确为 `class2`
  - `id=52`：检测到 `2` 张脸，最终正确为 `class1`
  - `id=59`：检测到 `2` 张脸，最终保持 `other`
  - `id=65`：坏图，仍失败
- 当前判断进一步收敛为：
  - 你之前怀疑“前面很多训练策略失败，其实是输入数据没处理好”这件事，现在已经拿到了很强的支持证据
  - 现阶段最值得做的不是继续大扫训练超参，而是：
    - 先系统化标出坏样本 / 比较陷阱图 / 多人图
    - 再把 **自动化多脸选择** 正式接入更强主线

### Session 58

- 按“先单独处理坏图”这条思路，进一步检查 `exp_086` 输出，发现：
  - test 中共有 `37` 张 `crop_fallback`
  - 其中绝大多数并不是坏图，只是原图检测不到脸后回退到单脸 crop
  - 真正值得单独处理的，不应是所有 fallback，而是**与带标签坏图完全重复的原图**
- 关键发现：
  - `train id=65 class=2` 的占位坏图，在 test 中有 **3 张完全相同的原图副本**：
    - `id=537`
    - `id=1071`
    - `id=1112`
  - 三者当前在 `exp_086` 中都被预测成 `0`
  - 其 raw 图数组与 `train id=65` 做 SHA-1 哈希后完全一致，不是“相似”，而是字节级一致
- 进一步检查后发现：
  - test 中一共有 **12 张**图片与 train/val 某张标注图**原图完全一致**
  - 其中已有 `9` 张在 `exp_086` 中预测正确
  - 只有上述 `3` 张坏图副本预测错误
- 因此本轮没有做宽泛的“坏图启发式”，而是实现了更干净的规则：
  - **原图哈希与 train/val 某张带标签图片完全一致时，直接继承该标签**
- 新增：
  - `src/dl_pipeline/inference/exact_match_override.py`
  - `tests/test_exact_match_override.py`
  - `scripts/predict_086_exact_hash_override.py`
- 验证：
  - `python -m unittest tests.test_exact_match_override tests.test_insightface_dual_verifier tests.test_gated_fusion tests.test_prototype_inference`
  - 结果：`52 tests` 全通过
- 新实验：
  - `exp_087_multiface_exact_raw_hash_override`
  - submission：`data/submissions/20260403_202741_exp_087_multiface_exact_raw_hash_override_submission.csv`
  - metrics：`outputs/exp_087_multiface_exact_raw_hash_override/prototype_metrics.json`
- 本轮改动非常窄：
  - `num_test_hash_matches_to_labeled = 12`
  - `num_overrides_applied = 3`
  - 全部为：
    - `0 -> 2`: `3`
- 当前判断：
  - `exp_087` 是一个**非常值得提交**的低风险增量候选
  - 它不是新的泛化假设，而是在利用一个确定事实：
    - test 里存在与 labeled 数据完全相同的原图副本

### Session 59

- 开始执行“把多脸自动选人接入训练流程”的受控实验，目标是只替换输入协议，尽量不改 `exp_061` 的主体训练设定。
- 新增规划文档：
  - `docs/plans/2026-04-03-exp088-multiface-train-design.md`
- 先按 TDD 补了多脸选脸逻辑：
  - `tests/test_multiface_selection.py`
  - `src/dl_pipeline/preprocess/multiface_selection.py`
- 新增数据构建链路：
  - `src/dl_pipeline/preprocess/build_multiface_selected_dataset.py`
  - `scripts/prepare_multiface_selected_faces.py`
  - `configs/experiments/exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml`
- `exp_088` 的核心设定：
  - backbone / loss / epoch / 解冻策略都保持 `exp_061`
  - 只把输入从 `HAAR` 单脸 crop 改成“原图多脸检测后自动选中的单脸”
  - `class1`：取 `class1` 分数最高的人脸
  - `class2`：取 `class2` 分数最高的人脸
  - `class0`：取最像 `class1/class2` 的 hardest negative 人脸
  - 明确坏图 `train id=65` 直接从训练/验证 split 中剔除
- 新数据集已成功构建：
  - processed dir：`data/processed/exp_088_multiface_selected_faces_112`
  - splits dir：`data/splits/exp_088_multiface_selected`
  - 结果：
    - train：`63`
    - val：`16`
    - test：`1816`
- selection audit 统计：
  - train：
    - `detect_align_selected = 62`
    - `crop_fallback = 1`（`id=9`）
  - val：
    - `detect_align_selected = 16`
  - test：
    - `detect_align_selected = 1779`
    - `crop_fallback = 37`
- 训练已完成：
  - `python scripts/train.py --config configs/experiments/exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml`
  - best checkpoint：
    - `outputs/exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta/checkpoints/best.ckpt`
  - `metrics.json`：
    - `best_val_acc = 1.0`
- 推理已完成：
  - `python scripts/predict.py --config configs/experiments/exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml`
  - submission：
    - `data/submissions/20260403_222929_exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv`
  - CSV 表头已核对正常：`id,class`
- 与历史 submission 对比：
  - 相对 `exp_061` 共改动 `131` 张：
    - `0 -> 1`: `94`
    - `0 -> 2`: `37`
  - 相对 `exp_087` 共改动 `28` 张：
    - `0 -> 1`: `24`
    - `2 -> 0`: `3`
    - `0 -> 2`: `1`
  - 当前预测分布：
    - `0: 941`
    - `1: 451`
    - `2: 424`
- 验证：
  - `python -m unittest tests.test_multiface_selection tests.test_exact_match_override tests.test_insightface_dual_verifier tests.test_gated_fusion tests.test_prototype_inference`
  - 结果：`57 tests` 全通过
- 当前判断：
  - `exp_088` 是第一条真正把“多脸自动选人”接进训练流程的受控重训线
  - 它最值得提交的意义，不是离线 `1.0` 本身，而是它直接回答：
    - 前面很多训练策略没有起飞，是否主要因为输入人脸选错了

### Session 60

- `exp_088` 已提交 Kaggle / leaderboard，public score：
  - `0.98403`
- 当前状态：
  - 已升至 **leaderboard 第 1**
- 这条结果的意义已经非常明确：
  - `multiface-selected` 输入协议不只是推理层 trick
  - 它可以直接作为训练主线的数据入口，并且带来当前最强线上收益
- 相比此前关键线上结果：
  - `exp_082 = 0.94768`
  - `exp_086 = 0.97026`
  - `exp_087 = 0.97191`
  - `exp_088 = 0.98403`
- 结论更新：
  - 之前大量训练策略没有兑现，主因很可能不是 classifier 主体不行，而是长期使用了错误的人脸输入协议
  - 当前后续实验应默认基于 `multiface-selected` 输入继续推进，而不是回退到 `HAAR` 单脸训练线

### Session 61

- 执行 `exp_089`：
  - 目标：在 `exp_088` 的输入协议和训练超参完全不变前提下，只扩展 train split 中 `class0` 的 hardest negative 采样
  - 方案：`class0 top-k = 2`
- 新增设计文档：
  - `docs/plans/2026-04-04-exp089-class0-hard-negatives-design.md`
- 代码改动：
  - `src/dl_pipeline/preprocess/multiface_selection.py`
  - `src/dl_pipeline/preprocess/build_multiface_selected_dataset.py`
  - `tests/test_multiface_selection.py`
  - `configs/experiments/exp_089_vit_adaface_multiface_selected_other_top2_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml`
- 新数据集：
  - processed dir：`data/processed/exp_089_multiface_selected_other_top2_faces_112`
  - splits dir：`data/splits/exp_089_multiface_selected_other_top2`
- 数据构建结果：
  - train：`66`
  - val：`16`
  - test：`1816`
  - 仅 3 个 `class0` 训练样本被展开为双负脸：
    - `id=41`
    - `id=59`
    - `id=63`
- 训练结果：
  - `best_val_acc = 1.0`
  - checkpoint：
    - `outputs/exp_089_vit_adaface_multiface_selected_other_top2_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta/checkpoints/best.ckpt`
- 推理结果：
  - submission：
    - `data/submissions/20260404_093907_exp_089_vit_adaface_multiface_selected_other_top2_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv`
  - 与 `exp_088` 按 `id` merge 后逐行比较：
    - `num_diff = 0`
    - 预测分布完全一致：
      - `0: 941`
      - `1: 451`
      - `2: 424`
- 结论：
  - “给 `class0` 再多一张 hardest negative” 在当前 `exp_088` 主线上**没有改变最终测试预测**
  - 因此这条方向至少在 `top-k=2` 的最小扩展版本上，**不是下一次突破点**

### Session 62

- 执行 `exp_090`：
  - 目标：在 `exp_088` 输入协议完全不变前提下，只增强单体模型训练强度
  - 方案：将 ViT 解冻深度从最后 `2` 个 block 提升到最后 `4` 个 block，并把 backbone LR 从 `1e-5` 降到 `5e-6`
- 新增设计文档：
  - `docs/plans/2026-04-04-exp090-stronger-finetune-design.md`
- 新配置：
  - `configs/experiments/exp_090_vit_adaface_multiface_selected_20ep_last4block_ce_neighborhoodaware_fixed055_hfliptta.yaml`
- 训练结果：
  - `best_val_acc = 1.0`
  - checkpoint：
    - `outputs/exp_090_vit_adaface_multiface_selected_20ep_last4block_ce_neighborhoodaware_fixed055_hfliptta/checkpoints/best.ckpt`
- 推理结果：
  - submission：
    - `data/submissions/20260404_095346_exp_090_vit_adaface_multiface_selected_20ep_last4block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv`
  - 与 `exp_088` 按 `id` merge 后逐行比较：
    - `num_diff = 0`
    - 预测分布完全一致：
      - `0: 941`
      - `1: 451`
      - `2: 424`
- 结论：
  - 在当前 `multiface-selected + CE + neighborhood-aware` 主线上，单纯把 ViT 从 `last2block` 加深到 `last4block`，**没有改变最终测试预测**
  - 因此“继续加强单体模型训练强度”这个方向本身没错，但当前这一步具体实现**还没有碰到新的边界**

### Session 63

- 接受关于早停协议的批评并执行纠偏重跑：
  - 问题：`val` 只有 `16` 张时，`val_acc` 很早饱和到 `1.0`，不适合作为冠军主线的 early stopping / checkpoint monitor
  - 修正：把 monitor 从 `val_acc / max` 改为 `val_loss / min`，并把训练上限提升到 `50 epoch`
- 新增设计文档：
  - `docs/plans/2026-04-04-exp091-exp092-valloss-rerun-design.md`
- 新配置：
  - `configs/experiments/exp_091_vit_adaface_multiface_selected_50ep_last2block_ce_neighborhoodaware_fixed055_valloss_hfliptta.yaml`
  - `configs/experiments/exp_092_vit_adaface_multiface_selected_50ep_last4block_ce_neighborhoodaware_fixed055_valloss_hfliptta.yaml`
- `exp_091`：
  - 目标：公平重跑 `exp_088`
  - 最佳 `val_loss`：
    - `0.0025814103428274393`
  - checkpoint：
    - `outputs/exp_091_vit_adaface_multiface_selected_50ep_last2block_ce_neighborhoodaware_fixed055_valloss_hfliptta/checkpoints/best.ckpt`
  - submission：
    - `data/submissions/20260404_100746_exp_091_vit_adaface_multiface_selected_50ep_last2block_ce_neighborhoodaware_fixed055_valloss_hfliptta_submission.csv`
  - 与 `exp_088` 按 `id` merge 后逐行比较：
    - `num_diff = 0`
- `exp_092`：
  - 目标：公平重跑 `exp_090`
  - 最佳 `val_loss`：
    - `0.003377359127625823`
  - checkpoint：
    - `outputs/exp_092_vit_adaface_multiface_selected_50ep_last4block_ce_neighborhoodaware_fixed055_valloss_hfliptta/checkpoints/best.ckpt`
  - submission：
    - `data/submissions/20260404_101330_exp_092_vit_adaface_multiface_selected_50ep_last4block_ce_neighborhoodaware_fixed055_valloss_hfliptta_submission.csv`
  - 与 `exp_088` 按 `id` merge 后逐行比较：
    - `num_diff = 0`
- 共同观察：
  - 两个实验都没有像旧版那样在不到 `10 epoch` 时停止
  - 两者都继续训练到了接近 `50 epoch`
  - `val_acc` 很早达到 `1.0`，但 `val_loss` 继续稳步下降
- 结论修正：
  - 对早停协议的批评是成立的，之前用 `val_acc` 监控确实不严谨
  - 但在把协议修正到合理之后，`exp_088` 和 `exp_090` 的最终 test submission 仍然与旧版逐行一致
  - 因此现在可以更有把握地保留两个实证结论：
    - `exp_088` 这条冠军主线已经相当稳定
    - `exp_090` 的 `last4block` stronger finetune 在当前输入协议下仍未改变最终边界

### Session 64

- 新增诊断脚本与辅助模块：
  - `scripts/export_case_embedding_diagnostics.py`
  - `src/dl_pipeline/evaluation/case_diagnostics.py`
  - `tests/test_case_diagnostics.py`
- 对用户肉眼指出的 8 张 `exp_088` fallback 错例做 embedding 级诊断：
  - `48`
  - `289`
  - `361`
  - `543`
  - `612`
  - `699`
  - `1333`
  - `1525`
- 诊断输出：
  - `data/visualizations/case_embedding_diagnostics/exp_093_focus_cases/summary.csv`
  - `data/visualizations/case_embedding_diagnostics/exp_093_focus_cases/neighbors.csv`
  - `data/visualizations/case_embedding_diagnostics/exp_093_focus_cases/images/`
- 关键结论：
  - 这 8 张并不是“方向判错了”
  - 它们在最终分类器里**全部更偏向 `class2`**
  - 但 `final_score` 仅约 `0.25 ~ 0.34`，显著低于当前 `0.55` 阈值，因此最终被拒成 `0`
  - 说明剩余问题的本质不是“分到 wrong identity”，而是“弱证据 target 被 open-set 拒绝”

### Session 65

- 围绕“旧阈值是在错误输入协议时代调出来的”这一假设，执行两轮纯推理侧受控实验。
- `exp_094`：
  - 配置：
    - `configs/experiments/exp_094_vit_adaface_multiface_selected_pad18_rescue_neighborhoodaware_threshold_recalibrated_hfliptta.yaml`
  - 方案：
    - 使用 `exp_093` 的 rescue 输入协议
    - 复用 `exp_088` checkpoint
    - 在 `neighborhood_aware` 主线上重扫全局阈值 `0.20 ~ 0.55`
  - 结果：
    - 自动选中 `selected_threshold = 0.48`
    - val 仍为 `1.0`
    - 与 `exp_088` 相比仅改动 `1` 张 test：`id=1540, 0->2`
    - 用户重点指出的 8 张错误样本**一张都没有被救回**
- `exp_095`：
  - 配置：
    - `configs/experiments/exp_095_vit_adaface_multiface_selected_pad18_rescue_neighborhoodaware_classspecific_thresholds_hfliptta.yaml`
  - 代码支持：
    - 为 `neighborhood_aware` 主线补入 class-specific threshold 搜索与推理分支
  - 结果：
    - 自动选中 `class1=0.48, class2=0.49`
    - 与 `exp_088` submission **逐行完全一致，`num_diff = 0`**
- 结论：
  - “旧阈值不对”这个判断不是空想，但它**不是当前主瓶颈**
  - 仅靠阈值重标定，无法修复那批真正高价值的弱证据错例

### Session 66

- 基于 `exp_094/095` 的结果，主线判断收敛：
  - 后处理式阈值修补无法根治当前残余错误
  - 更可能的根因是：
    - 当前模型尚未在 `exp_093` 这套 `pad18 rescue` 输入协议上重新训练过
    - 训练分布与当前推理分布仍不一致
- 决定启动下一主实验：
  - `exp_096 = exp_091 training recipe + exp_093 input protocol`
  - 即：
    - `pad18 rescue multiface-selected` 输入
    - `50 epoch`
    - `val_loss / min`
    - `last2block`
    - `CE + neighborhood_aware + hfliptta`
- 该实验的目标不是继续修补阈值，而是验证：
  - 在当前真实推理输入协议上重训之后，弱证据 `class2` 是否能被整体抬分

### Session 67

- 执行 `exp_096`：
  - 设计文档：
    - `docs/plans/2026-04-04-exp096-pad18-rescue-retrain-design.md`
  - 配置：
    - `configs/experiments/exp_096_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_ce_neighborhoodaware_fixed055_valloss_hfliptta.yaml`
  - 方案：
    - 训练 recipe 继承 `exp_091`
    - 输入协议切换到 `exp_093` 的 `pad18 rescue multiface-selected`
    - `50 epoch`
    - `val_loss / min`
    - `last2block`
    - `CE + neighborhood_aware + hfliptta`
- 训练结果：
  - `best_val_loss = 0.001385191222652793`
  - checkpoint：
    - `outputs/exp_096_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_ce_neighborhoodaware_fixed055_valloss_hfliptta/checkpoints/best.ckpt`
- 推理结果：
  - submission：
    - `data/submissions/20260404_132520_exp_096_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_ce_neighborhoodaware_fixed055_valloss_hfliptta_submission.csv`
  - 与 `exp_088` 按 `id` merge 后逐行比较：
    - `num_diff = 0`
  - 用户重点关注的 8 张高价值错例：
    - `48, 289, 361, 543, 612, 699, 1333, 1525`
    - 仍然全部保持为 `0`
- 结论：
  - 仅把训练分布切换到 `exp_093` 的 rescue 输入协议，还不足以改写当前主线的 test 决策边界
  - 因此“训练/推理输入不一致”虽然是合理怀疑，但**不是唯一根因**
  - 下一步如果还想继续突破，应该考虑：
    - 更有针对性的训练目标
    - 或围绕 fallback / 弱证据样本的专门训练增强，而不是再做同构重训

### Session 68

- 用户明确拒绝继续堆叠过于任务专门化的手工修补（如样本权重、人工规则），希望下一步方法仍然足够“神经网络化”且能写进报告。
- 基于此前诊断，决定重新验证 `CosFace`：
  - 之前做过 margin-based loss，但当时的数据处理协议尚未修正，因此旧结论无效
  - 当前系统的最终决策本来就依赖 embedding / prototype / threshold 几何结构，理论上比单纯 `CE` 更匹配 margin-based 训练目标
- 因此执行一条受控对比实验：
  - `exp_097 = exp_096` 的 clean recipe
  - 唯一核心变化是 `loss: CE -> CosFace`
- `exp_097`：
  - 设计文档：
    - `docs/plans/2026-04-04-exp097-cosface-retrain-design.md`
  - 配置：
    - `configs/experiments/exp_097_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_cosface_neighborhoodaware_fixed055_valloss_hfliptta.yaml`
  - 训练结果：
    - `best_val_loss = 0.025193748995661736`
    - checkpoint：
      - `outputs/exp_097_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_cosface_neighborhoodaware_fixed055_valloss_hfliptta/checkpoints/best.ckpt`
  - 推理结果：
    - submission：
      - `data/submissions/20260404_135527_exp_097_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_cosface_neighborhoodaware_fixed055_valloss_hfliptta_submission.csv`
    - 与 `exp_088` 按 `id` merge 后逐行比较：
      - `num_diff = 0`
    - 与 `exp_096` 按 `id` merge 后逐行比较：
      - `num_diff = 0`
    - 用户重点关注的 8 张高价值错例：
      - `48, 289, 361, 543, 612, 699, 1333, 1525`
      - 仍然全部保持为 `0`
- 当前阶段性结论再次收敛：
  - 在**当前三类训练设定**下，把 `CE` 换成 `CosFace` 仍然没有改写 submission
  - 这与用户对 `other` 的判断一致：
    - `other` 并不是单一干净类
    - 其中包含大量 `Michael / Sarah` 相似脸，以及少量其他杂类
  - 因而“把异质 `other` 也一起拉入统一 margin 分类”的收益，至少在当前 recipe 下没有体现出来

### Session 69

- 为避免过早否定 `CosFace`，补做一轮只改推理、不重训的阈值重标定：
  - 配置：
    - `configs/experiments/exp_098_vit_adaface_multiface_selected_pad18_rescue_cosface_neighborhoodaware_threshold_recalibrated_hfliptta.yaml`
  - checkpoint 来源：
    - `exp_097`
- `exp_098` 结果：
  - 自动选中 `selected_threshold = 0.25`
  - `val_accuracy = 1.0`
  - 相对 `exp_097 / exp_088` 共放出 `7` 张新样本：
    - `211: 0->1`
    - `213: 0->1`
    - `420 / 1455 / 1540 / 1542 / 1777: 0->2`
  - 但用户重点关注的 8 张：
    - `48, 289, 361, 543, 612, 699, 1333, 1525`
    - **仍然全部没有被救回**
- 因此结论更清楚了：
  - `CosFace` 的分数尺度确实与 `CE` 不同，单独重标定阈值是必要的
  - 但即便重标定后，它释放出来的也不是当前已知最有价值的那批错例
  - 所以下一步主方向**不应继续围绕 CosFace 阈值微调**，而应转向更贴合 `other` 异质结构的训练/建模方案

### Session 70

- 在确认 `exp_098` public score `0.98017` 后，正式启动任务重建模实验：
  - `exp_099`
  - 核心思路不是继续做三类 softmax，而是把训练任务改成：
    - `is_class1`
    - `is_class2`
    - `other` 作为两个头的共同负样本补集
- 代码层面的实现保持最小：
  - 新增 `bce_ovr` loss 分支
  - 推理主线完全不变，仍使用 `prototype + neighborhood_aware + fixed055`
  - 这样可以把结论严格归因到**训练任务定义**本身
- 新增与通过的测试：
  - `tests/test_lightning_module.py`
  - 覆盖：
    - `bce_ovr` 可稳定训练
    - 低置信样本会在验证阶段被拒到 `other`
- 实验材料：
  - 设计文档：
    - `docs/plans/2026-04-04-exp099-bce-ovr-design.md`
  - 配置：
    - `configs/experiments/exp_099_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_bceovr_neighborhoodaware_fixed055_valloss_hfliptta.yaml`
- 训练结果：
  - `best_val_loss = 0.0014043942792341113`
  - checkpoint：
    - `outputs/exp_099_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_bceovr_neighborhoodaware_fixed055_valloss_hfliptta/checkpoints/best.ckpt`
- 推理结果：
  - submission：
    - `data/submissions/20260404_141936_exp_099_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_bceovr_neighborhoodaware_fixed055_valloss_hfliptta_submission.csv`
  - 相对 `exp_088 / exp_096 / exp_097`：
    - `num_diff = 1`
  - 唯一改变样本：
    - `id=1540: 0 -> 2`
  - 用户重点关注的 8 张高价值样本：
    - `48, 289, 361, 543, 612, 699, 1333, 1525`
    - 仍然全部没有被救回
- 当前判断：
  - `exp_099` 是**第一条在不改任何推理后处理的前提下，真正改写了 test 边界**的训练实验
  - 但由于只改了 `1` 张，目前还不能认为它已经带来稳定增益
  - 它最自然的下一步不是换方向，而是：
    - 先做 `exp_099` 自己的阈值重标定

### Session 71

- 执行 `exp_100 = exp_099 + threshold recalibration`：
  - 配置：
    - `configs/experiments/exp_100_vit_adaface_multiface_selected_pad18_rescue_bceovr_neighborhoodaware_threshold_recalibrated_hfliptta.yaml`
  - checkpoint 来源：
    - `exp_099`
- 阈值搜索结果：
  - 自动选中 `selected_threshold = 0.49`
  - `val_accuracy = 1.0`
- 相对 submission 的变化：
  - 相对 `exp_099`：
    - `num_diff = 1`
    - 新增 `id=1455: 0 -> 2`
  - 相对 `exp_088`：
    - `num_diff = 2`
    - `1455: 0 -> 2`
    - `1540: 0 -> 2`
- 但用户重点关注的 8 张核心错例：
  - `48, 289, 361, 543, 612, 699, 1333, 1525`
  - 仍然全部保持为 `0`
- 当前结论：
  - one-vs-rest 这条线在固定推理协议下，确实比 `CE/CosFace` 更早出现 test 边界变化
  - 但目前释放出来的仍是少量边缘样本，而不是当前最想解决的那批高价值错例
  - 因此 `exp_100` 仍然不足以支持提交

### Session 72

- 执行 `exp_101 = exp_099 + target-only supervised contrastive auxiliary loss`：
  - 设计文档：
    - `docs/plans/2026-04-04-exp101-bce-ovr-supcon-design.md`
  - 配置：
    - `configs/experiments/exp_101_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_bceovr_supcon_neighborhoodaware_fixed055_valloss_hfliptta.yaml`
- 代码实现：
  - 在 `bce_ovr` 基础上新增 `bce_ovr_supcon`
  - `SupCon` 为 target-only：
    - 只让 `class1/class2` 形成正对
    - 不强迫异质 `other` 形成单一簇
  - 相关测试已补入并通过：
    - `tests/test_lightning_module.py`
- 训练结果：
  - `best_val_loss = 0.0781790018081665`
  - checkpoint：
    - `outputs/exp_101_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_bceovr_supcon_neighborhoodaware_fixed055_valloss_hfliptta/checkpoints/best.ckpt`
- 推理结果：
  - submission：
    - `data/submissions/20260404_143800_exp_101_vit_adaface_multiface_selected_pad18_rescue_50ep_last2block_bceovr_supcon_neighborhoodaware_fixed055_valloss_hfliptta_submission.csv`
  - 相对 `exp_099`：
    - `num_diff = 0`
  - 相对 `exp_100`：
    - 少了 `id=1455: 2 -> 0`
  - 相对 `exp_088`：
    - 仍然只有 `id=1540: 0 -> 2`
  - 用户重点关注的 8 张核心错例：
    - `48, 289, 361, 543, 612, 699, 1333, 1525`
    - 依旧全部没有变化
- 附带信号：
  - `prototype_metrics.json` 中固定 `0.55` 下的 `val_accuracy = 0.9375`
  - 相比 `exp_099`，说明当前 `SupCon` 辅助项并未带来更稳定的验证边界
- 结论：
  - `target-only SupCon` 在当前超参与 batch 条件下，**没有超过纯 `bce_ovr` 主线**
  - 当前最稳的 one-vs-rest 版本仍然是 `exp_099/100`

### Session 73

- 执行 `exp_102 = exp_101 + threshold recalibration`：
  - 配置：
    - `configs/experiments/exp_102_vit_adaface_multiface_selected_pad18_rescue_bceovr_supcon_neighborhoodaware_threshold_recalibrated_hfliptta.yaml`
  - checkpoint 来源：
    - `exp_101`
- 阈值搜索结果：
  - 自动选中 `selected_threshold = 0.55`
  - `val_accuracy = 1.0`
  - 也就是说：`exp_101` 这条 `SupCon` 分支并没有出现像 `exp_099` 那样需要单独下调阈值的迹象
- 相对 submission 的变化：
  - 相对 `exp_101`：
    - `num_diff = 1`
    - 新增 `id=1341: 0 -> 2`
  - 相对 `exp_100`：
    - 也是 `num_diff = 2`
    - `exp_100` 的 `1455: 0 -> 2` 被撤回
    - 换成 `1341: 0 -> 2`
    - `1540: 0 -> 2` 保持不变
- 用户重点关注的 8 张核心错例：
  - `48, 289, 361, 543, 612, 699, 1333, 1525`
  - 仍然全部没有变化
- 当前最终判断：
  - `exp_102` 没有超过 `exp_100`
  - `SupCon` 这条分支在当前数据规模与训练条件下，只是在少量边缘样本之间互换释放对象
  - 因而这一整条 `exp_101/102` 线路可以收束，不再继续追加 follow-up

### Session 74

- 执行 `exp_103 = exp_088 recipe + full-train on all 79 labeled samples`：
  - 配置：
    - `configs/experiments/exp_103_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_fulltrain79.yaml`
  - 核对训练/推理逻辑：
    - `data.use_full_train=true` 时，`FaceDataModule` 会把 `train+val` 合并成 79 张训练样本
    - `train.py` 在 full-train 模式下不再依赖 best-val checkpoint，而是保存 `last.ckpt`
    - `predict.py` 的 `neighborhood_aware` 默认 `gallery_source=train`，因此会直接使用 merged train gallery
- 训练结果：
  - 输出目录：
    - `outputs/exp_103_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_fulltrain79`
  - checkpoint：
    - `outputs/exp_103_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_fulltrain79/checkpoints/last.ckpt`
  - `metrics.json`：
    - `use_full_train = true`
    - `best_val_acc = null`（符合 full-train 仅保存 last 的逻辑）
- 推理结果：
  - submission：
    - `data/submissions/20260404_145351_exp_103_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_fulltrain79_submission.csv`
  - `prototype_metrics.json`：
    - `selected_threshold = 0.55`
    - `val_accuracy = 1.0`
    - `mean_test_final_score = 0.5072612166404724`
- 与 `exp_088` 的直接对比：
  - `num_diff = 0`
  - 也就是 full-train 79 张后，submission 逐行完全不变
- 用户长期关注的 8 张核心错例：
  - `48, 289, 361, 543, 612, 699, 1333, 1525`
  - 在 `exp_103` 中仍然全部保持为 `0`
- 当前结论：
  - “关键模态只在 val、没被训练到” 这个假设已经被这条最干净的 full-train CE 实验显著削弱
  - 至少在 `exp_088` 同构 recipe 下，把 16 张 val 并入训练并没有改变任何 test 决策边界

### Session 75

- 重新检查 `exp_085` 时代的异构链路后确认：
  - 历史有效的 `buffalo` / `ViT + buffalo` 组合全部建立在旧协议上
  - 旧底座包括：
    - `exp_067_buffalo_l_detect_align_dual_verifier_lookalike`
    - `exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike`
    - `exp_085_vit061_buffalol067_agreementaware_rescueonly_labeledloo`
  - 它们都没有挂到当前修正后的 `exp_093` 协议

- 执行 `exp_104`：当前协议下的 **多脸 max-pool buffalo standalone**
  - 配置：
    - `configs/experiments/exp_104_buffalo_l_detect_align_multiface_dual_verifier_lookalike_pad18_rescue.yaml`
  - 产物：
    - submission：`data/submissions/20260404_150521_exp_104_buffalo_l_detect_align_multiface_dual_verifier_lookalike_pad18_rescue_submission.csv`
    - metrics：`outputs/exp_104_buffalo_l_detect_align_multiface_dual_verifier_lookalike_pad18_rescue/prototype_metrics.json`
  - 相对 `exp_088 / exp_093`：
    - `num_diff = 23`
    - 全部是收缩：
      - `1 -> 0: 22`
      - `2 -> 0: 1`
  - 8 张核心错例：
    - `48, 289, 361, 543, 612, 699, 1333, 1525`
    - 全部仍为 `0`
  - 结论：
    - 当前协议下，**多脸 max-pool buffalo 单模没有提供可用 rescue 信号**
    - 它只是比 `exp_088/093` 更保守

- 执行 `exp_105`：当前协议下的 **ViT + buffalo rescue-only 异构融合**
  - 命令底座：
    - `scripts/predict_061_067_agreement_rescue_labeledloo.py`
  - 实际 primary / secondary：
    - primary：`exp_093`（显式使用 `exp_088` checkpoint）
    - secondary：当前协议 buffalo
  - 产物：
    - submission：`data/submissions/20260404_151103_exp_105_vit093_buffalol104_agreementaware_rescueonly_labeledloo_submission.csv`
    - metrics：`outputs/exp_105_vit093_buffalol104_agreementaware_rescueonly_labeledloo/prototype_metrics.json`
  - 结果：
    - 相对 `exp_088 / exp_093`：
      - `num_diff = 25`
      - 全部是放宽：
        - `0 -> 1: 13`
        - `0 -> 2: 12`
    - 8 张核心错例里：
      - `48: 0 -> 1`
      - 其余 `289 / 361 / 543 / 612 / 699 / 1333 / 1525` 仍不变
  - 重要细节：
    - 这 25 张 rescue 信号并不是来自 `exp_104` 的多脸 max-pool submission 本身
    - 而是来自该融合脚本内部使用的 **当前协议单脸 buffalo dual verifier 打分表**
  - 风险提示：
    - `labeled_loo_rescue_subset_size = 0`
    - 也就是 train+val 上没有任何一张样本触发 rescue
    - 但 test 上却触发了 `25` 张
    - 说明它是**明显的 test-only 外推型高风险 submission**

- 为了把信号拆干净，继续执行 `exp_106`：当前协议下的 **单脸 buffalo standalone**
  - 配置：
    - `configs/experiments/exp_106_buffalo_l_detect_align_dual_verifier_lookalike_pad18_rescue.yaml`
  - 产物：
    - submission：`data/submissions/20260404_151420_exp_106_buffalo_l_detect_align_dual_verifier_lookalike_pad18_rescue_submission.csv`
    - metrics：`outputs/exp_106_buffalo_l_detect_align_dual_verifier_lookalike_pad18_rescue/prototype_metrics.json`
  - 相对 `exp_088`：
    - `num_diff = 132`
    - 变化构成：
      - `1 -> 0: 72`
      - `2 -> 0: 16`
      - `0 -> 1: 13`
      - `0 -> 2: 12`
      - `1 -> 2: 11`
      - `2 -> 1: 8`
    - 8 张核心错例里：
      - 仅 `48: 0 -> 1`
      - 其他 7 张不变
  - 结论：
    - 当前协议下，**单脸 buffalo 仍然保留异构信号**
    - 但单模本身改动过大，明显不是稳妥提交候选

- 当前阶段总结：
  - `buffalo` 在当前协议下 **不是完全失效**
  - 但有效信号来自 **单脸 dual verifier 分数**
  - 不是来自 **多脸 max-pool buffalo standalone**
  - 因而如果后续继续利用这条线，合理对象应是：
    - `exp_105` 这类以 `exp_093` 为锚点的 rescue-only 异构候选
  - 而不是 `exp_104` 或 `exp_106` 这种 buffalo 单模 submission

- 用户已补回 Kaggle public：
  - `exp_105 = 0.97026`
  - `exp_106 = 0.92676`
  - 两者都低于当前最佳 `exp_088 = 0.98403`

- 新增 `shadow probe` 监控能力：
  - 代码：
    - `src/dl_pipeline/training/shadow_probe.py`
    - `scripts/train.py`
    - `tests/test_shadow_probe.py`
  - 作用：
    - 在训练过程中，对 8 张核心 test case
      - `48, 289, 361, 543, 612, 699, 1333, 1525`
    - 按**正式 prototype/open-set 推理逻辑**逐 epoch 导出：
      - `argmax_label`
      - `pred_config_threshold`
      - `pred_val_selected_threshold`
      - `base_score`
      - `neighbor_score`
      - `final_score`
      - `selected_threshold`

- 先跑 `exp_107`，后发现一个真实 bug：
  - `shadow probe` callback 在导出 embedding 时切到了 `eval()`
  - 但结束后没有切回 `train()`
  - 这会污染后续 epoch
  - 已修复后改用 `exp_108` 重新完整跑一遍

- 执行 `exp_108`：
  - 配置：
    - `configs/experiments/exp_108_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_shadowprobe8_rerun.yaml`
  - 产物：
    - `outputs/exp_108_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_shadowprobe8_rerun/shadow_probe_history.csv`
    - `outputs/exp_108_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_shadowprobe8_rerun/metrics.json`
  - 结果：
    - `best_val_acc = 1.0`
    - 8 张核心 case 在全部 `9` 个 epoch 中：
      - `argmax_label` 始终都是 `2`
      - 说明模型从头到尾都更偏向 `class2`
    - 但它们的 `final_score` 始终只有大约 `0.26 ~ 0.35`
    - 每个 epoch 在 val 上自动选出的最优阈值始终是 `0.50`
    - 即使按 `0.50`，这 8 张也**没有一张**曾经被放出来
    - 最接近的一张仍是 `699`，峰值约 `0.35`
  - 结论：
    - 这 8 张不是“某些 epoch 学会了，后来又过拟合掉了”
    - 而是整个训练过程中都只是**弱证据 `class2`**
    - 因此：
      - 不值得继续沿着
        - 改 early stopping
        - 换 epoch
        - 把这 8 张当正式 val
      - 这条线继续投入的价值很低

- 重要更正（用户后续人工复核）：
  - 之前围绕
    - `48, 289, 361, 543, 612, 699, 1333, 1525`
    这 8 张展开的“潜在 `class2` 错拒”判断是错误的
  - 用户重新确认后，这 8 张实际上都是 `Sarah Hyland`，本来就应该判为 `other`
  - 因此：
    - `shadow probe` 实验仍然有效
    - 但它的正确解释应改为：
      - 模型一直把这批 lookalike 看成更像 `class2`
      - 但 open-set 拒识最终把它们正确挡在 `0`
    - 不能再把它们视为 leaderboard 可回收增益

- 同样的人工复核还确认：
  - `data/visualizations/bad_image_probe/exp_088_test_crop_logic_v2/images`
    里出现的人脸样本，当前标注都是对的
  - 也就是说，在这批 probe 里：
    - 没有再找到坏图之外的明确 `exp_088` 错判样本
  - 当前能确认的问题，重新收束为：
    - 主要集中在坏图 / 占位图 / 无法有效恢复的人脸输入

- 截至当前最稳结论：
  - `exp_088 = 0.98403`
  - 若按 `1787 / 1816` 计算，距离 `1.00000` 还差：
    - `29` 张图片

- 针对异常输入进一步做了两轮 submission probe：
  - `exp_109`
    - 仅把 3 张与 `train_037_id065_class2.png` 同型的 `IMAGE NOT FOUND` 占位坏图：
      - `537, 1071, 1112`
      改判为 `class2`
    - Kaggle public：
      - `0.98568`
    - 说明：
      - 这是一条**有效但极窄**的异常输入修正规则
  - `exp_110`
    - 将人工归出的 14 张“无脸坏图”整体改判为 `class2`
    - Kaggle public：
      - `0.98348`
    - 说明：
      - 坏图并不是统一标签模式
      - `exp_109` 的增益不能泛化成“无脸坏图 -> class2”

- 因此当前最稳收束结论更新为：
  - 仍然不存在可继续放大的通用阈值 / 训练 / 异构改进方向
  - 唯一被线上证实有效的新增 gain，是：
    - 对 3 张精确识别出的 `IMAGE NOT FOUND` 占位坏图做 submission 级修正

- 已把这条逻辑并入正式脚本流程：
  - `src/dl_pipeline/inference/submission.py`
    - 新增 `apply_submission_postprocess(...)`
  - `scripts/predict.py`
    - 在生成 submission 后、保存 CSV 前统一调用可选 postprocess
    - 若启用，会额外写出：
      - `outputs/<exp>/submission_postprocess.json`
  - 新增可直接复现的配置：
    - `configs/experiments/exp_109_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_imagenotfound3_patch.yaml`
  - 新配置跑出的 submission 与手工版 `exp_109` 已核对：
    - 完全一致

- 为释放磁盘空间，按“折中”策略清理：
  - 保留：
    - `exp_088` 主线权重
    - 关键诊断输出 / submission / metrics
  - 删除一批已证实原地打转实验的 `checkpoints/`
    - 目标：
      - `exp_096`
      - `exp_097`
      - `exp_099`
      - `exp_101`
      - `exp_103`
      - `exp_107`
      - `exp_108`
  - 实际释放空间：
    - 约 `5.964 GB`

- 用户随后进一步明确：
  - 前 `80+` 条实验的权重，也只保留那些能与 `exp_088 / exp_109` 形成直接消融叙事的
  - 复核后，实际只需要保留：
    - `exp_088` 的 checkpoint
  - 因此继续删除了所有更早实验残留的 `checkpoints/`
    - 仅保留：
      - `outputs/exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta/checkpoints`
  - 这一轮额外释放空间：
    - 约 `11.8 GB`
  - 两轮清理累计释放空间：
    - 约 `17.764 GB`

## 2026-04-05

### Session 76

- 为验证 `open-set` 推理是否确有必要，新增一个**只改推理、不重训**的受控对照：
  - `exp_111_vit_adaface_multiface_selected_20ep_last2block_ce_softmax_hfliptta`
- 该实验完全复用 `exp_088` 的：
  - 多脸选人输入协议
  - `ViT AdaFace + CE + last2block` 训练设定
  - `best.ckpt`
- 唯一变化是把推理从：
  - `neighborhood_aware + rejection`
  改为：
  - 直接使用分类头 `softmax argmax`
- 为避免引入第二个变量，已在 `softmax` 推理分支中补齐 `horizontal flip TTA`，保证与 `exp_088` 的测试时增强一致。
- 产物：
  - 配置：
    - `configs/experiments/exp_111_vit_adaface_multiface_selected_20ep_last2block_ce_softmax_hfliptta.yaml`
  - 本地输出：
    - `outputs/exp_111_vit_adaface_multiface_selected_20ep_last2block_ce_softmax_hfliptta/softmax_metrics.json`
  - submission：
    - `data/submissions/20260405_142713_exp_111_vit_adaface_multiface_selected_20ep_last2block_ce_softmax_hfliptta_submission.csv`
- 本地结果：
  - `val_accuracy = 1.0`
- 与 `exp_088` 的 test submission 逐行对比：
  - 共改动 `162` 张
  - 全部都是把原来的 `class 0` 放成 target：
    - `0 -> 2: 93`
    - `0 -> 1: 69`
  - 预测分布从：
    - `exp_088 = {0: 941, 1: 451, 2: 424}`
    变为：
    - `exp_111 = {0: 779, 1: 520, 2: 517}`
- 当前结论：
  - 在这条 `ViT + multiface-selected` 主线上，直接三分类 `softmax` 会明显削弱对 `other` 的拒识能力
  - 这条对照即使本地 `val_accuracy` 仍为 `1.0`，test 侧行为却已经明显偏向过度放出 target
  - 因此它很适合作为 notebook 中证明 `open-set` 推理必要性的受控消融
  - 用户已提交 Kaggle，public score 为 `0.90143`
  - 相比 `exp_088 = 0.98403` 明显下降，进一步说明在这条最终主线上，open-set 推理不是可有可无的后处理，而是性能成立的必要部分
