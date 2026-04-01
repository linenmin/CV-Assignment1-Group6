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
