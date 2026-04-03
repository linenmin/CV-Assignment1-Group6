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

## 2026-04-01

### TTA 实验结论

- `exp_004_resnet18_haar_10ep_tta` 仅在推理阶段加入 `horizontal flip TTA`，训练权重仍使用 `exp_002` 的最佳 checkpoint。
- 该 submission 的 Kaggle public score 为 `0.51266`，比当前最好结果 `0.61178` 明显更低。
- 这说明在当前任务上，`horizontal flip` 不是一个安全的测试时增强：
  - 要么类别判别依赖左右方向或非镜像不变特征；
  - 要么当前小样本模型对镜像后的概率分布不稳定，平均后反而削弱了正确类别置信度。
- 因此现阶段不建议继续投入更复杂的 TTA 组合，优先级应转回训练侧改进。

### 当前最合理的下一步

- 不优先换更大 backbone。
- 不优先继续做推理侧技巧。
- 下一步最合理的单一实验不是“加强 TTA”，而是直接检验 `flip invariance` 假设：
  - 在训练增强里去掉 `HorizontalFlip`；
  - 其余训练配置尽量保持不变；
  - 只观察这一项改动对 holdout 和 Kaggle score 的影响。
- 这样做的理由是：
  - 当前最明确的负面证据就是 `horizontal flip TTA` 明显伤分；
  - 这意味着“左右镜像不改变类别”这个假设很可能不成立；
  - 而训练增强里目前仍然在用 `HorizontalFlip(p=0.5)`，它可能一直在向模型注入错误的标签保持假设。
- 如果去掉 `HorizontalFlip` 后分数回升，再考虑第二步做更保守的几何增强升级：
  - 用 `Affine` 替代 `ShiftScaleRotate`；
  - 保留轻度颜色扰动；
  - 再加入非常轻的噪声或模糊增强。

### `exp_005` 当前本地结果

- `exp_005_resnet18_haar_10ep_nohflip` 已完成训练和推理。
- 在保持 `resnet18 / haar / 10 epoch / 其余增强不变` 的前提下，仅关闭了训练时的 `HorizontalFlip`。
- 本地验证集最佳准确率从上一版的 `0.6944444179534912` 提升到 `0.7222222089767456`。
- 这与前面的 TTA 失败方向一致，进一步支持“当前任务并不满足左右翻转标签不变”这一判断。
- Kaggle public score 为 `0.56607`。
- 这说明“去掉 HorizontalFlip”确实修正了一部分错误归纳偏置，但还不足以超过当前最好版本 `0.61178`。
- 因此该实验的定位应当是：
  - 方向判断上有价值；
  - 但不是新的最优提交版本。

### `exp_006` 当前本地结果

- `exp_006_resnet18_haar_10ep_nohflip_affine` 在 `exp_005` 的基础上只改了一项：把 `ShiftScaleRotate` 换成了 `Affine`。
- 本地验证集最佳准确率下降到 `0.6111111044883728`，明显低于 `exp_005` 的 `0.7222222089767456`。
- Kaggle public score 进一步下降到 `0.48898`，比 `exp_005` 的 `0.56607` 和当前最佳 `0.61178` 都更低。
- 这说明当前数据规模下，`Affine` 版几何增强并没有带来更稳的泛化，反而削弱了模型。
- 它不能证明 `Affine` 在所有更强模型上都一定无效，但已经足够说明：在当前数据规模、当前裁脸方案和当前训练流程下，这个增强方向的收益概率太低，不值得继续作为主线投入。
- 因此当前最优训练增强判断仍然是：
  - 不用 `HorizontalFlip`
  - 暂时保留原先较轻的 `ShiftScaleRotate`
- 在这个节点上，更合理的下一步不是继续打磨几何增强，而是进入一次“跨层级升级”：
  - 保留当前更稳的增强设定；
  - 保持 detector、输入尺寸、训练轮数不变；
  - 只更换 backbone，但优先切向“人脸识别专用预训练”而不是继续尝试通用 ImageNet backbone。

### `exp_007` 当前本地结果

- 根据近期人脸识别文献与官方模型发布情况，主升级方向已从通用 ImageNet backbone 调整为“人脸识别专用预训练 backbone”。
- 当前首个落地实验为 `exp_007_ir101_adaface_haar_10ep`：
  - backbone：`IR101 + AdaFace`
  - 预训练：`WebFace4M`
  - 来源：CVLFace 官方 Hugging Face 权重
  - 训练方式：冻结 backbone，仅训练新的 3 类分类头
- 本地验证集最佳准确率达到 `1.0`，显著高于此前 `resnet18` 系列实验。
- 但这里必须保守解释：
  - 当前 holdout 只有 16 张；
  - 因此 `1.0` 很可能部分反映了验证集过小，而不一定等价于真实测试集性能。
- Kaggle public score 为 `0.62720`，超过此前最好结果 `0.61178`。
- 这说明本轮提升不是单纯的 holdout 偶然波动，而是确实在测试分布上带来了净收益。
- 即便如此，这个实验仍然说明一件重要的事：
  - 当前瓶颈更可能在“表征能力不足”，而不是继续细调轻量增强。
- 因此当前主线判断变为：
  - 增强侧：维持 `no HorizontalFlip + ShiftScaleRotate`
  - 模型侧：优先继续沿人脸专用预训练 backbone 方向推进
  - 最关键的下一步，不是回到通用 ImageNet backbone，而是在这条主线上继续做低风险升级

### `exp_008` 当前结果

- `exp_008_ir101_adaface_mtcnn_10ep` 保持了 `exp_007` 的 backbone、输入尺寸与训练策略，仅把预处理从 `HAAR` 裁框升级为 `MTCNN + 5 点对齐`。
- 这一步的理论依据很明确：
  - `AdaFace / IR101` 这类人脸识别模型默认假设输入是更接近标准化、对齐后的人脸；
  - `HAAR` 更像是 baseline 级别的人脸裁框，不擅长提供稳定的 landmark 对齐；
  - 因此如果 `exp_007` 已经证明 face-specific backbone 有效，那么下一步最合理的补强就是“把脸喂对”。
- 从工程结果看，这条新链路已经在真实比赛数据上跑通：
  - `MTCNN` 检测、5 点对齐、训练、预测全部成功；
  - 本地验证集最佳准确率仍为 `1.0`。
- 但这里同样要保持保守解释：
  - 当前 holdout 仍然只有 16 张；
  - 因此本地 `1.0` 依然只说明“模型容量 + 对齐方案”足以吃满这 16 张验证样本，不代表真实测试集一定同步提升。
- Kaggle public score 为 `0.60627`，低于 `exp_007` 的 `0.62720`。
- 这说明“更标准的人脸对齐”在当前 pipeline 下并没有带来测试集净收益。
- 作为工程判断，这个结果非常关键：
  - 不能再把主要精力继续放在 `HAAR -> MTCNN -> RetinaFace` 这种检测器升级链上；
  - 因为从 `0.62720` 到 `0.60627` 的回落表明，当前收益瓶颈并不在这里。
- 更合理的解释有三类：
  1. 当前比赛不只是标准人脸识别，`neither` 类可能包含大量背景、发型、姿态、构图等弱上下文信息，过强对齐反而把这些判别线索裁掉了。
  2. 你们现在用的是“80 张小样本上的 3 类分类头”，这对两位目标人物加一个开放集 `neither` 来说，任务建模本身过于粗糙。
  3. 本地 16 张 holdout 已经基本失去指导意义。`exp_007` 和 `exp_008` 都能把本地验证打到 `1.0`，但 Kaggle 分数却能差出 `0.02`，说明当前本地指标已经饱和。
- 因此当前的主结论应该更新为：
  - `IR101 + AdaFace` 这条人脸专用预训练 backbone 主线是对的；
  - 但继续在“裁脸/对齐更精细”这一层微调，收益已经明显变小；
  - 下一步必须转向**更符合任务结构的建模方式**，而不是继续堆检测器或继续只做 3 类 softmax 分类。

### `exp_009` 当前结果

- `exp_009_ir101_adaface_haar_10ep_laststage_ft` 回到 `exp_007` 的输入与预处理设定，仅调整训练策略：
  - 继续使用 `IR101 + AdaFace`
  - 保持 `HAAR`
  - 保持 `112x112`
  - 不再完全冻结 backbone，而是只解冻最后一个 stage
  - 为 backbone 使用比分类头更小的学习率
- 这一步的理论依据是：
  - `exp_007` 已经证明 face-specific backbone 方向正确；
  - 但“完全冻结”可能过于保守，无法让 backbone 适应这次比赛的具体分布；
  - 与此同时，直接全量微调在 80 张数据下风险太高，因此“只解冻最后一个 stage”是更稳妥的中间路线。
- 训练阶段出现了一个明确的工程现象：
  - 在冻结 backbone 时，`batch_size=16` 可以接受；
  - 进入部分微调后，同样的 `batch_size=16` 会导致本机资源压力明显上升；
  - 将 `batch_size` 下调到 `8` 后，训练稳定完成。
- 本地验证集最佳准确率为 `0.9444444179534912`。
- 这个数值本身并不比 `exp_007` / `exp_008` 的本地 `1.0` 更好，但这里恰好说明了另一个更重要的问题：
  - 当前 16 张 holdout 已经不能可靠区分真正更优的策略；
  - 因为这次 Kaggle public score 却显著提升到了 `0.78799`。
- 这是当前最关键的发现：
  - 相比 `exp_007` 的 `0.62720`，`exp_009` 提升了约 `0.16`；
  - 这不是小波动，而是方法层级上的实质改进。
- 因此当前结论应更新为：
  - 你们真正需要继续深挖的，不是检测器升级链，也不是再回到轻量增强打磨；
  - 当前收益最大的杠杆，是**在强人脸预训练 backbone 上做受控的部分微调**；
  - 这条线已经明显比“冻结 backbone 只训分类头”更接近第一梯队。

### `exp_010` 当前结果

- `exp_010_ir101_adaface_haar_10ep_laststage_ft_lr1e5` 保持了 `exp_009` 的全部结构，只把 backbone 学习率从 `3e-5` 降到了 `1e-5`。
- 这是一个非常干净的单变量实验，因此结论可信度很高。
- 从训练曲线看，这次优化行为比 `exp_009` 更稳定：
  - `train_loss` 下降更慢，没有像 `exp_009` 那样迅速压到极低；
  - `val_loss` 从 `0.438` 持续下降到 `0.111` 左右，而不是早早见底后持续恶化；
  - `val_acc` 在第 4 个 epoch 达到 `1.0`，之后虽有轻微回落，但整体仍保持在很高水平。
- 这说明：
  - `exp_009` 的 backbone 微调虽然方向对，但更新幅度略偏大；
  - 把 backbone learning rate 再收缩一档之后，模型对比赛分布的适配变得更平滑，泛化更好。
- Kaggle public score 为 `0.87885`，相较 `exp_009` 的 `0.78799` 再提升约 `0.09`。
- 这个增幅已经足够说明：
  - 你们当前已经不需要再寻找“全新的方向”；
  - 现阶段最值得做的是沿这条成功主线继续做精细化调参和更合理的模型选择。
- 当前最关键的进一步发现是：
  - 当 `val_acc` 很快到达 `1.0` 时，它已经失去足够的分辨力；
  - 这时真正更有信息量的是 `val_loss`，因为它仍能区分不同 checkpoint 的置信度与泛化状态。
- 因此下一步最合理的升级，不是手工挑 epoch，也不是硬编码保留某几个 epoch，而是：
  - 将 checkpoint 选择和 early stopping 的监控指标从 `val_acc` 切换到 `val_loss`；
  - 让训练流程用更稳定、更有分辨力的指标自动选模型。

### `exp_011` 当前结果

- `exp_011_ir101_adaface_haar_10ep_laststage_ft_lr1e5_valloss` 与 `exp_010` 保持同一条训练主线：
  - `IR101 + AdaFace`
  - `HAAR`
  - `112x112`
  - 只解冻最后一个 stage
  - `head lr = 3e-4`
  - `backbone lr = 1e-5`
- 本次唯一改动是**模型选择标准**：
  - 不再用 `val_acc / max` 选 checkpoint
  - 改为用 `val_loss / min` 选 checkpoint
- 这一步的意义不在于“换一个更强模型”，而在于解决一个已经被 `exp_010` 暴露的问题：
  - 当 `val_acc` 很快到 `1.0` 时，它已经失去区分不同 checkpoint 的能力；
  - 这时继续用 `val_acc` 选模型，实际上是在用一个已经饱和的指标做决策；
  - `val_loss` 虽然变化幅度更小，但还能反映置信度与泛化差异，因此更适合承担选模职责。
- 当前本地结果为：
  - 最佳 `val_loss = 0.1152695044875145`
  - 对应 epoch 的 `val_acc = 1.0`
  - 最优点出现在第 `4` 个 epoch 附近
- 从训练曲线形状看，这次的最优区间与 `exp_010` 非常接近：
  - `val_loss` 从 `0.447 -> 0.215 -> 0.150 -> 0.136 -> 0.115`
  - 在 epoch 4 之后没有继续实质改善，后面略有回升
- 这说明两件事：
  1. 之前对曲线的判断是对的，最佳区间确实就在 epoch 4 左右；
  2. 把 monitor 改成 `val_loss` 后，训练流程终于能用原则化方式自动抓住这个最优点，而不需要任何硬编码 epoch 的做法。
- 当前 Kaggle public score 为 `0.85572`，低于 `exp_010` 的 `0.87885`。
- 这说明在“只解冻最后一个 stage”这条主线上，`val_acc` 与 `val_loss` 的最优点本来就在同一最优区间附近：
  - `exp_010` 的最低 `val_loss` 在 epoch 4
  - `exp_011` 的最低 `val_loss` 也在 epoch 4
- 因此这轮实验的重要结论不是“`val_loss` 无效”，而是：
  - 对 `exp_010` 这条一阶段微调主线而言，monitor 切换并没有改变真正被选中的最优区间；
  - 当前主要提升来源仍然是“更温和的 backbone 微调”，而不是“改选模指标”。

### `exp_012` 当前结果

- `exp_012_ir101_adaface_haar_10ep_last2stage_ft_head1e4` 在 `exp_010` 的基础上做了更激进的结构改动：
  - 解冻范围从最后 `1` 个 stage 扩展到最后 `2` 个 stage
  - 同时把分类头学习率收缩到 `1e-4`
  - backbone 学习率收缩到 `5e-6`
  - `batch_size` 降到 `4`
- 从本地曲线看，它明显弱于 `exp_010`：
  - 最低 `val_loss = 0.18338939547538757`
  - 最优 epoch 在 `4`
  - `val_acc` 从 epoch 1 开始就卡在 `0.9444444179534912`
- 但 Kaggle public score 为 `0.87775`，仅比 `exp_010` 的 `0.87885` 低 `0.00110`。
- 这个结果很重要，因为它说明：
  - 两层解冻并没有在线上造成明显退化；
  - 当前本地验证集会放大这条线的负面信号；
  - 真正的问题更像是“这组学习率和选模规则还没配平”，而不是“两层解冻”方向本身错误。

### `exp_013` 当前结果

- `exp_013_ir101_adaface_haar_10ep_last2stage_ft_head1e4_valloss` 保持 `exp_012` 的结构和学习率完全不变，只把 monitor 从 `val_acc / max` 改成 `val_loss / min`。
- 这是因为 `exp_012` 的关键现象非常明确：
  - `val_acc` 在 epoch 1 就达到并保持 `0.9444444179534912`
  - 但 `val_loss` 直到 epoch 4 还在继续下降
  - 也就是说，`val_acc` 在两层解冻这条线上已经失去选模能力
- 本地结果确实按预期改善：
  - `exp_012` 的最低 `val_loss` 为 `0.18338939547538757`
  - `exp_013` 的最低 `val_loss` 降到 `0.1475604921579361`
  - 最优 epoch 也从 `4` 后移到 `8`
- 但 Kaggle public score 为 `0.87720`，仍然没有超过：
  - `exp_012` 的 `0.87775`
  - 更没有超过 `exp_010` 的 `0.87885`
- 因此当前必须更新判断：
  - 在“两层解冻”这条线上，`val_loss` 选模确实修复了本地 checkpoint 选择问题；
  - 但这条线的真实上限在当前超参数下仍没有超过“一层解冻 + backbone lr=1e-5”的 `exp_010`；
  - 这说明当前的决定性因素不是 monitor，而是**解冻范围与优化强度之间的匹配关系**。

### `exp_014` 当前结果

- `exp_014_ir101_adaface_haar_10ep_laststage_ft_lr1e5_degradeaug` 回到 `exp_010` 的最优主线，只加了一类新的工程手段：
  - 更强但“人脸安全”的退化增强
  - 具体包括轻量 `GaussianBlur / MotionBlur / GaussNoise / ImageCompression`
  - 再叠加很轻的 `CoarseDropout`
- 这轮设计的逻辑是合理的：
  - `exp_010` 的曲线显示轻度过拟合；
  - 因此引入更接近真实拍摄退化的增强，理论上有机会让模型对测试图像分布更稳。
- 从本地曲线看，这个方向确实带来了更强的正则化信号：
  - `exp_010` 的最低 `val_loss = 0.11138796806335449`
  - `exp_014` 的最低 `val_loss = 0.09722547978162766`
  - 且最优区间从 epoch 4 附近延长到了 epoch 5-6
- 但 Kaggle public score 为 `0.87665`，低于 `exp_010` 的 `0.87885`。
- 这说明一个非常重要的工程事实：
  - 当前验证集会把“更强增强带来的更低置信度过拟合”视为更优；
  - 但线上测试集并不买账，说明这组增强已经开始偏离真实分布，或者削弱了身份判别的关键细节。
- 因此这轮实验应被视为：
  - 一个有价值的负例；
  - 它证明“更强正则化/更低 `val_loss`”不等于更高 Kaggle 分数；
  - 也说明继续往“再强一点增强”这个方向加码，当前不是最高优先级。

### `exp_015` 当前结果

- `exp_015_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype` 没有再训练新模型，而是直接复用了当前最强训练主线 `exp_010` 的 checkpoint。
- 本次唯一改变的是**推理建模方式**：
  - 从闭集三分类 softmax
  - 改成基于 AdaFace embedding 的 prototype + threshold 开放集识别
- 具体做法是：
  - 用训练集 `class=1` 和 `class=2` 的 embedding 分别构建 `Jesse` / `Mila` prototype
  - 在验证集上搜索阈值
  - 当测试图像与两个 prototype 的最高 cosine similarity 低于阈值时，输出 `class=0 (other)`
- 自动搜索结果为：
  - `selected_threshold = 0.7`
  - 本地 `val_accuracy = 0.875`
- 这里最重要的不是本地数值本身，而是它与线上结果的关系：
  - `exp_010` 本地 `val_acc = 1.0`，Kaggle `0.87885`
  - `exp_015` 本地 `val_acc = 0.875`，Kaggle `0.89842`
- 这是到目前为止最有解释力的一条证据：
  - 当前 16 张 holdout 对“开放集识别能力”的评估明显失真；
  - 把 `other` 当作闭集第三类，会让模型在本地看起来更准，但在线上开放集分布下更差；
  - 而 prototype 推理虽然牺牲了少量本地闭集准确率，却显著提升了真实测试集表现。
- 从测试集预测分布的变化也能看出这点：
  - `exp_010` 预测 `class=0` 数量为 `822`
  - `exp_015` 预测 `class=0` 数量增加到 `1131`
  - 一共有 `309` 张测试图被重新判定，其中全部来自原先的 `class=1/2 -> class=0`
- 这说明 prototype + threshold 的核心作用不是“更会区分 Jesse 和 Mila”，而是：
  - **更保守地拒识非目标人物**
  - 也就是更准确地解决了这个任务真正困难的部分
- 因此这轮实验带来的结论比单次分数提升更重要：
  - 你们当前的主要突破口不在训练层，而在推理层的开放集建模；
  - `other` 不是一个应被硬压成单类的紧密类别，而应该被视为“拒识区域”。

### `exp_016` 当前结果

- `exp_016_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_finegrained` 保持 `exp_015` 的全部思路不变，只把阈值搜索从粗粒度改成细粒度。
- 这轮自动选出的阈值从：
  - `exp_015` 的 `0.7`
  - 降到了 `exp_016` 的 `0.55`
- 本地验证集准确率也从：
  - `0.875`
  - 提升到 `0.9375`
- Kaggle public score 则进一步从：
  - `0.89842`
  - 提升到 `0.91299`
- 这个结果很有解释力，因为它表明：
  - `exp_015` 的 prototype 主线本身是对的；
  - 但当时的阈值偏高，模型过于保守，把一部分本该识别为 `Jesse / Mila` 的样本误拒识成了 `other`。
- 从测试集预测分布变化可以更具体地看到这一点：
  - `exp_015` 预测 `class=0` 数量为 `1131`
  - `exp_016` 降到 `1086`
  - 一共只有 `45` 张测试图发生改变：
    - `17` 张从 `0 -> 1`
    - `28` 张从 `0 -> 2`
- 也就是说，这轮提升不是来自大规模重排，而是来自**少量但关键的“拒识回收”**。
- 这说明当前开放集推理线还有两个重要结论：
  1. 全局阈值确实是高敏感参数，粗粒度搜索会直接损失分数；
  2. 目前最优阈值已经压到 `0.55`，说明 `other` 与目标人物之间的边界比最初想象的更靠近，需要更温和的拒识策略。
- 因此 `exp_016` 把当前主判断进一步收紧为：
  - 当前最有效的杠杆不是训练，而是**开放集边界如何设定**；
  - 下一步最值得做的，不是重新训一个新模型，而是继续细化“prototype 如何表示、阈值如何按类区分、拒识规则如何更贴近分布”。

### `exp_017` 当前结果

- `exp_017_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_classspecific` 继续复用 `exp_010` 的 checkpoint，不重训模型，只把 `exp_016` 的单一全局阈值扩展成“每个目标类一个独立阈值”。
- 设计这个实验的动机是合理的：
  - `exp_016` 的增益主要来自把一小批被过度拒识的样本从 `other` 拉回 `Jesse / Mila`；
  - 因此自然会怀疑两个目标类在 embedding 空间里的紧致程度不同，可能需要不同的接受边界。
- 但实际自动搜索结果是：
  - `selected_thresholds_by_class = {1: 0.45, 2: 0.45}`
  - 本地 `val_accuracy = 0.9375`
- 这个结果本身就已经暴露了问题：
  - 它没有学出真正的类别差异阈值；
  - 它只是把 `exp_016` 的全局阈值 `0.55` 进一步放宽成更低的统一阈值 `0.45`。
- 对测试集预测分布的影响也非常激进：
  - `exp_016` 预测 `class=0` 数量为 `1086`
  - `exp_017` 降到 `950`
  - 一共有 `136` 张测试图发生改变：
    - `131` 张从 `0 -> 1`
    - `5` 张从 `0 -> 2`
- Kaggle public score 从 `0.91299` 直接降到 `0.85572`，说明这 `136` 次放宽里，大部分都是**错误接收**。
- 因而这轮实验给出的关键结论非常明确：
  1. 开放集推理这条主线仍然是对的，但当前 16 张 holdout 不足以支撑更高自由度的阈值建模；
  2. 在这种极小验证集上做二维阈值搜索，容易把“并列最优中的最先命中项”误当成稳健最优；
  3. 当前更需要的是**更稳健的全局阈值估计**，而不是继续增加边界参数或提高推理自由度。

### `exp_018` 当前结果

- `exp_018_ir101_adaface_haar_10ep_laststage_ft_lr1e5_prototype_cvglobal` 继续复用 `exp_010` 的 checkpoint，不重训模型，只把阈值估计方式从“单一 holdout”改成“训练集分层交叉验证平均”。
- 这轮的目标非常明确：
  - 既然 `exp_017` 说明高自由度阈值拟合会被 16 张 val 带偏；
  - 那就退回单一全局阈值，但用更稳健的估计过程来选它。
- 自动搜索结果为：
  - `selected_threshold = 0.45`
  - `crossval_mean_accuracy = 0.90625`
  - 本地 `val_accuracy = 0.9375`
- 但 Kaggle public score 为 `0.85848`，仍然远低于 `exp_016` 的 `0.91299`。
- 这轮最关键的信号是：
  - 它再次把阈值压到了 `0.45`，而不是接近 `exp_016` 的 `0.55`；
  - 说明当前训练集内部的分层交叉验证，同样会奖励更宽松的接受边界；
  - 但这种更宽松的边界在线上会把太多 `other` 错接收成目标人物。
- 从测试集预测分布可以直接看出这一点：
  - `exp_016` 预测 `class=0` 数量为 `1086`
  - `exp_018` 降到 `955`
  - 一共有 `131` 张测试图发生改变：
    - `126` 张从 `0 -> 1`
    - `5` 张从 `0 -> 2`
- 因此这轮实验进一步收紧了结论：
  1. `exp_016` 的 `0.55` 已经不是偶然噪声，而是当前最可信的最佳阈值；
  2. 在现有数据规模下，不论是二维阈值搜索还是交叉验证全局阈值搜索，都会系统性把边界放得过松；
  3. 当前下一步不该继续搜索阈值，而应该**固定 `0.55`，把它当作稳定推理层，然后再评估新的模型是否能在这个固定边界下带来增益**。

### `exp_019 ~ exp_022` 当前结果

- 这四轮实验不再搜索阈值，而是把 `exp_016` 已被线上验证的最佳阈值 `0.55` 固定下来，统一重评估已有模型：
  - `exp_019`: `exp_009` checkpoint + prototype + fixed `0.55`
  - `exp_020`: `exp_012` checkpoint + prototype + fixed `0.55`
  - `exp_021`: `exp_013` checkpoint + prototype + fixed `0.55`
  - `exp_022`: `exp_014` checkpoint + prototype + fixed `0.55`
- Kaggle public score 为：
  - `exp_019 = 0.91685`
  - `exp_020 = 0.91519`
  - `exp_021 = 0.91079`
  - `exp_022 = 0.91024`
- 这些结果的意义比“谁高了 `0.001`”更重要：
  1. **固定 open-set 推理协议后，最优模型发生了变化。**
     - softmax 下的最好训练模型是 `exp_010`
     - 但在固定 `prototype + threshold=0.55` 评估下，最好变成了 `exp_019`，也就是 `exp_009` 的 checkpoint
     - 这说明 softmax submission 分数并不能可靠代表 embedding 质量
  2. **更激进的一阶段微调其实给出了更好的 embedding。**
     - `exp_009` 的 backbone lr 更大（`3e-5`）
     - softmax 下它只有 `0.78799`
     - 但换成 prototype 固定阈值后，它直接成为当前最好分数 `0.91685`
     - 这说明之前真正被 softmax 压制的是“开放集决策层”，不是 backbone 本身
  3. **两阶段解冻、valloss 选模、强退化增强都没有带来决定性优势。**
     - `exp_020/021/022` 都在 `0.910 ~ 0.915` 区间
     - 与 `exp_019` 的差距不大，但方向已经很清楚：这些训练技巧最多带来小幅波动，不能解释离 leaderboard `0.97` 还差的 `0.05+`
- 因而这四轮共同给出的判断是：
  - 当前最值得优化的已经不是“再调训练细节”，而是**把 open-set 决策从“单 centroid + 单阈值”升级成更强的 gallery / multi-prototype / negative-aware 推理**；
  - 只有这种层级的方法变化，才更有可能带来接近 `0.97` 所需的量级提升。

### `exp_023` 当前结果

- `exp_023_ir101_adaface_haar_10ep_laststage_ft_exemplar_knn_margin` 继续复用当前最强 embedding checkpoint `exp_009`，不重训模型，只把推理从 `single prototype` 升级为：
  - exemplar gallery
  - top-k nearest-neighbor
  - target-vs-other margin rejection
- 这条线原本的动机是合理的：
  - 单 centroid 会丢掉局部结构；
  - exemplar / kNN 理论上更能保留姿态、光照、表情等多模态信息；
  - margin 也能把“既像目标又像 other”的样本拒识掉。
- 但当前自动搜索结果是：
  - `selected_top_k = 1`
  - `selected_margin = 0.0`
  - `selected_threshold = 0.55`
  - 本地 `val_accuracy = 0.9375`
- 这意味着这轮并没有真正学到“更稳的 kNN + margin”规则，而是退化成了：
  - **1-NN**
  - **无 margin**
  - **只保留固定阈值**
- Kaggle public score 为 `0.84911`，相比 `exp_019 = 0.91685` 出现明显退化。
- 从预测分布变化看，问题非常明确：
  - `exp_019` 预测分布：`{0: 1091, 1: 343, 2: 382}`
  - `exp_023` 预测分布：`{0: 878, 1: 514, 2: 424}`
  - 一共有 `215` 张测试图发生变化：
    - `172` 张从 `0 -> 1`
    - `42` 张从 `0 -> 2`
    - 只有 `1` 张从 `1 -> 0`
- 也就是说，这轮几乎完全是在大规模放松拒识，把原本判成 `other` 的样本大量吸进目标类。
- 这说明什么：
  1. **1-NN 太不稳。**
     - 它高度依赖单张 exemplar，容易被噪声样本、偶然相似样本、异常姿态样本牵着走；
     - 在人脸开放集任务里，这种不稳定性会直接表现为“误接收 other”。
  2. **当前 16 张验证集再次失效。**
     - 它把 `1-NN + 0 margin` 视为与更稳规则同样好；
     - 但线上说明这是明显错误方向。
  3. **问题不在“exemplar 思路一定错”，而在“让验证集自由选成 1-NN + 无抑制”这个结果错了。**
     - exemplar 要想有效，必须配合更强的聚合或更强的负类竞争，而不能退化成最脆弱的单样本最近邻。
- 所以这轮实验带来的更新结论是：
  - “multi-exemplar”本身仍然值得保留；
  - 但下一步不能再让搜索空间包含 `top_k = 1, margin = 0` 这种明显过松、过脆弱的组合；
  - 更靠谱的方向是：**保留 exemplar 信息，但用更稳健的相似度聚合和更强的 other-aware margin，而不是直接做裸 1-NN。**

### `exp_024` 当前结果

- `exp_024_ir101_adaface_haar_10ep_laststage_ft_prototype_fixed055_alllabeled` 继续复用 `exp_009` checkpoint，不重训模型，只把 `exp_019` 的 prototype gallery 从训练子集扩展到 `all 80 labeled`。
- 这是一个很干净的验证：如果当前瓶颈只是“gallery 样本覆盖不足”，那把 `val` 一并纳入 gallery 至少应该改变部分测试预测。
- 但实际结果是：
  - `exp_024` 的 submission 与 `exp_019` **逐行完全一致**
  - 差异条数为 `0`
- 这个结果非常关键，因为它说明：
  1. 当前问题不在“gallery 点太少”，而在“如何聚合这些点”；  
  2. 在 `single prototype + fixed threshold 0.55` 这条规则下，把更多样本压成每类一个均值，并不会增加有效判别信息；  
  3. 所以如果后续还要继续利用 `all 80 labeled`，重点应放在更丰富的聚合/决策规则上，而不是继续坚持单 centroid。

### `exp_025` 当前结果

- `exp_025_ir101_adaface_haar_10ep_laststage_ft_richer_openset_alllabeled` 继续复用 `exp_009` checkpoint，不重训模型，在 `all 80 labeled` 上引入更丰富的开放集打分：
  - `target top-k mean`
  - `other top-k mean`
  - `best-vs-second target margin`
- 这条线的理论动机是对的：当前单阈值规则太粗，确实缺少“目标与 other 的竞争关系”和“两个目标类之间的分差”。
- 但当前 leave-one-out 搜索结果为：
  - `selected_threshold = 0.53`
  - `selected_target_top_k = 3`
  - `selected_other_top_k = 3`
  - `selected_other_margin = 0.0`
  - `selected_target_margin = 0.0`
  - `leave_one_out_accuracy = 0.9125`
- 也就是说，这轮最终依然退化成了：  
  - 更低阈值  
  - 无 margin  
  - 只比 `exp_019` 略微放松一点拒识  
- 相比 `exp_019`，它只改了 `8` 张预测，几乎全部是在把 `other` 轻微往目标类放宽。
- 因而这轮给出的判断是：
  1. richer scorer 这个方向并没有错，但当前这组新特征还不够强；  
  2. 如果自动搜索最后仍然把两个 margin 都压到 `0`，说明这些新特征并没有真正改变决策边界的性质；  
  3. 这类“只比 `exp_019` 轻微更松一点”的规则，不值得占用 Kaggle 日提交配额。

### `exp_026` 当前结果

- `exp_026_ir101_adaface_haar_10ep_quality_aware_openset_alllabeled` 尝试把 `AdaFace` 的质量信息显式引入推理：
  - quality-weighted gallery aggregation
  - low-quality probe threshold boost
- 这是一个合理方向，因为 `AdaFace` 的设计本身就强调质量感知，按理说 embedding norm 应当可以给开放集拒识提供额外帮助。
- 但当前搜索结果非常直接：
  - `selected_quality_alpha = 0.0`
  - `selected_low_quality_threshold_boost = 0.0`
  - `leave_one_out_accuracy = 0.9125`
- 这意味着：**在当前实现和当前 embedding 上，quality-aware inference 最终没有被选中。**
- 它相对 `exp_019` 只改了 `12` 张预测，而且主要是轻微的 `0 -> 1` 放宽。
- 所以这轮实验最重要的不是“有没有涨一点”，而是给出了一个更硬的负结论：
  1. 质量信息在当前 pipeline 下没有转化成有效的新判别信号；  
  2. 当前最缺的不是“再加一个轻量质量特征”，而是**真正改变决策函数形态**；  
  3. 因为这轮没有学出质量感知行为，所以它不值得继续消耗 Kaggle 配额验证。

### `exp_027` 当前结果

- `exp_027_ir101_adaface_haar_10ep_verifier_openset_alllabeled` 尝试把问题改写成两个 one-vs-rest verifier：
  - Jesse verifier
  - Mila verifier
  - 各自阈值 + reject
- 这条线的出发点也很合理：
  - 与其把任务当作闭集三分类，不如把它拆成两个“是否属于目标人”的验证问题；
  - 这更接近开放集识别的真实结构。
- 但当前 leave-one-out 搜索结果是：
  - `selected_target_top_k = 3`
  - `selected_negative_top_k = 3`
  - `selected_thresholds_by_class = {1: 0.06, 2: 0.06}`
  - `leave_one_out_accuracy = 0.9375`
- 这个阈值已经暴露出问题：它太低，等于在大幅放宽接收边界。
- 相比 `exp_019`：
  - 一共有 `119` 张预测发生变化
  - 全部是把 `other` 放宽为目标类：
    - `52` 张 `0 -> 1`
    - `67` 张 `0 -> 2`
- 这和 `exp_017 / exp_018 / exp_023` 的失败模式高度一致。
- 所以这轮更新后的判断是：
  1. verifier-style 思路本身可能仍有理论价值，但当前实现版并没有解决核心问题；  
  2. 在现有 embedding 上，leave-one-out 依然会系统性奖励更松的接受边界；  
  3. 因而 verifier 方向在“当前实现版本”上可先视为负例，不值得继续用 Kaggle 提交去验证。

### `exp_028` 当前结果

- `exp_028_ir101_adaface_haar_10ep_ensemble_prototype_fixed055` 尝试对当前几条较强 checkpoint 做同 backbone embedding 平均：
  - `exp_009`
  - `exp_010`
  - `exp_012`
  - `exp_014`
- 当前集成策略非常克制：
  - 每个模型 embedding 先 `L2 normalize`
  - 跨模型直接平均
  - 再做一次 `L2 normalize`
  - 推理仍然固定为 `single prototype + threshold=0.55`
- 本地结果是：
  - `val_accuracy = 0.875`
- 相比 `exp_019`：
  - 只改了 `20` 张预测
  - 全部是把目标类更保守地推回 `other`：
    - `1 -> 0`: `3`
    - `2 -> 0`: `17`
- 这轮的关键结论是：
  1. 简单平均并没有形成有效互补，反而稀释了 `exp_009` 中最有任务特异性的偏移；  
  2. 当前失败的不是“集成”这个思想，而是“同源 checkpoint 的朴素平均”；  
  3. 因而后续若还考虑融合，必须要么换更异构的 backbone / 预训练来源，要么换更有针对性的融合方式，而不是继续堆同 backbone checkpoint。

### `other` look-alike 结构诊断

- 重新核对作业文本后，当前任务的一个关键事实已被确认：
  - `other` 不是随机陌生人；
  - 它由 `Michael Cera` 与 `Sarah Hyland` 两组 look-alike 构成。
- 这意味着此前很多“generic other rejection”的建模假设并不完全贴题。  
  更准确的任务结构是：
  - `Jesse` vs `Michael-like`
  - `Mila` vs `Sarah-like`
  - 再加拒识
- 为验证这一点，已用 `exp_009` 的 `all 80 labeled` embedding 对 `other=0` 的 `20` 张样本做 `k=2` 聚类。
- 当前结果非常清楚：
  - 两个簇各 `10` 张
  - `silhouette_score = 0.3656`
  - 一个簇显著更像 `Jesse`
  - 一个簇显著更像 `Mila`
- 两个簇的均值相似度为：
  - `michael_like`:
    - `mean_sim_to_jesse = 0.3316`
    - `mean_sim_to_mila = 0.0618`
  - `sarah_like`:
    - `mean_sim_to_jesse = 0.0251`
    - `mean_sim_to_mila = 0.3821`
- 这个结果带来两个层面的判断：
  1. **方向修正是成立的。**  
     当前任务更像“look-alike-aware rejection”，而不是把所有 `other` 当作一个无结构类别。
  2. **但增益量级不能高估。**  
     当前训练集上的 look-alike 样本与目标 prototype 的相似度仍明显低于当前接受阈值 `0.55`。这说明 look-alike-aware inference 值得做，但更像是一个低风险诊断实验，而不是已经被证明的大幅涨分方向。

### 当前综合判断（更新）

- 现阶段最稳的基线仍然是 `exp_019 = 0.91685`。
- 当前新的高价值信息不是“又找到了一个更复杂 scorer”，而是：
  - `other` 内部确实有强结构；
  - 且这个结构与目标人物一一对应。
- 所以下一步最值得做的不是继续搜索新的 margin，而是：
  - 先做**零新超参数**的 `4 prototype` 推理诊断：
    - `Jesse`
    - `Mila`
    - `michael_like`
    - `sarah_like`
  - 再看相对 `exp_019` 的预测变化规模与方向；
  - 只有在这个诊断版显示出足够多、且方向合理的修正后，才值得进一步做 `4 类训练` 或 `4 类 ArcFace`。

### `exp_029` 当前结果

- `exp_029_ir101_adaface_haar_10ep_lookalike_prototype_fixed055` 已完成上面这条零新超参数诊断：
  - 继续复用 `exp_009` checkpoint
  - 不重训
  - 仅把 `other` 显式拆成 `michael_like` 与 `sarah_like` 两个 prototype
  - 推理仍固定为 `threshold = 0.55`
- 当前本地结果：
  - `val_accuracy = 0.9375`
  - `assigned_clusters = {michael_like_cluster_id: 1, sarah_like_cluster_id: 0}`
  - `test_nearest_label_counts = {1:393, 2:525, 3:431, 4:467}`
- 但最关键的不是这些统计，而是：
  - **相对 `exp_019`，submission 逐行完全一致，差异条数为 `0`。**
- 这给出了一个比聚类诊断更硬的结论：
  1. 虽然 `other` 内部确实存在清晰的 look-alike 结构；  
  2. 但在当前 `exp_009` embedding 和 `0.55` 接收阈值下，这个结构并没有穿过现有决策边界；  
  3. 因而“显式建两个 look-alike prototype”不会改变任何测试预测。  
- 更新后的判断是：
  - 当前差距更可能来自 embedding 本身的表征上限、训练目标不对齐、或测试集中存在当前 prototype 规则无法触及的 harder cases；
  - 而不是因为 `exp_019` 少建了两个 `other` 子原型。

### `exp_030` 当前结果

- `exp_030_ir101_adaface_haar_10ep_laststage_ft_arcface2_prototype_recalib` 尝试做一个受控的 metric-aligned pilot：
  - backbone 与微调强度尽量沿用 `exp_009`
  - 训练目标改为 `2 类 ArcFace`
  - 只对 `Jesse / Mila` 计算 margin loss
  - `other` 样本不参与 loss
  - 推理仍回到 `prototype`，只做窄范围阈值重校准
- 这条线的理论动机是对的：它直接攻击了“CE 训练目标与 cosine/prototype 推理不一致”这个结构性问题。
- 但当前实测结果是负面的：
  - `best_val_target_acc = 0.9167`
  - `selected_threshold = 0.55`
  - `val_accuracy = 0.9375`
  - Kaggle public score = `0.76927`
  - 相比 `exp_019`，有 `278` 张预测变化，而且**全部是更宽松的接收**：
    - `195` 张 `0 -> 1`
    - `83` 张 `0 -> 2`
- 也就是说，这轮 ArcFace pilot 在当前实现下并没有学出更好的开放集分离，反而更像：
  - 把目标类特征拉得更“吸附”
  - 但没有同步学到对 `other` 的排斥
  - 最终把大量原本该拒识的样本卷进来
- 这轮带来的更新判断是：
  1. 训练目标对齐仍然可能是对路方向；  
  2. 但“2 类 ArcFace + 现有 prototype reject”这个最简组合在当前任务上不够；  
  3. 如果后续继续走 metric-learning 线，必须显式补上 `other` 的拒识约束，而不能只优化 `Jesse / Mila` 两类的紧致度与分离度。  

### `exp_031` 当前结果

- `exp_031_ir50_adaface_haar_10ep_laststage_ft_prototype_fixed055` 是一个受控的 backbone 切换实验：
  - 仅把 `IR101 AdaFace` 替换为 `IR50 AdaFace`
  - 其他训练和推理协议尽量保持 `exp_009 / exp_019` 不变
- 这条线的目的不是直接冲榜，而是判断：
  - 当前问题是否主要来自 `IR101` 的容量/表征特点；
  - 还是同家族 backbone 的切换本身就不足以带来真正新信息。
- 当前结果是负面的：
  - `best_val_acc = 0.9444`
  - `val_accuracy = 0.875`
  - 相比 `exp_019`，有 `237` 张预测变化，且几乎全部是更宽松的接收：
    - `233` 张 `0 -> 1`
    - `2` 张 `0 -> 2`
    - `2` 张 `2 -> 0`
- 这说明：
  1. `IR50` 在当前协议下没有带来更强的 open-set 拒识；  
  2. 它的行为模式和 `exp_030` 在方向上相似，都是更容易把 `other` 吸进去；  
  3. 因而“同家族 iResNet 容量切换”不是当前最有希望的主杠杆。  
- 更新后的判断是：
  - 如果继续沿“新 backbone”这条轴推进，下一步不该再停留在同类 iResNet 上，而应优先考虑结构差异更大的 `ViT` 或其他异构 backbone。

### 当前综合判断

- 经过 `exp_024 ~ exp_027` 这四轮离线实验后，可以把现阶段结论收紧成：
  1. `exp_019 = 0.91685` 曾是阶段性最强线上基线，但这个位置已经被 `exp_032 = 0.92180` 替代；  
  2. 小幅改造 open-set scorer，如果最终表现成“阈值更低、更多 `0 -> 1/2`”，基本都不值得提交；  
  3. `exp_030 = 2 类 ArcFace` 已被线上 `0.76927` 明确证伪，因此“只优化 Jesse/Mila、不显式建模 other 拒识”的训练思路可以先停；  
  4. 当前最值得投入的下一个方向，应该是**显式把 other 的排斥带进训练**，而不是继续微调 scorer。

### `exp_032` 当前结果

- `exp_032_vit_adaface_haar_1ep_frozen_prototype_fixed055` 是一个刻意压低工程量的 `ViT` 诊断实验：
  - 使用 `minchul/cvlface_adaface_vit_base_webface4m`
  - 不做 `ViT` 微调
  - 仅用 `1 epoch` 头部训练产出可加载 checkpoint
  - 实际关心的是 frozen `ViT` embedding 在 `prototype + threshold=0.55` 下的 open-set 行为
- 这轮最重要的不是训练精度，而是 prototype 侧结果：
  - `best_val_acc = 0.75`
  - `val_accuracy = 0.9375`
  - 相比 `exp_019`，submission 仅有 `11` 张差异：
    - `9` 张 `0 -> 1`
    - `1` 张 `0 -> 2`
    - `1` 张 `2 -> 0`
- 这个结果和 `exp_030 / exp_031` 很不一样：
  - `exp_030` 与 `exp_031` 都表现成大规模放宽接收边界；
  - `exp_032` 则几乎维持了 `exp_019` 的整体行为，只做了极小幅度的决策移动。
- 这轮现在已经有线上结果：
  - Kaggle public score = `0.92180`
  - 相比 `exp_019 = 0.91685`，净提升 `+0.00495`
- 这带来一个新的判断：
  1. `ViT` 至少不是“预训练 embedding 完全不适合当前任务”的方向；  
  2. 它在**不经过任何 backbone 微调**的情况下，不仅能逼近当前最佳离线协议，还在线上真正超过了 `IR101` 主线；  
  3. 这说明当前 public leaderboard 更偏好 `ViT` 带来的那少量决策修正，而不是大幅改变边界；  
  4. 因而 `ViT` 不再只是“值得继续投入”的候选，而是已经成为新的 strongest online baseline。  

### 当前综合判断（更新）

- 现在可以把方向判断再收紧一层：
  1. `exp_019` 仍然是当前线上最强基线，不能轻易浪费提交配额去做高风险跳跃；  
  2. `IR50` 与 `2 类 ArcFace` 已经给出负面信号，因此继续在“同家族 iResNet 容量切换”或“只训 Jesse/Mila”这两条线上投入，收益预期很低；  
  3. frozen `ViT` 已给出足够接近 `exp_019` 的离线行为，这说明下一步最值得做的是 **`ViT fine-tune pilot`**，而不是回退到 all-80 final strategy 或继续改造 scorer；  
  4. 这个 `ViT` 方向的价值不在于它已经证明会涨分，而在于它首次提供了一个**异构 backbone 且不明显退化**的新轴，值得继续用最小工程改动往前推一轮。  

### `exp_033` 当前结果

- `exp_033_vit_adaface_haar_3ep_lightft_prototype_recalib` 是对 `exp_032` 的直接跟进：
  - 不再冻结 `ViT`
  - 只用很小的 `backbone_learning_rate = 5e-6`
  - 训练 `3` 个 epoch
  - 同时检查阈值是否需要围绕 `0.55` 做小幅移动
- 这轮给出了两个清晰信号：
  1. **阈值确实发生了微调。**  
     最优阈值从 `0.55` 移到了 `0.525`。
  2. **但这种微调方向并不理想。**  
     相比 `exp_019`，新的 submission 只多改了 `12` 张，而且全部是把 `other` 放宽为目标类：
     - `10` 张 `0 -> 1`
     - `2` 张 `0 -> 2`
- 这说明：
  - `ViT` 轻量微调确实让模型在封闭集分类意义上更快贴合了当前任务；
  - 但这种贴合并没有转化成更强的 open-set prototype 行为；
  - 当前最直接的表现仍然是“把接受边界推松了一点”。  

### 当前综合判断（再次更新）

- 现在可以更明确地收敛到下面这几点：
  1. `exp_032 = 0.92180` 已经证明 frozen `ViT` 不是陪跑，而是当前最强线上方案；  
  2. `exp_033` 表明继续把 `ViT` 单模型往轻量 fine-tune 推进，并不会自然带来更好的 open-set 行为，反而更像把边界轻微放松；  
  3. 这意味着当前最值得保护的不是 `exp_019`，而是 `exp_032` 这条“frozen ViT + fixed 0.55”主线；  
  4. 下一步最值得尝试的，是以 `exp_032` 为 anchor，做 `exp_032 (ViT) + exp_019 (IR101)` 的异构融合，而不是继续单独深挖 `ViT fine-tune` 或回退到 all-80。  

### `exp_034` 当前结果

- `exp_034_ir101_vit_score_mean_fixed055` 采用最保守的首发融合方案：
  - `IR101 + frozen ViT`
  - score-level `mean`
  - 固定阈值 `0.55`
- 当前本地结果：
  - `val_accuracy = 0.9375`
  - 相比 `exp_032`，有 `7` 张预测变化：
    - `5` 张 `1 -> 0`
    - `1` 张 `2 -> 0`
    - `1` 张 `0 -> 2`
- 这说明首发的 `0.5/0.5` 线性平均并没有保护 `exp_032` 的收益，反而把它往 `IR101` 那条更保守的边界拉回去了。

### `exp_035` 当前结果

- `exp_035_ir101_vit_score_mean_vit075_fixed055` 进一步把权重偏向当前线上最优的 `ViT`：
  - `IR101:ViT = 0.25:0.75`
  - 其余协议不变
- 当前本地结果：
  - `val_accuracy = 0.9375`
  - 相比 `exp_032`，仍有 `5` 张预测变化：
    - `3` 张 `1 -> 0`
    - `1` 张 `2 -> 0`
    - `1` 张 `0 -> 2`
- 这比 `exp_034` 稍好，但方向没有变：
  - 线性均值融合依旧在吞掉 `exp_032` 的少量高价值接受。

### 当前综合判断（最终更新）

- 现在已经可以把融合方向也收紧：
  1. `exp_032` 的优势不是“大幅重排预测”，而是少量高价值修正；  
  2. 因而任何把分数往 `IR101` 拉回去的线性平均，都很容易把这些修正吞掉；  
  3. `exp_034` 与 `exp_035` 已经给出足够一致的证据，说明**当前不值得继续扫线性 mean fusion 权重**；  
  4. 如果还要继续做异构融合，更合理的下一步应是：
     - 更偏 `ViT` 的保守 gating
     - 或直接以 `exp_032` 为 anchor，只在少量高分歧样本上引入 `IR101` 作为二次裁决  
  5. 在没有更强证据前，`exp_032 = 0.92180` 仍应被视为当前最值得保护和提交对照的主线。  

### `exp_036` 当前结果

- `exp_036_vit_adaface_haar_10ep_last2block_ft_prototype_fixed055` 是第一次真正受控的 `ViT` 微调实验：
  - 不再像 `exp_033` 那样全开 `ViT` 参数
  - 只解冻最后 `2` 个 transformer blocks
  - 同时解冻 `norm` 与 `feature`
  - `backbone lr = 1e-5`
  - 推理协议保持 `prototype + fixed threshold 0.55`
- 本地结果为：
  - `best_val_acc = 1.0`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
- 与当前 strongest baseline `exp_032` 相比：
  - 只有 `2` 张预测变化
  - 两张都为 `0 -> 2`
- 这个结果的关键信息不是“它已经超过 `exp_032`”，而是：
  1. `exp_033` 失败并不能推出“ViT 微调无空间”；  
  2. 真正被证伪的是“全开 1.15 亿参数的轻量扰动”这条路；  
  3. 当 `ViT` 微调被限制在最后少量 blocks 时，模型行为重新回到与 `exp_032` 高度接近的稳定区间；  
  4. 因而当前单模型 `ViT` 线最值得继续探索的轴，不是更复杂融合，也不是重新放开全模型，而是**解冻块数 / 更保守学习率 / 固定阈值协议下的受控 partial fine-tune**。  

### `exp_036` 的阶段性判断

- 这轮离线结果不足以直接宣布 `exp_036` 优于 `exp_032`：
  - 因为它相对 `exp_032` 只多接收了 `2` 张 `Mila`
  - 方向上仍然是更松，而不是更保守
- 但它也明显不同于 `exp_033`：
  - 没有出现十几到几十张的系统性边界漂移
  - 这说明受控 `ViT last-block fine-tune` 已经是一个可以继续推进、也值得占用少量 Kaggle 配额验证的方向
- 因此当前最合理的结论是：
  1. `exp_032 = 0.92180` 仍是当前 strongest online baseline；  
  2. `exp_036` 是第一个**没有破坏 `exp_032` 主体行为**的单模型 `ViT` 微调版本，应视为下一条最合理的近邻候选；  
  3. 如果后续继续推进 `ViT` 单模型线，优先级应放在：
     - `last-1 / last-2 / last-3 blocks` 的小范围探索
     - 或维持 `last-2 blocks` 不变，只进一步收缩 `backbone lr`
     而不是回到全开微调或重新做线性融合。  

### `exp_037` 当前结果

- `exp_037_vit_adaface_haar_10ep_last2blockonly_ft_prototype_fixed055` 是对上一轮批判的直接验证：
  - 继续保留 `last-2 blocks`
  - 但冻结 `feature`
  - 同时冻结 `norm`
  - 从而把可训练参数真正压缩到 `blocks-only`
- 本地结果为：
  - `best_val_acc = 1.0`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
- 相比 `exp_036`：
  - 只差 `1` 张
  - 方向为 `2 -> 0`
- 相比 `exp_032`：
  - 也只差 `1` 张
  - 方向为 `0 -> 2`
- 这轮结果的价值高于表面上的“只改 1 张”，因为它回答了一个结构性问题：
  1. `exp_036` 的确不能被解释成“纯 last-block 微调”，因为 `feature` 参与更新时会把行为往更宽松的接收方向推；  
  2. 一旦冻结 `feature/norm`，模型立即比 `exp_036` 更接近 `exp_032`，这说明前一轮批判抓到了真实主因；  
  3. 因而当前真正成立的单模型 `ViT` 主线，不是“继续围绕 `feature` 做学习率小修”，而是**blocks-only partial fine-tune**。  

### `exp_037` 的阶段性判断

- 这轮还不足以宣布 `exp_037` 已经超过 `exp_032`：
  - 因为它仍保留 `1` 张 `0 -> 2`
  - 线上效果仍需 Kaggle 才能验证
- 但它已经把后续方向显著收敛了：
  1. “是否冻结 `feature`”这个问题基本已经被回答：应该冻结；  
  2. 下一步最有信息量的实验不再是继续讨论 `feature`，而是：
     - `blocks-only` 条件下的 `last-1 / last-3` 对照
     - 或和训练正交的 `TTA`
  3. 继续回到 `exp_036` 这种带 `feature` 的版本、继续扫线性融合、或继续做全开微调，优先级都已经明显下降。  

### `exp_038` 审计结果

- 用户提出的 `KP-RPE WebFace12M` 方向在方法层是有吸引力的，但第一轮接入审计已经证明：它**不是当前代码里的 drop-in backbone**。
- 已确认的事实有三条：
  1. 官方模型卡示例是 `model(input, keypoints)`，说明 `KP-RPE` backbone 前向依赖额外的 `5x2 keypoints`；  
  2. 官方示例还需要额外加载 `minchul/cvlface_DFA_mobilenet` aligner 来生成 keypoints；  
  3. `KP-RPE` 仓库依赖 `rpe_ops` C++/CUDA 扩展，当前环境中不存在可用的纯 Python fallback。  
- 当前机器上的环境阻塞也已经明确：
  - 缺少 `cl` / `g++` / `ninja`
  - 本机只有 `nvcc 11.0`
  - `gpu_env` 中的 PyTorch 是 `CUDA 12.1`
  - 因而 `rpe_ops` 编译直接失败，错误为 CUDA version mismatch
- 这意味着：
  1. `KP-RPE` 仍然可能是高上限方向；  
  2. 但当前回合的最大阻塞不是算法，而是**编译链与接口接线**；  
  3. 在这个环境里，不能把 `exp_038` 包装成一个已经跑过的 pilot，只能诚实地记录为“完成了接入审计，发现环境级 blocker”。  

### `exp_038` 之后的现实排序

- 如果用户愿意投入环境改造，这条线的正确顺序应是：
  1. 解决 `rpe_ops` 编译链
  2. 给 loader 增加 aligner 支持
  3. 把 keypoints 接进 dataset / model forward
  4. 再跑 frozen `KP-RPE WebFace12M` pilot
- 如果当前不想先碰系统级环境问题，那么更现实的下一条高杠杆实验仍是：
  - `IR101 WebFace12M` 这类真正 drop-in 的更大预训练数据源

### `exp_039` 当前结果

- `exp_039_ir101_adaface_haar_10ep_laststage_ft_webface12m_prototype_fixed055` 是对上面这条 drop-in 假设的直接验证：
  - backbone 仍是 `IR101`
  - 训练协议仍尽量复用 `exp_019`
  - 推理仍固定为 `prototype + threshold=0.55`
  - 唯一变化是把预训练权重从 `WebFace4M` 切到 `WebFace12M`
- 本地结果为：
  - `best_val_acc = 1.0`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
- 相比 `exp_019`：
  - 有 `264` 张预测变化
  - `263` 张为 `0 -> 1`
  - `1` 张为 `0 -> 2`
- 相比 `exp_032`：
  - 有 `257` 张预测变化
  - 仍几乎全部是把 `other` 放宽为目标类：
    - `254` 张 `0 -> 1`
    - `2` 张 `0 -> 2`
    - `1` 张 `2 -> 0`
- 这轮结果说明两点：
  1. `WebFace12M` 并不自动等价于“当前任务上更强的 open-set embedding”；  
  2. 在当前 `IR101 + prototype + 0.55` 组合里，它的主效应是把更多样本吸进 `Jesse`，而不是形成更干净的拒识边界。  
- 线上 Kaggle public score 现已确认是 `0.80341`，这与离线画像完全一致：
  - 变化不是少量高价值修正
  - 而是数百张级别的系统性放宽
  - 因此这轮可以视为对“`IR101 WebFace12M` 在当前协议下不可直接提交”的实证确认

### `exp_039` 的阶段性判断

- 这轮结果不足以支持“优先继续做 `IR101 WebFace12M`”：
  - 因为变化规模不是少量高价值修正，而是数百张级别的系统性边界放松
  - 其离线风险画像与 `exp_030 / exp_031` 更接近，而不是与 `exp_032 / exp_037` 接近
- 这并不等价于“更大预训练数据无用”，更稳妥的解释是：
  - `WebFace12M` 改变了 embedding 标尺
  - 但当前 `prototype + fixed 0.55` 协议并没有自动适配这条新标尺
  - 而在当前阶段，我们不希望为了它再回到大范围阈值搜索
- 因此当前更合理的下一步，不是继续深挖 `IR101 WebFace12M`，而是转向更正交的输入质量轴，例如：
  - `frozen ViT + MTCNN` 的重新验证
  - 或在现有 strongest baseline 周围做更小、更可解释的输入侧改动

### `exp_040` 当前结果

- `exp_040_vit_adaface_mtcnn_1ep_frozen_prototype_fixed055` 是对上面这条输入质量轴的直接验证：
  - backbone 仍是 frozen `ViT`
  - 预训练权重仍是 `WebFace4M`
  - 推理仍固定为 `prototype + threshold=0.55`
  - 唯一主变量是 `HAAR -> MTCNN + 5 点对齐`
- 本地结果为：
  - `best_val_acc = 0.75`
  - `prototype val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
- 相比 `exp_032`：
  - 仅 `1` 张预测变化
  - 方向为 `1 -> 0`
- 相比 `exp_019`：
  - 共 `12` 张变化
  - 整体画像几乎就是 `exp_032` 再额外收紧 `1` 张 `Jesse`
- 这轮结果回答了一个重要问题：
  1. 早期 `exp_008` 的 `MTCNN` 负例不是“以后都别碰 MTCNN”；  
  2. 在当前 frozen `ViT` 主线上，`MTCNN` 不会像早期 IR101 那样把大量目标样本压回 `other`；  
  3. 它表现为一个**极小幅、偏保守**的输入侧改动，而不是灾难性方向。  

### `exp_040` 的阶段性判断

- 这轮还不能称为“新的突破”：
  - 因为相对 strongest baseline `exp_032` 只改了 `1` 张
  - 离线没有给出足够大的新信息量
- 但它是一个比 `exp_039` 健康得多的结果：
  - 没有出现系统性放宽边界
  - 也没有重演早期 `MTCNN` 的大规模崩塌
- 因此当前最稳妥的结论是：
  1. `exp_032 = 0.92180` 仍然是主线；  
  2. `exp_040` 是一个值得提交验证的**近邻候选**；  
  3. 如果线上继续没有增益，再决定是否继续深挖 `MTCNN` 这条输入轴。  

### `exp_041` 当前结果

- `exp_041_vit_adaface_haar_1ep_frozen_prototype_fixed055_hfliptta` 是对 frozen `ViT` 主线做的第一条低风险推理增强：
  - 不改 checkpoint
  - 不改阈值
  - 不改 prototype 协议
  - 只在 embedding 提取时做 `original + horizontal flip` 平均
- 本地结果为：
  - `val_accuracy = 0.9375`
  - `selected_threshold = 0.55`
  - `tta_horizontal_flip = true`
- 相比 `exp_032`：
  - 仅 `1` 张预测变化
  - 方向为 `0 -> 2`
- 这轮结果说明：
  1. `horizontal flip TTA` 在当前 strongest baseline 上不会引入大规模边界漂移；  
  2. 它带来的信号是健康但极弱的；  
  3. 因而它更适合作为后续结构性方法的 base embedding 增强层，而不是单独冲榜的主方法。  

### `exp_042` 当前结果

- `exp_042_vit_adaface_haar_1ep_frozen_graphrefine_fixed055_hfliptta` 是第一版保守式图修正：
  - 基于 `exp_041` 的 `frozen ViT + hflip TTA` embedding
  - 只对低边际样本开放修正
  - labeled gallery 始终为硬锚点
  - `0 -> 1/2` 的接受门槛高于 `1/2 -> 0`
- 本地结果为：
  - `base_val_accuracy = 0.9375`
  - `refined_val_accuracy = 0.9375`
  - `num_changed_test_predictions = 0`
- 这轮结果说明两件事：
  1. 第一版 conservative graph refinement 没有触发 look-alike 污染；  
  2. 但当前 gating 明显过于保守，导致它在测试集上完全没有动作。  
- 因此当前不应把 `exp_042` 视为负例，而应视为“安全边界已经找到，但参数区间还没进入有效修正带”。如果继续走这条线，下一步应优先放松 gating 设计，而不是直接换回激进传播。

### `exp_043` 当前结果

- `exp_043_vit_adaface_haar_1ep_frozen_globallabelspread_hfliptta` 是对“全局图结构是否能修正高置信错例”的第一版直接验证：
  - base embedding 仍是 `frozen ViT + hflip TTA`
  - 不再限制于边界样本
  - 直接在 `gallery + test` 的全局 cosine kNN 图上做 label spreading-style 传播
- 本地结果为：
  - `val_accuracy = 1.0`
  - `alpha = 0.2`
  - `top_k = 20`
- 相比 `exp_032`：
  - 有 `153` 张预测变化
  - `115` 张为 `0 -> 2`
  - `38` 张为 `0 -> 1`
- 这轮结果说明：
  1. 当前错误并不只存在于 threshold 附近，图结构确实能推动大量高置信样本改判；  
  2. 因而 `exp_042` 的“0 变化”并不代表图方法无效，只代表那版 gating 太局部、太保守；  
  3. 但第一版全局传播的实际方向又重新落回“系统性放宽接受边界”，这与 `exp_030 / exp_031 / exp_039` 的风险画像高度相似。  
- 因此 `exp_043` 是一个**有信息量但当前不可提交**的实验：
  - 好消息：全局方法终于能真正动到高置信样本；
  - 坏消息：当前动法几乎全是 `0 -> 1/2` 扩张，没有任何保守修正。

### `exp_044` 当前结果

- `exp_044_vit_adaface_haar_1ep_frozen_lookalikemargin_hfliptta` 是对“look-alike margin rejection 是否能捕捉高置信相似人误接收”的第一版验证：
  - 复用 `exp_029` 的 `michael_like / sarah_like` clustered prototype
  - 目标类先过固定阈值 `0.55`
  - 再用 `target score - matched look-alike score` 做拒识
- 本地结果为：
  - `selected_margin = 0.01`
  - `val_accuracy = 0.9375`
- 相比 `exp_032`：
  - 仍然只有 `1` 张变化
  - 方向是 `0 -> 2`
- 这轮结果说明：
  1. 在当前 frozen `ViT` embedding 上，简单的 look-alike margin rejection 还没有形成足够强的修正信号；  
  2. 它没有像 `exp_043` 那样真正触发全局改判；  
  3. 因而这条线当前更像一个弱对照，而不是主突破方向。  

### `exp_045` 当前结果

- `exp_045_vit_adaface_haar_1ep_frozen_spectralcluster_hfliptta` 是对“完全绕开阈值，直接用全局聚类结构恢复三类”的第一版验证：
  - 用 `4` 个 spectral clusters 划分 test
  - 再通过 gallery 近邻给每个 cluster 匹配 `0/1/2`
- 本地结果为：
  - `val_accuracy = 0.9375`
- 相比 `exp_032`：
  - 有 `494` 张预测变化
  - `382` 张是 `2 -> 0`
  - `75` 张是 `0 -> 2`
  - `37` 张是 `0 -> 1`
- 这轮结果有一个重要新信号：
  1. 它终于不是“纯 0 -> 1/2 扩张”；  
  2. 它确实产生了大规模 `1/2 -> 0` 的保守方向修正；  
  3. 但当前强度过头，几乎把 `Mila` 整体决策边界重写成了保守版本。  
- 因此 `exp_045` 是一个**值得继续研究其结构，但当前版本明显过激**的实验：
  - 它证明“全局聚类”这条轴不是死路；
  - 但第一版的 cluster-to-label matching 还太粗，无法直接作为提交候选。

### `exp_046` 当前结果

- `exp_046_vit_adaface_haar_1ep_frozen_pcawhitenedproto_hfliptta` 是对 `PCA whitening + prototype` 的直接验证。
- 本地结果为：
  - `val_accuracy = 0.3125`
  - `n_components = 32`
- 相比 `exp_032`：
  - 有 `671` 张预测变化
  - `289` 张是 `1 -> 0`
  - `382` 张是 `2 -> 0`
- 这轮结果说明：
  1. PCA whitening 不是“零成本小修正”，而是在当前 embedding 空间里引入了非常强的几何扭曲；  
  2. 扭曲的结果几乎完全是把目标类压回 `other`；  
  3. 因而在当前 frozen `ViT` 主线上，PCA whitening 可以视为明确负例。  

### `exp_047` 当前结果

- `exp_047_vit_adaface_haar_1ep_frozen_neighborhoodaware_fixed055_hfliptta` 是对 `neighborhood-aware scoring` 的直接验证：
  - `base_score = best prototype similarity`
  - `neighbor_mean_score = top-15 test-test 邻域的 base score 均值`
  - `final_score = 0.5 * base_score + 0.5 * neighbor_mean_score`
  - `threshold = 0.55`
- 本地结果为：
  - `val_accuracy = 0.9375`
  - `mean_test_base_score = 0.4637`
  - `mean_test_neighbor_score = 0.5085`
  - `mean_test_final_score = 0.4861`
- 相比 `exp_032`：
  - 有 `8` 张预测变化
  - `3` 张是 `0 -> 1`
  - `5` 张是 `0 -> 2`
- 这轮结果说明：
  1. 邻域信息在当前任务上确实能提供少量额外有效信号；  
  2. 和 `exp_043` 那种全局传播不同，这种局部 score 融合没有退化成大规模边界扩张；  
  3. 变化量级和方向都符合当前 public leaderboard 已经验证过的“小而准修正”模式。  
- Kaggle public score 已确认是 `0.92621`，超过 `exp_032 = 0.92180`。  
- 因而 `exp_047` 已成为当前 strongest online baseline（后被 **exp_061** 超越，见下）。

### `exp_061` ViT CE 微调 + neighborhood_aware（2026-04-02）

- 实验名：`exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta`。
- 训练：`CrossEntropy`；解冻最后 **2** 个 ViT block + `norm` + `feature`；`learning_rate=3e-4`、`backbone_learning_rate=1e-5`；增强含 **HorizontalFlip**、**Affine**（`transforms.build_train_transform` 中已含 ColorJitter）；`monitor=val_acc / max`；`max_epochs=20`、`early_stopping_patience=6`，实际约 **epoch 9** 早停；`best_val_acc=1.0`。
- 推理：与 `exp_047` 相同协议——`neighborhood_aware`、`top_k=15`、`base_weight=0.5`、`threshold=0.55`、**hflip TTA**。
- submission：`data/submissions/20260402_224630_exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv`。
- 与 `exp_047` submission 相比：约 **2 / 1816** 条测试样本类别不同。
- Kaggle public score：**`0.92731`**，超过 `exp_047 = 0.92621` 约 **`0.00110`**。
- **结论**：`exp_061` 现为 **strongest online baseline**。说明在固定推理锚点下，**受控 CrossEntropy 微调** 能改善 embedding，与 **exp_060**（3-class ArcFace、提交与 frozen 线一致）形成负面对照。

### 方向 C：`adaptive_neighborhood_aware`（2026-04-02）

- **代码**：`neighborhood_aware_predictions_adaptive_threshold`（`src/dl_pipeline/inference/prototype.py`）在原有 `final_score` 上，用 query 的 top-k 近邻与当前 argmax 标签的**一致比例** `agree_frac`，设逐样本阈值 `effective_th = th_max - (th_max - th_min) * agree_frac`（邻居越一致，阈值越低）。
- **编排**：`scripts/predict.py` 新增 `inference.mode: adaptive_neighborhood_aware`；验证集仍用 **train-only prototype** 算 `val_accuracy`；测试集 gallery 起始为 **train+val**，可选 `pseudo_label_refinement`：将 `final_score >= min_final_score` 且非 `other` 的测试样本并入 gallery，重算 prototype 后再跑一轮（可多轮至 `max_rounds`）。
- **配置示例**：`configs/experiments/exp_062_vit061_adaptive_neighborhood_pseudo2_hfliptta.yaml`（权重 `checkpoint_source_experiment: exp_061_...`；默认 `th_min=0.45`、`th_max=0.65`；伪标签一轮 `max_rounds: 2`）。
- **运行**：`python scripts/predict.py --config configs/experiments/exp_062_vit061_adaptive_neighborhood_pseudo2_hfliptta.yaml`（需本机存在 `exp_061` 的 `metrics.json` / checkpoint）。
- **单测**：`tests/test_prototype_inference.py::test_adaptive_threshold_matches_fixed_when_th_min_equals_th_max`（`th_min == th_max` 时与固定阈值行为一致）；若本地 NumPy/SciPy 与 sklearn 不兼容，需在可用环境中重跑 pytest。

### `exp_062` 当前结果（Kaggle public）

- 实验名：`exp_062_vit061_adaptive_neighborhood_pseudo2_hfliptta`；复用 **exp_061** checkpoint；推理为 `adaptive_neighborhood_aware`（`th_min=0.45`、`th_max=0.65`）+ 伪标签 refinement（`min_final_score=0.72`、`max_rounds=2`）。
- submission：`data/submissions/20260402_231317_exp_062_vit061_adaptive_neighborhood_pseudo2_hfliptta_submission.csv`。
- 与 **exp_061** submission 按 `id` 对齐：共 **6 / 1816** 条不同，全部为 **`0 -> 1`（4 条）或 `0 -> 2`（2 条）**；预测分布由 `{0:1072, 1:357, 2:387}` 变为 `{0:1066, 1:361, 2:389}`。
- Kaggle public score：**`0.92731`**，与 **exp_061** 相同；说明在当前线上划分下，自适应阈值 + 伪标签未带来额外 public 增益。
- **结论**：线上最强仍为 **exp_061**（与 exp_062 并列分数）；方向 C 可作为方法记录，提交优先级不高于 exp_061。

### `exp_071` / `exp_077`：IResNet50（Glint360K）+ neighborhood_aware（2026-04-03）

- 与 **exp_047 / exp_061** 使用相同的推理协议：`neighborhood_aware`、`threshold=0.55`、`top_k=15`、`base_weight=0.5`、水平翻转 TTA；backbone 为 InsightFace 风格 **IResNet50**，预训练权重 `16backbone.pth`（Glint360K），`CrossEntropy` + 线性头，**Weight Imprinting** 初始化头；`batch_size=8`、`accumulate_grad_batches=8`。
- **`exp_071`**：`iresnet_finetune_mode` 为 **bn_only**（仅 BN 可训）；`max_epochs=20`；`learning_rate=2e-4`、`backbone_learning_rate=5e-4`。
  - submission：`data/submissions/20260403_154045_exp_071_arcface_r50_glint360k_bnonly_wi_accum8_20ep_neighborhoodaware_fixed055_hfliptta_submission.csv`
  - **Kaggle public score：`0.89207`**
- **`exp_077`**：**last_stage**（layer4 + bn2 + fc + features）；较 exp_073 **降低** head / backbone LR，并设 **`gradient_clip_val=1.0`**；`max_epochs=35`、`early_stopping_patience=12`。
  - submission：`data/submissions/20260403_163843_exp_077_arcface_r50_laststage_mildlr_clip_wi_accum8_35ep_neighborhoodaware_fixed055_hfliptta_submission.csv`
  - **Kaggle public score：`0.91134`**
- **对照**：`exp_077` 较 `exp_071` **+0.01927**（public）；二者均 **低于 `exp_061 = 0.92731`**。结论：在固定阈值与邻域设定下，**IResNet 该训练配方仍整体弱于 ViT CE 微调线**，但 **适度解冻 last stage + 温和 LR + 梯度裁剪** 相对 **仅 BN** 可复现线上提升。

### `exp_078` / `exp_079` / `exp_080`：三路异构 submission 投票正规化（2026-04-03）

- 这轮不是继续发明新 scorer，而是把此前未入库的 `exp_069` 草稿结果补成可复现实验入口：
  - 新增脚本：`scripts/predict_submission_vote.py`
  - 输入固定为三路已完成 submission：
    1. `exp_061`：`ViT AdaFace + CE + neighborhood_aware`
    2. `exp_067`：`buffalo_l detect_align dual verifier`
    3. `exp_068`：`AdaFace ViT dual verifier`
- 正式生成了三种标签级融合：
  - `exp_078_vote_majority_061_067_068`
  - `exp_079_vote_conservative_061_067_068`
  - `exp_080_vote_diverse_061_067_068`

#### 本轮最关键的事实

- 三种策略在当前三路输入上**完全塌缩成同一份 submission**：
  - 三者预测分布一致：`{0: 1055, 1: 361, 2: 400}`
  - `exp_078` 与旧 `exp_069_diverse_ensemble` **0 diff**
- 这意味着：
  1. 当前三路模型在标签层的结构关系比预想中更简单；
  2. “多数投票 / 保守投票 / 多样性优先”在当前样本分布上并没有形成不同决策面；
  3. 因而如果继续深挖这条线，下一步不能再只是换投票规则名字，而必须引入**分数级**或**样本级 gating**，否则只会重复得到同一份 submission。

#### 与各基线的对比

- 相对 `exp_061`：
  - 共 `50 / 1816` 条不同
  - 迁移为：
    - `0 -> 1`: `11`
    - `0 -> 2`: `15`
    - `1 -> 0`: `6`
    - `1 -> 2`: `8`
    - `2 -> 0`: `3`
    - `2 -> 1`: `7`
- 相对 `exp_067`：
  - 共 `95` 条不同
  - 主要是把 `buffalo_l` 的更激进目标类接受收回：
    - `1 -> 0`: `44`
    - `2 -> 0`: `29`
- 相对 `exp_068`：
  - 仅 `13` 条不同
  - 说明这轮投票融合的实际主导面，更接近 `AdaFace ViT dual verifier`

#### 阶段性判断

- 这轮结果有价值，但要解释准确：
  - 它不是“新的已证实最优方法”
  - 它是**第一版可复现的异构标签级融合候选**
- 从实验管理角度，这轮是值得保留和必要的，因为它回答了一个之前悬而未决的问题：
  - 旧 `exp_069` 草稿不是偶然文件，而是可被稳定复现的
- 从 Kaggle 提交优先级看：
  - 它**值得用一次配额验证**
  - 但不是高置信度“应该直接替换 `exp_061`”的候选
- 若这轮线上不涨，下一步应明确停止继续扫 label-vote 变体，转向：
  1. `exp_061 / exp_067 / exp_068` 的**分数级融合**
  2. 仅在三模型分歧样本上生效的**样本级 gating / meta-decider**

### `exp_081` / `exp_082`：label vote 证伪后，061+067 score-level gate 的负结果（2026-04-03）

- 用户已回报 **`exp_081_vote_majority_061_067_068_fixedcsv` 的 Kaggle public score = `0.90143`**。这是一个强负例：
  - 说明三路模型的互补性**不能**通过 submission 标签多数投票直接兑现
  - 也说明 `exp_061` 必须被视为**强锚点**，不能在标签层被 `067/068` 对称覆盖
- 基于这个结论，本轮实现了真正的 **score-level / sample-wise gating**：
  - 新增门控逻辑模块：`src/dl_pipeline/inference/gated_fusion.py`
  - 新增实验脚本：`scripts/predict_061_067_gated_fusion.py`
  - 规则是：
    1. `exp_061` 仍负责主决策
    2. 仅当 `061` 低置信时，允许 `067` 做三类动作：`other -> target` rescue、`target -> target` swap、`target -> other` veto
    3. `067` 的可信度不看最终标签本身，而看 **selected excess / best excess**（相对其 per-identity threshold 的超额）

#### 为何这版 gate 比之前更严谨

- `exp_061` 侧严格复现原线上协议：
  - **val**：`train` prototype
  - **test**：`train+val` prototype
- `exp_067` 侧不再沿用原日志里“all_labeled 自己给自己打 val 分”的宽松口径，而改为：
  - **val**：`train` gallery + `train` look-alike 阈值校准
  - **test**：`train+val` gallery + `train+val` 阈值校准
- 这一步很关键，因为它让 gate 的参数搜索不再建立在一个自泄漏的 `067 val_acc` 上。

#### 结果

- `exp_061` / `exp_067` 的 test 预测都与历史正式 submission **0 diff**，说明这轮复算没有偏移。
- `exp_082_vit061_buffalol067_gated_score_fusion_valgrid`：
  - val：
    - `exp_061 = 0.9375`
    - `exp_067 (train-only gallery) = 0.8750`
    - **fused = 0.9375**
  - 网格：`low_conf_threshold × rescue_margin × swap_margin × veto_excess_threshold = 400` 组
  - 最优参数对应：
    - `low_conf_threshold = 0.50`
    - `rescue_margin = 0.03`
    - `swap_margin = 0.05`
    - `veto_excess_threshold = -0.05`
  - 但这个“最优”解在 val 上 **对 `exp_061` 完全没有改动**（`num_val_changed_vs_061 = 0`）

#### 这说明什么

- 这条线最大的失败点不是“线上没试”，而是 **离线已经不给你正信号**：
  - 在 val 上，任何 gate 都没能超过 `exp_061`
  - 也就是说，`067` 在当前规则形态下，并没有在这个验证口径里提供可利用的稳定补充信息
- 更危险的是：虽然 val 完全不动，test 上 gate 却会大量触发
  - `exp_082` 相对 `exp_061` 改了 **99 / 1816** 张
  - 全部大头都是：
    - `0 -> 1`: `55`
    - `0 -> 2`: `44`
  - 分布从 `{0:1072, 1:357, 2:387}` 变成 `{0:973, 1:412, 2:431}`
- 这是一种非常不健康的模式：
  - **验证集零收益**
  - **测试集大规模扩大目标类接受**
  - 本质上仍然是 `067` 把 `061` 的 `other` 边界顶开了，只是从 label vote 改成了 heuristic gate

#### 二次审查后的更强结论

- 对全部 400 组参数做二次筛查后：
  - **没有任何组合** 在 val 上超过 `0.9375`
  - 即使把 `rescue_margin` 一路调大，test 改动只是从 `90+` 慢慢降到 `42`，但**始终没有带来 val 增益**
- 所以这轮结果已经足够下结论：
  - `061 + 067` 的**手工 heuristic gate** 不是当前通往 `0.97` 的主线
  - 它没有把互补性转化成可验证的离线收益
  - 如果继续在这个规则空间里扫，只会得到“不同强度的 067 注入量”，不会得到真正可解释的性能提升

#### 下一步启示

- 该停止的不是“异构融合”本身，而是：
  - submission label vote
  - 手工阈值式 heuristic gating
- 仍然值得继续的，是更高一层的融合形态：
  1. **OOF / CV score stacking**
  2. **disagreement-only meta-decider**
  3. **更强外部数据驱动的 verification 校准**

### `exp_082` 线上突破后，`exp_083` / `exp_084`：rescue-only + class-aware + labeled LOO（2026-04-03）

- 用户随后回报：**`exp_082` Kaggle public score = `0.94768`**。
- 这条结果非常重要，因为它说明：
  - 我们此前对 `exp_082` 的离线风险判断并没有错，但**固定 val 明显低估了这类 rescue 机制**
  - 真正有效的不是通用 gate，而是：**用 `067` 有条件地救回 `061` 的 false reject**

#### 因此，本轮策略发生了一个明确收缩

- 停止继续沿着 `swap / veto` 扩张
- 只保留最有证据的动作：`061 == other` 时的 **rescue-only**
- 同时把参数从全局统一改成：
  - `rescue_margin_jesse`
  - `rescue_margin_mila`
- 更关键的是，把参数搜索从固定 16 张 val，切换到 **80 张 labeled 的 leave-one-out (LOO)** 口径

#### 为什么要换成 labeled LOO

- 在固定 val 上，`061 == 0` 且 `067 != 0` 的 rescue 子集是 **0 个样本**
- 这意味着旧 val 根本无法评估 rescue-only 机制
- 继续在那 16 张上扫参数，只会得到伪稳健结论
- LOO 虽然也不是完美终极答案，但至少让参数搜索真正落在“会发生 rescue 的样本”上

#### `exp_083`：第一版 rescue-only 候选

- 新脚本：`scripts/predict_061_067_rescueonly_labeledloo.py`
- 新逻辑：
  - 仅当 `exp_061` 判 `other`
  - 且 `exp_067` 的对应身份 excess 超过类特异 margin
  - 才允许 `other -> Jesse/Mila`
- `exp_083` 的最优参数为：
  - `low_conf_threshold = 0.45`
  - `rescue_margin_jesse = 0.06`
  - `rescue_margin_mila = 0.06`
  - `require_anchor_argmax_match = false`
- **LOO**：
  - `exp_061 = 0.9250`
  - `exp_067 = 0.9125`
  - **fused = 0.9625**
- 这个信号比固定 val 强得多，说明 rescue-only 机制在更广的 labeled 口径下确实成立

#### 但 `exp_083` 仍偏激进

- 虽然 LOO 上只改了 **3 张**
- 但 test 上相对 `exp_061` 仍改动了 **91** 张
- 相对 `exp_082` 只收回了 **8** 张
- 这说明：直接拿“第一个 LOO 最优参数”上线，仍然太激进

#### plateau 现象与 `exp_084`

- 对完整网格进一步检查后发现一个很有价值的结构：
  - 大量参数组合在 **LOO 上达到完全相同的最优精度 `0.9625`**
  - 且同样只改动 **3** 个 labeled 样本
  - 但是它们在 test 上的改动量差异很大：从 **91** 到 **64**
- 这说明 rescue-only 参数空间里存在明显 plateau：
  - **离线同分**
  - **线上风险不同**
- 因此补跑了更保守的 plateau 候选 **`exp_084`**：
  - `low_conf_threshold = 0.45`
  - `rescue_margin_jesse = 0.10`
  - `rescue_margin_mila = 0.30`
  - `require_anchor_argmax_match = false`
- 它与 `exp_083` 在 **LOO 上完全同分**：
  - `fused = 0.9625`
  - `num_changes = 3`
- 但它在 test 上更收缩：
  - 相对 `exp_061`：改动 **64** 张（低于 `exp_083` 的 91）
  - 相对 `exp_082`：收回 **35** 张
    - `2 -> 0`: `26`
    - `1 -> 0`: `9`

#### 这轮的真正收获

- 不是“又找到一个更高分 submission”，因为线上还没回报
- 而是我们把 `exp_082` 的成功机制拆解得更清楚了：
  1. 成功来自 **false reject rescue**
  2. rescue-only 在更广 labeled 口径下是有离线支持的
  3. 该机制存在明显 parameter plateau，所以**保守选点**是必要的

#### 当前最合理的在线候选顺序

- 若只提交一版新的验证候选：
  - **优先 `exp_084`，而不是 `exp_083`**
- 理由很直接：
  - `exp_083` 与 `exp_084` 在 LOO 上同分
  - `exp_084` 对 `exp_082` 更保守、收回更多低把握 rescue
  - 因而它更适合作为“同机制、更稳健”的下一跳验证

### `exp_048` / neighborhood 参数网格（2026-04-02）

- 已运行 `scripts/sweep_neighborhood_aware.py`（`gpu_env`），在固定 `exp_047` 协议（`threshold=0.55`、TTA、train+val prototype）下扫描 `top_k ∈ {5,10,15,20,30}` 与 `base_weight ∈ {0.3,0.4,0.5,0.6,0.7}`。
- **判断标准应以 baseline `submission.csv` 为准**：候选预测必须与 baseline **按 `id` merge** 后再统计 `0->1`、`2->0` 等迁移；不可假设 test DataLoader 行序与 submission 排序一致。
- **选参不应以小验证集为主排序**（与作业数据规模一致）；脚本已改为先最小化 submission 上的 `(0->1+0->2)`，再最小化总变化数，再贴近 `exp_047` 超参。自动 winner 为 **`(top_k=15, base_weight=0.5)`**，与 `exp_047` 一致。
- **没有任何组合的验证集准确率高于 `0.9375`**（旁注）；降低 `base_weight` 在多个 `top_k` 上仍出现大规模 `0 -> 1/2`，与历史负例一致。
- 与 `exp_047` baseline **预测完全一致** 的参数仍构成一片 plateau；保守外推候选 `top_k=5, base_weight=0.6`（相对 `047` 仅 1 张 `2->0`）已提交 Kaggle：**public score `0.92511`**，低于 `exp_047 = 0.92621`。该张 `2->0` 在 public 上为错误收紧，小验证集无法预见。后续 **exp_061** 已以 **`0.92731`** 成为新的最强线上基线；此组邻域超参扫描结论仍适用于「仅 frozen checkpoint」语境，不再单独推进 `exp_048` 式保守外推。
