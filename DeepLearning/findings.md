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
