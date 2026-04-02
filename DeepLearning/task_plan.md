# DeepLearning 子项目任务计划

## 项目目标

围绕 `CV_GA1(1).pdf` 中第 2.3 部分，建立一个独立于队友其余部分的深度学习子项目工作区，用于两人协作、记录方案迭代、管理实验过程，并为后续 Kaggle 榜单冲刺提供稳定工程基础。

## 当前约束

- 工作目录固定为 `Computer Vision/assignment/Group/DeepLearning`
- 需要单独分支 `DL`
- 使用 `pi-planning` 方式持续记录
- 目录结构需要工程化拆分，避免重复造轮子
- 代码注释使用中文
- 需要准备 `.gitignore`
- 当前阶段先做项目起步，不直接开始实现模型训练细节
- 开发策略采用 `script-first`
- 迭代期只通过 `csv` 提交 Kaggle，不先处理最终 notebook
- 两人协作方式是共享同一条 pipeline，轮流迭代，而不是长期按模块分治维护

## 阶段规划

| 阶段 | 状态 | 内容 |
|---|---|---|
| Phase 0 | completed | 读取 AGENTS、PDF、仓库现状，确认约束 |
| Phase 1 | completed | 明确 2.3 的任务边界、输入输出、评价指标与提交格式 |
| Phase 2 | completed | 设计深度学习子项目的协作流程、目录结构与迭代节奏 |
| Phase 3 | completed | 与用户确认设计方案 |
| Phase 4 | completed | 创建分支 `DL`、初始化项目骨架与 `.gitignore` |
| Phase 5 | completed | 落地第一版可运行基线并建立实验记录规范 |
| Phase 6 | completed | 建立 `IR101 + AdaFace` 人脸专用 backbone 主线并完成部分微调实验 |
| Phase 7 | completed | 建立 `open-set prototype + fixed threshold` 统一评测协议 |
| Phase 8 | completed | 完成 `exp_023 ~ exp_027` 的多种开放集推理变体探索并记录负例 |
| Phase 9 | in_progress | 在有限 Kaggle 配额下筛选“值得提交”的高杠杆方向 |
| Phase 10 | completed | 诊断 `other` 的 look-alike 结构，并完成零新超参数的 4-prototype 拒识诊断 |
| Phase 11 | in_progress | 基于 look-alike 诊断结果，筛选下一条真正会改变线上预测的高杠杆方向 |

## 决策记录

- 计划文件放在 `DeepLearning` 子目录中，和后续代码同级，便于该子项目独立协作。
- 2.3 暂按“独立深度学习增强线”规划：与 2.1/2.2 主要是结果对比关系，而不是代码共建关系。
- 工程开发采用 `script-first`，最终 notebook 暂不纳入当前起步阶段。
- Kaggle 迭代阶段优先采用 `csv` 提交，等方案收敛后再回填 notebook。
- 协作方式采用共享 pipeline、轮流迭代，不按子模块永久分工。
- 当前最强线上结果是 `exp_047 = 0.92621`，对应协议为：`frozen ViT AdaFace + hflip TTA + neighborhood-aware scoring + fixed threshold 0.55`。
- `exp_048` 已完成对 `neighborhood_aware` 的 `top_k × base_weight` 全网格离线扫描（`reports/sweeps/neighborhood_aware_grid_latest.json`）；**迁移统计与选参均以 baseline submission 按 `id` 对齐为准**；自动 winner 与 `exp_047` 一致 `(15, 0.5)`。保守候选 `top_k=5, bw=0.6` 已提交 Kaggle，**public `0.92511` < `exp_047` 的 `0.92621`**，已证伪为线上主分支。
- `threshold = 0.55` 视为当前锁定的评测协议；在模型探索阶段不再继续自由搜索阈值。
- `exp_023 ~ exp_027` 已验证多种更复杂的开放集推理思路，但当前实现大多会把过多 `other` 放宽为目标类，因此暂不消耗 Kaggle 配额继续提交。
- 当前阶段的实验优先级应从“小幅边界调参”转为“只有明显不同的方法层变化才值得提交”，例如 score/model fusion 或训练目标对齐。
- `_cv_ga1_extracted.txt` 已确认 `other` 并非随机陌生人，而是 `Michael Cera` 与 `Sarah Hyland` 两组 look-alike；这意味着当前任务更接近“相似人拒识”而不是泛化开放集拒识。
- `exp_028` 的同 backbone 多 checkpoint 平均没有形成有效互补，说明继续堆同源 checkpoint 不是当前最高优先级。
- `exp_009` embedding 上对 `other` 做 `k=2` 聚类后，得到两个各 `10` 张的清晰簇，且分别显著靠近 `Jesse` 与 `Mila`；后续优先验证 look-alike-aware 推理，再考虑 4 类训练。
- `exp_029` 已完成零新超参数的 look-alike-aware `4 prototype` 推理诊断，但其 submission 与 `exp_019` 完全一致，说明当前 `0.55` 协议下显式拆分 `michael_like / sarah_like` 并不会改变任何测试预测。
- `exp_030` 的 2 类 ArcFace pilot 已被线上 `0.76927` 明确证伪；它在当前实现下明显放宽了接受边界：相对 `exp_019` 有 `278` 张预测变化，且全部是 `0 -> 1/2`。
- `exp_031` 的 `IR50` 替换实验也给出负面离线信号：相对 `exp_019` 有 `237` 张预测变化，几乎全部是 `0 -> 1`，说明同家族 iResNet 容量切换不是当前主杠杆。
- `exp_032` 的 frozen `ViT AdaFace` prototype pilot 给出了和 `exp_019` 非常接近的离线行为：仅有 `11` 张预测差异，且没有出现大规模 `0 -> 1/2` 漂移，说明 `ViT` 作为异构 backbone 值得继续投入。
- `exp_032` 已在线上拿到 `0.92180`，超过 `exp_019 = 0.91685`；这说明 frozen `ViT` 虽然离线只改动了 `11` 张，但这些改动的质量很高，当前 public leaderboard 对这类少量修正非常敏感。
- `exp_033` 的 `ViT` 轻量微调 pilot 虽然把分类验证精度从 `0.75` 提到 `0.9167`，但 prototype 侧只带来 `12` 张相对 `exp_019` 的变化，而且全部是 `0 -> 1/2`；同时阈值从 `0.55` 微调到 `0.525`，说明轻量微调的主要效果仍然是放宽接收边界，而不是形成更稳的开放集分离。
- `exp_034/035` 的 `IR101 + frozen ViT` 线性 score fusion 已完成首轮验证；无论是 `0.5/0.5 mean` 还是 `0.25/0.75 mean`，都会把 `exp_032` 的一部分线上潜在收益往回拉，说明当前最优解更接近“保护 ViT 的少量高价值修正”，而不是简单做线性平均。
- `exp_036` 已完成第一次受控的 `ViT partial fine-tune`：只解冻最后 `2` 个 transformer blocks，并保持 `threshold = 0.55`。它相对 `exp_032` 仅多出 `2` 张 `0 -> 2` 的变化，没有出现 `exp_033` 那种明显的边界放宽漂移，说明真正被证伪的是“全开 ViT 的轻量乱动”，而不是“所有 ViT 微调都无效”。
- `exp_037` 进一步把 `ViT partial fine-tune` 收紧成真正的 `blocks-only` 版本：冻结 `feature` 与 `norm`，只解冻最后 `2` 个 blocks。结果相对 `exp_036` 回收了 `1` 张 `2 -> 0`，相对 `exp_032` 只剩 `1` 张 `0 -> 2`，这直接支持了“`exp_036` 的主导更新来自 feature 层”这一判断。

## 当前阶段目标

- 保持 `exp_032` 作为当前 strongest baseline，不轻易用小改动消耗 Kaggle 日提交次数。
- 所有新推理实验先做本地离线对比：重点看与 `exp_032` 的预测差异规模和方向，先过滤掉明显“过度接收 other”的方案。
- 对 look-alike-aware 方案，优先采用“零新超参数”的诊断版规则，避免再次在 16 张验证集上搜索 margin/threshold。
- 既然 `exp_029` 与 `exp_019` 完全同预测，则当前约束下不再继续打磨同类 look-alike inference 规则；后续必须转向会真正改变 embedding 或训练目标的方案。
- `exp_030` 已说明“直接上 2 类 ArcFace + 现有 prototype 规则”会把大量 `other` 收进目标类，因此后续若继续走 metric-aligned training，需要更强的拒识约束或更保守的训练/推理配套设计。
- 当前下一条训练线的优先级应转向“显式利用 `other` 结构做排斥”，优先于继续做 `2 类 ArcFace` 变体。
- 当前如果继续走“换 backbone”这条线，应优先尝试结构差异更大的模型，而不是继续在 `IR50 / IR101` 之间横跳。
- `exp_032` 已证明“只看预训练 ViT embedding”并非弱信号，因此下一条更有价值的 backbone 方向不是继续试别的 iResNet，而是做最小化工程改动的 `ViT fine-tune pilot`。
- `exp_033` 已经回答了这条线里的一个子问题：**全开 ViT 参数的轻量微调**会把边界推松，因此这不是可持续主线。
- `exp_034/035` 已说明“直接做线性 mean fusion”不是当前最优策略；若继续走异构融合，也应放到单模型 `ViT` 进一步收敛之后，而不是继续扫更多线性平均权重。
- `exp_036` 则说明：**受控的 last-block fine-tune** 和 `exp_033` 不是同一类实验；当前它至少没有破坏 `exp_032` 的核心收益，因此后续若继续推进单模型 `ViT`，应优先围绕“解冻块数”和“更保守学习率”展开，而不是回到全开微调。
- `exp_037` 已经把“feature 层是否应该参与微调”这个问题基本回答清楚：冻结 `feature/norm` 后，模型行为比 `exp_036` 更接近 `exp_032`。因此后续若继续推进单模型 `ViT`，更高优先级应是：
  - 在 `blocks-only` 设定下扫描 `last-1 / last-2 / last-3`
  - 或做和训练正交的 `TTA`
  而不是重新放开 `feature` 或继续做线性融合。
- `exp_038` 的 `KP-RPE integration audit` 已完成第一轮结论：`minchul/cvlface_adaface_vit_base_kprpe_webface12m` 不是当前代码里的真 drop-in。它至少需要：
  - keypoint-aware 前向接口（`model(input, keypoints)`）
  - 单独的 aligner 模型接线
  - `rpe_ops` 本地 C++/CUDA 扩展成功编译
- 当前机器上的 `exp_038` 运行被环境层阻塞：
  - `cl` / `g++` / `ninja` 均不存在
  - 仅有 `nvcc 11.0`
  - `rpe_ops` 编译失败，报 `CUDA 11.0` 与 `PyTorch 12.1` 不匹配
  因此这轮不能如期完成 frozen `KP-RPE WebFace12M` pilot。
- 后续只有在满足以下至少一项时，才值得占用 Kaggle 提交：
  - 本地规则没有退化成更松的接收边界；
  - 与 `exp_019` 相比不是简单地大规模 `0 -> 1/2`；
  - 方法层有明确新信息，而不是旧 scorer 的轻微变体。
- `exp_039` 已完成 `IR101 WebFace12M` 的真实 drop-in 验证，但结果给出明确负信号：
  - 相比 `exp_019` 有 `264` 张变化，其中 `263` 张是 `0 -> 1`
  - 相比 `exp_032` 有 `257` 张变化，也几乎全部是更松的接收
  - 因而“更大预训练数据源”这条轴在当前 `IR101 + prototype + fixed 0.55` 组合下，并没有自动转化成更好的 open-set 行为
  - 线上 Kaggle public score 已确认是 `0.80341`，因此这条线当前不再值得继续消耗配额深挖
- 这意味着当前优先级应进一步从“继续试 `IR101 WebFace12M` 变体”转向更正交的输入质量轴，例如：
  - 在当前 strongest baseline `exp_032` 上重新验证 `MTCNN` 对齐
  - 或其他不会天然引入大规模边界放松的输入侧改动
- `exp_040` 已完成上述第一条验证，结果是：
  - 相比 `exp_032` 仅 `1` 张 `1 -> 0`
  - 相比 `exp_019` 保持了与 `exp_032` 基本相同的整体画像
  - 没有重演早期 `exp_008` 的大规模负迁移
- 因此当前阶段结论更新为：
  - `MTCNN` 在 `frozen ViT` 主线上不是被否掉的方向
  - 但它目前也还没有提供“离线足够明显的新突破”
  - 更合理的定位是：`exp_040` 可作为近邻提交候选，与 `exp_032` 做一次线上对照
- `exp_041` 已完成 `frozen ViT + horizontal flip TTA` 的最低风险推理增强验证，结果是：
  - 相比 `exp_032` 仅 `1` 张 `0 -> 2`
  - 没有引入大规模边界放宽或塌缩
  - 因而它更适合作为后续结构性方法的 base embedding 增强层，而不是单独的冲榜主方法
- `exp_042` 已完成第一版 conservative graph refinement，结果是：
  - `base_val_accuracy` 与 `refined_val_accuracy` 同为 `0.9375`
  - `num_changed_test_predictions = 0`
  - 说明当前 gating 已经足够保守，不会污染 look-alike 邻域
  - 但也说明它暂时还没有进入“会产生有效修正”的区间
- 因此当前阶段优先级进一步更新为：
  - `exp_032` 仍是 strongest baseline
- `exp_041` 可视为当前最健康的增强版 embedding 基座
- 若继续走图方法，应优先调整 conservative graph refinement 的 gating 区间，而不是回到激进传播
- `KP-RPE` 仍然后置，直到环境链路打通
- `exp_043` 已给出新的关键信息：
  - 全局图传播确实可以动到大量高置信样本，说明“错误都卡在 boundary”这个假设不成立
  - 但第一版 `LabelSpreading-style` 传播的 `153` 张变化全部是 `0 -> 1/2`
  - 因而当前最需要解决的问题不是“图方法能不能动”，而是“如何让全局图修正产生保守方向，而不是系统性扩张”
- `exp_044` 也已经验证：
  - 简单的 look-alike margin rejection 在当前 frozen `ViT` embedding 上几乎不触发
  - 因而它暂时不构成主线突破方向
- 因此当前阶段优先级继续更新为：
  - `exp_032` 仍是 strongest baseline
- `exp_043` 是当前最有信息量的新负例：证明全局图传播有杠杆，但默认形式风险极高
- 后续若继续沿图结构推进，应优先研究**如何限制全局传播的扩张方向**，而不是继续放松局部 gating
- `exp_045` 则提供了另一条新的结构性信号：
  - `Spectral clustering + gallery label matching` 不再是纯扩张，而是出现了大规模 `2 -> 0` 的保守修正
  - 这说明“全局结构方法”并非只能把 `other` 吸进目标类，也可以反过来大幅收紧某个目标类边界
  - 但当前版本过于激进，已经超出可提交范围
- `exp_046` 已明确否掉 `PCA whitening + prototype`：
  - 在当前 frozen `ViT` 主线上，它会把大量目标类压回 `other`
  - 因而这条线当前不再继续投入
- 因此当前阶段进一步收敛为：
  - `exp_032` 仍是 strongest baseline
- `exp_045` 是最新的高信息量结构实验，但需要“去极端化”后才值得继续
- 后续若继续突破 0.92，最值得保留的方向是**更可控的全局结构方法**，而不是 whitening、look-alike margin 或继续微调 ViT
- `exp_047` 已经提供了一个比全局传播更干净的新信号：
  - `neighborhood-aware scoring = 0.5 * base_score + 0.5 * neighbor_mean_score`
  - 相比 `exp_032` 仅 `8` 张变化，且全部是 `0 -> 1/2`
  - 变化量级与 `exp_032` 当初的线上成功模式相匹配
- 因此当前阶段结论更新为：
- `exp_047` 已成为 strongest online baseline
- `exp_032` 退为最重要的纯 prototype 对照基线
- 若继续围绕 `neighbor_mean_score` 细化，必须以“只做小而准修正、不破坏 `exp_047` 当前收益”为前提

## 错误记录

| 时间 | 问题 | 处理 |
|---|---|---|
| 2026-03-21 | 直接在 PowerShell 打印 PDF 提取文本时触发 `UnicodeEncodeError` | 改用 UTF-8 文件中转或显式编码输出 |
| 2026-03-21 | `conda` 的 PowerShell alias 无法稳定执行嵌套命令 | 改用 `cmd /c` + `conda.bat activate gpu_env` |
| 2026-03-21 | Kaggle CLI 下载时报认证缺失 | 确认为本机缺少 `~/.kaggle/kaggle.json`，等待用户提供 |
