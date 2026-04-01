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

## 决策记录

- 计划文件放在 `DeepLearning` 子目录中，和后续代码同级，便于该子项目独立协作。
- 2.3 暂按“独立深度学习增强线”规划：与 2.1/2.2 主要是结果对比关系，而不是代码共建关系。
- 工程开发采用 `script-first`，最终 notebook 暂不纳入当前起步阶段。
- Kaggle 迭代阶段优先采用 `csv` 提交，等方案收敛后再回填 notebook。
- 协作方式采用共享 pipeline、轮流迭代，不按子模块永久分工。
- 当前最强线上结果是 `exp_019 = 0.91685`，对应协议为：`exp_009 checkpoint + prototype + fixed threshold 0.55`。
- `threshold = 0.55` 视为当前锁定的评测协议；在模型探索阶段不再继续自由搜索阈值。
- `exp_023 ~ exp_027` 已验证多种更复杂的开放集推理思路，但当前实现大多会把过多 `other` 放宽为目标类，因此暂不消耗 Kaggle 配额继续提交。
- 当前阶段的实验优先级应从“小幅边界调参”转为“只有明显不同的方法层变化才值得提交”，例如 score/model fusion 或训练目标对齐。

## 当前阶段目标

- 保持 `exp_019` 作为当前 strongest baseline，不轻易用小改动消耗 Kaggle 日提交次数。
- 所有新推理实验先做本地离线对比：重点看与 `exp_019` 的预测差异规模和方向，先过滤掉明显“过度接收 other”的方案。
- 后续只有在满足以下至少一项时，才值得占用 Kaggle 提交：
  - 本地规则没有退化成更松的接收边界；
  - 与 `exp_019` 相比不是简单地大规模 `0 -> 1/2`；
  - 方法层有明确新信息，而不是旧 scorer 的轻微变体。

## 错误记录

| 时间 | 问题 | 处理 |
|---|---|---|
| 2026-03-21 | 直接在 PowerShell 打印 PDF 提取文本时触发 `UnicodeEncodeError` | 改用 UTF-8 文件中转或显式编码输出 |
| 2026-03-21 | `conda` 的 PowerShell alias 无法稳定执行嵌套命令 | 改用 `cmd /c` + `conda.bat activate gpu_env` |
| 2026-03-21 | Kaggle CLI 下载时报认证缺失 | 确认为本机缺少 `~/.kaggle/kaggle.json`，等待用户提供 |
