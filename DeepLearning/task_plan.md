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
| Phase 0 | in_progress | 读取 AGENTS、PDF、仓库现状，确认约束 |
| Phase 1 | pending | 明确 2.3 的任务边界、输入输出、评价指标与提交格式 |
| Phase 2 | completed | 设计深度学习子项目的协作流程、目录结构与迭代节奏 |
| Phase 3 | completed | 与用户确认设计方案 |
| Phase 4 | in_progress | 创建分支 `DL`、初始化项目骨架与 `.gitignore` |
| Phase 5 | pending | 落地第一版可运行基线并建立实验记录规范 |

## 决策记录

- 计划文件放在 `DeepLearning` 子目录中，和后续代码同级，便于该子项目独立协作。
- 2.3 暂按“独立深度学习增强线”规划：与 2.1/2.2 主要是结果对比关系，而不是代码共建关系。
- 工程开发采用 `script-first`，最终 notebook 暂不纳入当前起步阶段。
- Kaggle 迭代阶段优先采用 `csv` 提交，等方案收敛后再回填 notebook。
- 协作方式采用共享 pipeline、轮流迭代，不按子模块永久分工。

## 错误记录

| 时间 | 问题 | 处理 |
|---|---|---|
| 2026-03-21 | 直接在 PowerShell 打印 PDF 提取文本时触发 `UnicodeEncodeError` | 改用 UTF-8 文件中转或显式编码输出 |
