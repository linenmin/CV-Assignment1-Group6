# Task Plan: CV Assignment 1 — Final Submission Notebook

## Goal
将 `Part1/ga1_Group_6.ipynb` 整理成最终可提交的 Kaggle notebook。它需要同时满足三点：
- 前半部分保留课程要求的传统方法主线
- 后半部分清楚整合 DeepLearning 的最终结果
- 最终能在 Kaggle 上产出 `submission.csv`

## Current Phase
Phase 3（notebook integration）

## Phases

### Phase 1: 传统方法部分整理
- [x] 完成 Section 4.2, 4.3, 4.4, 4.5 的 classical pipeline
- [x] 记录 augmentation leakage 问题并修正为 augmentation-aware CV
- [x] 将前半部分 notebook 整理为较完整的课程基线
- **Status:** complete

### Phase 2: 分支整合与结构对齐
- [x] 将队友分支 `2_1_2_2_Build_feature_classification` 合并到当前 `DL`
- [x] 保留 `Part1/ga1_Group_6.ipynb` 作为最终 notebook 主体
- [x] 审查老师要求，确认最终评分对象是 Kaggle notebook
- **Status:** complete

### Phase 3: DeepLearning 结果写入 notebook
- [x] 新增 `4.6` 到 `4.8` 的 DL 主线叙事
- [x] 新增 `# 5. Publishing Best Results`
- [x] 修正前文中与 preprocessing 不一致的描述
- [x] 在 `0.3` 结尾补上从 baseline preprocessing 过渡到 DL preprocessing 的桥接句
- [ ] 为 `4.7` 补上旧 crop vs multi-face-selected crop 的对比图
- [ ] 继续补齐最终 Kaggle runtime 代码，使 notebook 可实际生成 submission
- **Status:** in_progress

### Phase 4: Kaggle 可运行版本
- [ ] 确定 Kaggle dataset 目录结构（权重、必要脚本、配置）
- [ ] 在 notebook 中补齐 final inference cells
- [ ] 在 Kaggle 上实际运行最终 pipeline 并导出 `submission.csv`
- [ ] 检查 notebook 是否能独立表达完整方法链
- **Status:** pending

### Phase 5: 收尾与提交
- [ ] 清理 notebook 中不必要的术语和重复解释
- [ ] 最终检查 markdown 语气和图表位置
- [ ] 更新 findings.md 和 progress.md
- [ ] 提交当前分支变更
- **Status:** pending

## Key Questions
1. 最终 notebook 中哪些代码必须可运行，哪些结果只需文档化？
   - 当前答案：最终 pipeline 必须可运行，历史实验只需清楚总结。
2. DL 主线应该怎样写才既清楚又不喧宾夺主？
   - 当前答案：以 `exp_061 -> exp_088 -> exp_109` 为主线，其他实验只作一句带过。
3. 前文 baseline preprocessing 和后文 final preprocessing 会不会冲突？
   - 当前答案：不会。前文明确是 classical baseline，后文明确是 competition-stage upgrade。

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 不把 `multiface` 预处理直接吞进 `0.2/0.3` | 避免改写前半部分 baseline，使 baseline 与 improvement 保持清楚分层 |
| `DL` 内容放在 `4.5` 之后，`6. Discussion` 之前 | 与老师的 `Improve performance` 部分对齐，也不会打断前面课程主线 |
| 只保留 `exp_061`, `exp_088`, `exp_109` 为主结果链 | 这是最短且最有说服力的故事链 |
| `exp_110` 不放主表 | 它只是负对照，放主表会分散注意力 |
| 最终 notebook 通过 Kaggle dataset 加载权重和必要脚本 | 这样 notebook 可运行，同时不需要把整个工程代码塞满正文 |

## Open Items

| Item | Next Action |
|------|-------------|
| `4.7` 的 crop 失败示例图 | 从已有可视化中选 2–4 个代表例子并插入 notebook |
| 最终 inference cells | 基于 `exp_109` 配置和 `exp_088` 权重补完整 notebook 代码 |
| Kaggle dataset 组织 | 明确需要上传的权重、配置和最小 helper code |
