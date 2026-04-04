# Task Plan: CV Assignment 1 — 增强感知 CV 改进

## Goal
实现无数据泄露的增强感知交叉验证（Section 4.5），修复 Section 4.4 的泄露问题，提交诚实可靠的预测结果。

## Current Phase
Phase 2（实现中）

## Phases

### Phase 1: 问题诊断与记录
- [x] 分析 Section 4.4 的数据泄露根因
- [x] 确认 Kaggle 公榜 vs CV 分数的差距（0.9781 CV vs 0.80506 公榜）
- [x] 记录历史实验结果到 findings.md
- **Status:** complete

### Phase 2: 实现增强感知 CV（Section 4.5）
- [x] 创建 pi-planning 文件
- [ ] 在 notebook cell 52 后插入 Section 4.5 代码
- [ ] 更新 Cell 54 的最优模型选择（加入 [H]/[I]）
- [ ] 运行 notebook 验证结果
- **Status:** in_progress

### Phase 3: 验证与提交
- [ ] 确认 [H]/[I] 诚实 CV 分数 < [E]/[F] 但 > [A]/[C]
- [ ] 生成新的 submission_improved.csv
- [ ] 提交 Kaggle 验证真实分数
- [ ] 更新 findings.md 和 memory
- **Status:** pending

### Phase 4: 文档整理
- [ ] 在 progress.md 记录最终结果
- [ ] 更新 memory/project_cv_assignment.md
- **Status:** pending

## Key Questions
1. ✅ 为什么 CV 0.9781 但公榜 0.80506？→ 数据泄露（增强前做了全量增强再 CV）
2. 增强感知 CV 的诚实分数预期是多少？→ 预计 0.88–0.93

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 增强感知 CV 而非简单 hold-out | 80 样本太少，5-fold CV 比 hold-out 估计方差更小 |
| ParameterGrid 手动搜索（非 GridSearchCV） | GridSearchCV 无法在折内做增强，需要自定义循环 |
| 最终模型用全部 80+aug 训练 | 超参数已通过诚实 CV 选定，用全量数据训练提升模型质量 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Section 4.4 数据泄露 | 1 | 重新设计为折内增强（Section 4.5） |
| Kaggle 公榜 0.80506 << CV 0.9781 | 1 | 确认为泄露导致，用增强感知 CV 修复 |
