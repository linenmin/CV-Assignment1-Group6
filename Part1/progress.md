# Progress Log — CV Assignment 1

## Session: 2026-04-03

### 完成内容

**Session 1（之前）: 基线 + 增强实验**
- 新增 Section 4.2: HOG+LBP+GridSearchCV（[A-D]），最优 0.8875
- 新增 Section 4.3/4.4: 数据增强（H-flip + ±10° 旋转）+ 增强 GridSearch（[E-G]）
- [E] CV = 0.9781，提交 Kaggle → **公榜 0.80506**
- 发现问题：增强前做全量 CV → 数据泄露

**Session 2（本次）: 修复泄露 + 增强感知 CV**
- 诊断并记录泄露机制（findings.md, task_plan.md）
- 创建 pi-planning 文件（task_plan.md, findings.md, progress.md）
- 计划实现 Section 4.5：折内增强的无泄露 CV

### 当前状态
- [ ] Section 4.5 cell 已插入 notebook（待完成）
- [ ] Cell 54 更新（待完成）
- [ ] 运行验证（待完成）

## 关键指标追踪

| 日期 | 管线 | CV | Kaggle 公榜 | 备注 |
|------|------|----|-------------|------|
| 2026-04-02 | [A] HOG+PCA+SVM | 0.8875 | — | 基线 |
| 2026-04-02 | [E] HOG+PCA+SVM (aug,泄露) | 0.9781 | 0.80506 | 泄露！ |
| 2026-04-03 | [H] HOG+PCA+SVM (aug-aware) | TBD | — | 待运行 |
| 2026-04-03 | [I] HOG+LBP+PCA+SVM (aug-aware) | TBD | — | 待运行 |

## TODO（下次 Session）
- [ ] 运行 Section 4.5，记录 [H]/[I] 的诚实 CV 分数
- [ ] 提交新的 submission 到 Kaggle，验证真实分数
- [ ] 更新 findings.md 的 TBD 字段
