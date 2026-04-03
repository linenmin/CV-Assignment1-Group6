# Findings & Decisions — CV Assignment 1

## Requirements
- 3 类人脸识别（Look-alike=0, Jesse=1, Mila=2），80 训练 / 1816 测试
- 目标：最大化 Kaggle 公榜准确率
- 使用传统特征（HOG、LBP）+ 经典分类器（SVM）

## 实验结果汇总

### 基线实验（无数据增强，80 原始样本）

| 管线 | CV 准确率 | 最优参数 | 诚实？ |
|------|-----------|----------|--------|
| [A] HOG+PCA+SVM | 0.8875 | C=10, γ=auto, n_components=30 | ✓ |
| [C] HOG+LBP+PCA+SVM | 0.8875 | C=10, γ=scale, n_components=50 | ✓ |
| [B] LBP+SVM | 0.7500 | C=10, γ=0.001 | ✓ |
| [D] Pixel PCA+SVM | 0.7375 | C=1, γ=scale, n_components=50 | ✓ |

### 数据增强实验（Section 4.3/4.4 — 存在数据泄露！）

| 管线 | CV 准确率 | Kaggle 公榜 | 泄露说明 |
|------|-----------|-------------|----------|
| [E] HOG+PCA+SVM (aug) | **0.9781** | **0.80506** | ← 先增强再 CV → 泄露！ |
| [F] HOG+LBP+PCA+SVM (aug) | 0.9750 | 未测试 | 同样泄露 |
| [G] LBP+SVM (aug) | 0.8219 | 未测试 | 同样泄露 |

**泄露机制：** 80 张图先增强为 320 张，再做 StratifiedKFold(5)。
val 折中含有某图的增强版本（H-flip 或 ±10° 旋转），而该图原图在 train 折中。
SVM 见过"几乎相同"的图 → CV 虚高 ~17 个百分点。

### 增强感知 CV 实验（Section 4.5 — 进行中）

| 管线 | CV 准确率 | 泄露？ | 状态 |
|------|-----------|--------|------|
| [H] HOG+PCA+SVM（折内增强） | TBD | ✗ | 待运行 |
| [I] HOG+LBP+PCA+SVM（折内增强） | TBD | ✗ | 待运行 |

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 每折内增强（而非全量先增强） | 避免 val 折看到 train 折的增强版本 |
| ParameterGrid 手动 CV 循环 | GridSearchCV 无法将增强逻辑嵌入折内 |
| 验证集 = 原始图（不增强） | 反映真实部署场景——预测时不做增强 |
| 最终模型用全量 80+aug 训练 | 超参已诚实选定，全量训练提升模型 |

## Resources

- Notebook: `Kaixi_Work/Kaixi_Work.ipynb`
- 数据目录: 见 cell 4（自动检测路径）
- Python 环境: `/c/Users/31667/.conda/envs/biometrics/python.exe`
- Submission: `Kaixi_Work/submission_improved.csv`

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| Section 4.4 增强全量再 CV → 泄露，Kaggle 0.805 | 重新实现 Section 4.5 折内增强 |
| nbformat 4.4 cells 无 id 字段 | 直接用 Python 修改 notebook JSON 插入 cell |
