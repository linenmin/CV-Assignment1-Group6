# Task Plan: CV Group Assignment 1 — Face Recognition Notebook

## Goal
完成 KUL H02A5a Computer Vision Group Assignment 1 notebook，实现多种特征提取方法（PCA、HOG 等）结合 SVM 分类器进行人脸识别，提交到 Kaggle competition 取得最佳成绩。

## Current Phase
Phase 3

## Phases

### Phase 1: 环境 & 数据加载
- [x] 修复数据路径（Kaggle → 本地，支持任意 CWD）
- [x] 修复 HAARPreprocessor（`COLOR_BGR2GRAY` → `COLOR_RGB2GRAY`，使用 cv2 内置 xml）
- [x] 验证数据加载：80 训练样本 / 1816 测试样本
- **Status:** complete

### Phase 2: 特征提取器实现
- [x] Section 1.0：`IdentityFeatureExtractor`（模板已有）
- [x] Section 1.2：`PCAFeatureExtractor`（fit/transform/inverse_transform，n=50）
- [x] Section 1.2.1：特征脸可视化（Top-16 eigenfaces）
- [x] Section 1.2.1：累积解释方差曲线
- [x] Section 1.2.1：不同 k 值重建质量对比
- [x] Section 1.2.2：t-SNE 特征空间可视化
- [x] Section 1.2.3：讨论（设计决策、方差解释、分离性）
- [ ] Section 1.1：`HOGFeatureExtractor`（当前为 stub，待实现）
- [ ] Section 1.1.1：HOG 特征的 t-SNE 可视化
- [ ] Section 1.1.2：HOG 讨论
- **Status:** in_progress

### Phase 3: 分类器 & 实验
- [x] Section 3.0：`RandomClassificationModel`（模板已有）
- [x] Section 3.1：`SVMClassifier`（RBF kernel，C=10）
- [x] Section 4.0：Identity + Random 基线实验
- [x] Section 4.1：PCA(n=50) + SVM(RBF) — 包含 5-fold CV
- [ ] Section 4.2：HOG + SVM 实验
- [ ] Section 4.3：超参数调优（GridSearchCV：不同 n_components、C、gamma）
- [ ] Section 4.4：（可选）PCA + HOG 融合特征
- **Status:** in_progress

### Phase 4: 评估 & 验证
- [ ] 确认所有实验有 5-fold CV accuracy（防止 overfitting）
- [ ] 比较所有 pipeline 性能（表格形式）
- [ ] 选择最佳模型生成最终提交文件
- **Status:** pending

### Phase 5: 撰写讨论 & 交付
- [ ] Section 6：完整 Discussion（方法对比、局限性、改进方向）
- [ ] 确认 notebook 可从头到尾完整运行（Restart & Run All）
- [ ] 提交 submission_*.csv 到 Kaggle
- **Status:** pending

## Key Questions
1. HOG 的最优参数是什么？（orientations, pixels_per_cell, cells_per_block）
2. PCA n_components 的最优值是多少？（当前 n=50，是否需要调优）
3. HOG 特征与 PCA 特征哪个对这个数据集更有效？
4. 融合 PCA + HOG 特征是否能提升性能？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| PCA n_components=50 | 平衡重建精度与小数据集（80样本）过拟合风险 |
| SVM RBF kernel, C=10, gamma='scale' | 非线性边界，scale自适应高维特征 |
| 5-fold StratifiedKFold CV | 数据量少，分层保证各折类别比例 |
| Pipeline(PCA+scaler+SVM) for CV | 确保每折内独立 fit PCA，防止数据泄露 |
| HAARPreprocessor 中心裁剪兜底 | 避免未检测到人脸时产生 NaN 行破坏 PCA/SVM |
| 修复 COLOR_RGB2GRAY bug | 图像已转换为 RGB，原模板用 BGR→GRAY 导致颜色通道错误 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| FileNotFoundError: Kaixi_Work/kul-... | 1 | `os.path.abspath('__file__')` 在 Jupyter 中返回 CWD 而非文件路径 |
| FileNotFoundError: Kaixi_Work/kul-... | 2 | 改用循环检查 `.`, `..`, `../..` 找到数据目录 |

## Notes
- 当前 notebook 路径：`Kaixi_Work/Kaixi_work.ipynb`
- 数据路径：`../kul-computer-vision-ga-1-2026/`（相对于 Kaixi_Work/）
- 训练集：80 样本（class 0: look-alike 20, class 1: Jesse 30, class 2: Mila 30）
- 测试集：1816 样本（无标签）
- 图像尺寸：原始约 400×300，裁剪后 100×100
