# Progress Log — CV GA1 Face Recognition

## Session: 2026-03-26

### Phase 1: 环境 & 数据加载
- **Status:** complete
- **Started:** 2026-03-26
- Actions taken:
  - 修复数据路径从 Kaggle 路径 → 本地 `kul-computer-vision-ga-1-2026/`
  - 修复 `FileNotFoundError`：`os.path.abspath('__file__')` 在 Jupyter 中行为与脚本不同
  - 最终方案：循环检查 `.`, `..`, `../..` 找到数据目录
- Files created/modified:
  - `Kaixi_Work/Kaixi_work.ipynb` cell-4（数据加载路径）

### Phase 2: 特征提取器实现
- **Status:** in_progress（PCA 已完成，HOG 待实现）
- **Started:** 2026-03-26
- Actions taken:
  - 修复 `HAARPreprocessor`（cv2 内置 xml，RGB/BGR bug，中心裁剪兜底）
  - 完整实现 `PCAFeatureExtractor`（fit/transform/inverse_transform）
  - 添加可视化：Top-16 eigenfaces，累积解释方差，重建质量对比
  - 添加 t-SNE 特征空间可视化
  - 添加 Section 1.2.3 讨论
  - 添加 `SVMClassifier` 包装类
  - 修复 cell-45/46 类型错误（markdown → 正确的 markdown/code）
- Files created/modified:
  - `Kaixi_Work/Kaixi_work.ipynb`（cells 0, 4, 10, 12, 21, 27-33, 38-39, 46-47）

### Phase 3: 分类器 & 实验
- **Status:** in_progress（PCA+SVM 已完成，HOG 实验待实现）
- Actions taken:
  - 实现 5-fold CV（Pipeline 内 PCA+Scaler+SVM，防止数据泄露）
  - 更新提交文件为 PCA+SVM 预测结果
- Files created/modified:
  - `Kaixi_Work/Kaixi_work.ipynb` cell-46（PCA+SVM 实验）
  - `Kaixi_Work/submission_pca_svm.csv`（待生成）

## 下一步工作（优先顺序）

1. **实现 HOG 特征提取器**（Section 1.1）
   - 使用 `skimage.feature.hog`
   - 参数：orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)
   - 添加 t-SNE 可视化（1.1.1）
   - 添加讨论（1.1.2）

2. **HOG + SVM 实验**（Section 4.2）
   - 5-fold CV
   - 与 PCA+SVM 比较

3. **超参数调优**（Section 4.3）
   - GridSearchCV：n_components ∈ {20,50,75}, C ∈ {0.1,1,10,100}
   - 找到最优参数组合

4. **撰写 Discussion**（Section 6）

5. **运行完整 notebook 并生成最终 submission**

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 数据加载 | cell-4 运行 | 无 FileNotFoundError | 待验证（修复后） | 待测 |
| PCA 特征提取 | train_X(80,100,100,3) | train_features(80,50) | (80,50) | ✓ |
| 5-fold CV pipeline | flat_train(80,30000) | CV score > 0.5 | 待运行 | 待测 |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-26 | FileNotFoundError: Kaixi_Work/kul-.../train_set.csv | 1 | 分析：os.path.abspath('__file__') 在 Jupyter 返回 CWD |
| 2026-03-26 | FileNotFoundError: Kaixi_Work/kul-.../train_set.csv | 2 | 修复：循环检查 '.', '..', '../..' |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 3 — 分类器 & 实验 |
| Where am I going? | 实现 HOG + 超参数调优 + Discussion |
| What's the goal? | 完成 notebook，提交 Kaggle 取得最佳成绩 |
| What have I learned? | 见 findings.md |
| What have I done? | PCA+SVM 完整实现，数据路径修复，可视化完成 |

---
*Update after completing each phase or encountering errors*
