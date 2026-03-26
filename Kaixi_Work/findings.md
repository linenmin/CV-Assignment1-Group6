# Findings & Decisions — CV GA1 Face Recognition

## Requirements
- 实现多种特征提取方法，至少包含 PCA 和一种其他方法
- 使用 Jupyter notebook（`Kaixi_Work/Kaixi_work.ipynb`）
- 3 类分类：class 0（look-alike）, class 1（Jesse Eisenberg）, class 2（Mila Kunis）
- 提交 CSV 到 Kaggle competition（test set 预测）
- Notebook 需自包含、充分文档化，并说明设计决策

## Dataset
- **训练集**：80 张图（class 0: 20张, class 1: 30张, class 2: 30张）
- **测试集**：1816 张图（无标签）
- **图像格式**：`.npy` 文件，BGR 存储，需转 RGB 显示
- **原始尺寸**：约 400×300×3
- **预处理后**：100×100×3（经 HAAR 人脸检测裁剪 + INTER_AREA 缩放）
- **数据集来源**：VGG Face Dataset 子集

## 已完成的工作（截至 2026-03-26）

### 数据加载与预处理
- 数据路径使用循环检查 `.`, `..`, `../..`，适配任意 CWD
- `HAARPreprocessor` 修复两个 bug：
  1. 使用 `cv2.data.haarcascades` 内置路径（无需下载）
  2. `cv2.COLOR_RGB2GRAY` 替代 `cv2.COLOR_BGR2GRAY`（图像已是 RGB）
  3. 未检测到人脸时中心裁剪兜底（避免 NaN）

### PCA 特征提取器（Section 1.2）
- `PCAFeatureExtractor(n_components=50)`
  - `_flatten`: `(N, H, W, C) → (N, H*W*C)` float64
  - `fit(train_X)`: 仅在训练集调用
  - `transform(X)`: SVD 投影 → (N, 50)
  - `inverse_transform(X_pca)`: 重建 → (N, H, W, C)
- 可视化：Top-16 eigenfaces，累积解释方差，重建质量对比
- t-SNE：50维→2D，perplexity=15（适合~80个点）

### SVM 分类器（Section 3.1）
- `SVMClassifier(C=10, gamma='scale', kernel='rbf')`
- 包装 sklearn `SVC`

### PCA + SVM 实验（Section 4.1）
- 训练准确率：通常接近 100%（模型容量足够 80 样本）
- 5-fold CV：使用 `Pipeline(PCA+StandardScaler+SVM)`，每折独立 fit PCA

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| `float64` for flattening | SVD 数值敏感，高精度避免舍入误差 |
| `StandardScaler` in CV pipeline | 归一化 PCA 投影坐标，改善 SVM 收敛 |
| `random_state=42` | 可复现性 |
| `StratifiedKFold(n_splits=5)` | 小数据集，分层 K 折确保各折类别比例 |
| perplexity=15 for t-SNE | 经验值：约为样本量的 1/5 |

## 待实现内容

### Section 1.1：HOG 特征提取器
```python
# 建议实现方案
from skimage.feature import hog

class HOGFeatureExtractor(IdentityFeatureExtractor):
    def __init__(self, orientations=9, pixels_per_cell=(8,8),
                 cells_per_block=(2,2), channel_axis=-1):
        ...
    def transform(self, X):
        # X: (N, H, W, C) → hog features → (N, feat_dim)
        ...
```
- 典型参数：orientations=9, ppc=(8,8), cpb=(2,2)
- 输入 100×100×3：feat_dim = 9 × (100/8-1) × (100/8-1) × 4 ≈ 4356 维
- 需要 `pip install scikit-image`

### 超参数调优（待做）
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'pca__n_components': [20, 50, 75],
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto'],
}
```

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `os.path.abspath('__file__')` 在 Jupyter 中返回 CWD+字符串 | 改用循环检查多个候选路径 |
| `ga1_group_X.ipynb` 消失 | 已在 `Kaixi_Work/Kaixi_work.ipynb` 中维护所有修改 |

## Resources
- 数据目录：`../kul-computer-vision-ga-1-2026/` (相对于 Kaixi_Work/)
- Notebook：`Kaixi_Work/Kaixi_work.ipynb`
- PCA 参考实现：`Kaixi_Work/PCA.py`
- scikit-image HOG 文档：`skimage.feature.hog`
- sklearn Pipeline：防止 CV 中数据泄露的关键

*Update this file after every 2 view/browser/search operations*
