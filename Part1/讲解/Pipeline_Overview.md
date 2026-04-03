# 完整流程：从数据加载到最终分类

## 问题定义

3 分类人脸识别任务：

| 类别 | 含义 |
|------|------|
| 0 | Look-alike（长相相似的人，负样本） |
| 1 | Jesse |
| 2 | Mila |

核心挑战：**80 张训练图 vs 1816 张测试图**，极度的 few-shot 场景。

---

## 数据流图

```
原始图像（任意尺寸，RGB）
    │
    ▼ HAARPreprocessor
裁剪人脸 → resize (100, 100, 3)
    │
    ├──────────────────────────────────────┐
    │  质量过滤（blur / face / eye 检查）    │
    │  → train_X_clean (≤80, 100, 100, 3)  │
    └──────────────────────────────────────┘
    │
    ▼ 特征提取（三选一或组合）
    ├─ PCA        → (N, 50)
    ├─ HOG        → (N, 1764)
    ├─ LBP        → (N, 256)
    └─ HOG + LBP  → (N, 2020)
    │
    ▼ StandardScaler 标准化
    │
    ▼ SVM (RBF kernel) 分类
    │
    ▼ GridSearchCV + 5-fold CV 选最优超参
    │
    ▼ 无泄漏增强 CV 验证最终模型
    │
    ▼ 预测 test_X → submission_improved.csv
         类别 0 / 1 / 2
```

---

## 第 1 步：数据加载（`data_loader.py`）

```python
train, test = load_data()
```

- 自动在 `.` / `..` / `../..` 中搜索数据目录
- 读取 `train_set.csv` 和 `test_set.csv`
- 用 `np.load()` 读取每张图的 `.npy` 文件
- `cv2.COLOR_BGR2RGB` 转色彩空间（OpenCV 默认 BGR，需转成 RGB）
- 结果：`train` DataFrame（含 name、class、img 列），`test` DataFrame（含 img 列）

---

## 第 2 步：人脸预处理（`HAARPreprocessor`）

```python
preprocessor = HAARPreprocessor(face_size=(100, 100))
train_X = preprocessor(train)   # shape: (80,  100, 100, 3)
test_X  = preprocessor(test)    # shape: (1816, 100, 100, 3)
```

对每张图的操作：

```
原始图（任意尺寸，RGB）
    ↓ 转灰度（COLOR_RGB2GRAY）
    ↓ HAAR Cascade 检测人脸框
    ↓
  ┌─ 检测到人脸 → 裁剪人脸区域 [y:y+h, x:x+w]
  └─ 未检测到   → 中心正方形裁剪（fallback，保证不丢样本）
    ↓
  cv2.resize → (100, 100)
```

**关键设计**：fallback 策略确保每张图都有输出，避免下游 PCA/SVM 遇到 NaN。

---

## 第 3 步：质量过滤（`quality_check()`）

```python
quality_df = quality_check(train, preprocessor, blur_threshold=80.0)
```

对每张训练图打 3 个质检分：

| 检查项 | 方法 | 标记条件 |
|--------|------|---------|
| 人脸检测 | HAAR cascade | 未检测到人脸 |
| 清晰度 | Laplacian 方差 | 方差 < 80 |
| 眼睛遮挡 | Eye cascade | 未检测到眼睛 |

手动填 `BAD_INDICES = [...]` 剔除坏样本，得到：
- `train_X_clean`（形状 ≤ (80, 100, 100, 3)）
- `train_y_clean`

---

## 第 4 步：特征提取（3 种方案）

### 方案 A：PCA（Eigenfaces）

```
PCAFeatureExtractor.fit(train_X_clean)   ← 只在训练集上 fit！
    ↓
(80, 100, 100, 3) → 展平 → (80, 30000) → PCA → (80, 50)
```

- 将 30,000 维像素空间压缩到 50 维主成分
- 每个主成分是一张 Eigenface（特征脸），捕捉人脸最大方差方向
- n_components=50 约能解释 ~95% 方差

### 方案 B：HOG（梯度方向直方图）

```
extract_hog(X):  (N, 100, 100, 3)
    ↓ 转灰度 + 直方图均衡化
    ↓ resize 到 (64, 64)
    ↓ HOGDescriptor(winSize=64, blockSize=16, blockStride=8, cellSize=8, nbins=9)
    → (N, 1764)
```

- 把图像分成 8×8 的 cell，每个 cell 计算 9 方向的梯度直方图
- 对光照鲁棒，捕捉边缘/纹理结构

### 方案 C：LBP（局部二值模式）

```
extract_lbp(X):  (N, 100, 100, 3)
    ↓ 转灰度 + 直方图均衡化
    ↓ 每个像素与8邻域比较 → 8位二进制码（0-255）
    ↓ 统计全图 256 bins 直方图，归一化
    → (N, 256)
```

- 编码局部微纹理模式，对人脸识别非常有效

### 方案 D：HOG + LBP 拼接

```
extract_combined(X) = hstack([extract_hog(X), extract_lbp(X)])
                    → (N, 1764 + 256) = (N, 2020)
```

---

## 第 5 步：数据增强（`augment_dataset()`）

训练集只有 80 张，严重不足。增强方法：

```
原始 80 张
  + 水平翻转        (80 张)
  + 旋转 +10°      (80 张)
  + 旋转 -10°      (80 张)
= 320 张
```

**重要**：增强只应用于训练集，测试集和 CV 验证集绝不增强。

---

## 第 6 步：分类器（SVM with RBF kernel）

```python
SVMClassifier(C=10, gamma='scale')
```

- **3 分类策略**：one-vs-one，构造 3 个二分类器（0 vs 1、0 vs 2、1 vs 2），投票决定最终类别
- **RBF 核**：`K(x,y) = exp(-γ·||x-y||²)`，处理非线性边界
- **C**：惩罚系数，控制过拟合 vs 欠拟合
- **gamma**：控制核函数宽度，影响决策边界复杂度

---

## 第 7 步：超参数搜索（GridSearchCV）

对每种特征组合做 **GridSearchCV + 5-fold 分层交叉验证**：

```python
# 搜索空间示例（Pipeline A: HOG → PCA → Scaler → SVM）
{
    'pca__n_components': [30, 50, 80, 100],
    'svm__C':            [0.1, 1, 10, 100, 500],
    'svm__gamma':        ['scale', 'auto', 1e-3, 1e-2],
}
```

参与对比的 Pipeline：

| 名称 | 特征 | 说明 |
|------|------|------|
| [A] | HOG → PCA → Scaler → SVM | 原始 80 张 |
| [B] | LBP → Scaler → SVM | 原始 80 张 |
| [C] | HOG+LBP → PCA → Scaler → SVM | 原始 80 张 |
| [D] | 像素 → PCA → Scaler → SVM | 原始 80 张 |
| [E] | HOG → PCA → Scaler → SVM | 增强 320 张（有泄漏） |
| [F] | HOG+LBP → PCA → Scaler → SVM | 增强 320 张（有泄漏） |
| [G] | LBP → Scaler → SVM | 增强 320 张（有泄漏） |

---

## 第 8 步：无泄漏增强 CV（`run_aug_aware_cv()`）

直接在全量增强数据上做 CV 会导致数据泄漏：val fold 可能包含 img_i 的增强版，而 train fold 里有 img_i 本身，CV 分数虚高（曾出现 CV=0.978 但 Kaggle 实际仅 0.805）。

**正确做法（in-fold augmentation）**：

```
for 每一个 CV fold:
    train fold（~64 张原图）
        ↓ augment_dataset → ~256 张
        ↓ 提取特征
        ↓ pipeline.fit()

    val fold（~16 张原图，绝不增强）
        ↓ 提取特征
        ↓ pipeline.predict()
        ↓ accuracy_score
```

得到两个诚实模型：
- `[H] HOG+PCA+SVM`（aug-aware）
- `[I] HOG+LBP+PCA+SVM`（aug-aware）

---

## 第 9 步：模型选择与最终预测

```python
LEAKY = {'[E]', '[F]', '[G]'}   # 排除有泄漏的模型

best_name = select_best_honest_model(_candidates, LEAKY)
# → 在 [A][B][C][D][H/I] 中选 CV 最高的
```

最终训练：用全部 80 张 + 增强（320 张）重新训练最佳模型，预测测试集：

```python
test_pred = best_gs.best_estimator_.predict(get_test_feat())
submission = test.drop('img', axis=1).copy()
submission['class'] = test_pred
submission.to_csv('submission_improved.csv')
```

---

## 各步骤对应文件

| 步骤 | 文件 | 关键函数/类 |
|------|------|------------|
| 数据加载 | `data_loader.py` | `load_data()` |
| 人脸预处理 | `preprocessors.py` | `HAARPreprocessor` |
| 质量过滤 | `preprocessors.py` | `quality_check()` |
| 特征提取 | `feature_extractors.py` | `PCAFeatureExtractor`, `extract_hog/lbp/combined` |
| 分类器 | `classifiers.py` | `SVMClassifier` |
| 数据增强 | `augmentation.py` | `augment_dataset()` |
| 无泄漏 CV | `augmentation.py` | `run_aug_aware_cv()` |
| 实验管理 | `experiments.py` | `run_grid_search()`, `HonestResult`, `select_best_honest_model()` |
| 可视化 | `visualization.py` | `plot_eigenfaces()`, `plot_tsne()` |
| 主流程 | `ga1_Group_6.ipynb` | 按 Cell 顺序执行 |
