# PCA + SVM 管线详解

## 总体结构

```
图像 (N, 100, 100, 3)
    ↓ 展平
像素向量 (N, 30000)
    ↓ PCA
低维特征 (N, 50)
    ↓ StandardScaler
标准化特征 (N, 50)
    ↓ SVM (RBF kernel)
预测标签 (N,)  →  0 / 1 / 2
```

---

## 第一阶段：PCA（降维）

### 为什么需要 PCA？

100×100×3 的图像展平后是 **30,000 维**向量。而训练集只有 80 张图，这是典型的「维度灾难」：维度 >> 样本数，SVM 会严重过拟合。

### 数学原理

PCA 的目标是找到一组正交方向（主成分），使数据在这些方向上的方差最大，从而用最少的维度保留最多的信息。

**fit 阶段**（只在训练集上做）：

```python
# feature_extractors.py: PCAFeatureExtractor.fit()
def _flatten(self, X):
    return X.reshape(X.shape[0], -1).astype(np.float64)
    # (80, 100, 100, 3) → (80, 30000)

def fit(self, X):
    self.image_shape = X.shape[1:]  # 记住 (100, 100, 3)，用于重建
    self.pca.fit(self._flatten(X))
```

sklearn 的 `PCA.fit()` 内部做了什么：

1. 计算数据均值 $\mu$（形状 30000），从每张图减去均值
2. 计算协方差矩阵（或用 SVD 等价计算）
3. 对协方差矩阵做特征分解，得到特征向量（Eigenfaces）和特征值
4. 按特征值从大到小排列，取前 50 个特征向量

这 50 个特征向量就是 **Eigenfaces**——每个都是一张抽象的「特征脸」，代表训练集人脸的一个主要变化模式。

> **Technical Notes**
>
> 1. **2D Matrix Conversion**: We converted the 4D image tensor into a 2D matrix by flattening each 100×100×3 image into a 1D vector of length 30,000. Stacking the N=80 training samples results in a 80×30,000 data matrix.
>
> 2. **SVD vs. Eigenvalue Decomposition**: A standard eigenvalue decomposition would require computing a 30,000×30,000 covariance matrix, which is computationally prohibitive. Because N << D, sklearn uses SVD, which implicitly operates on an 80×80 matrix.
>
> 3. **Mean Subtraction**: Strictly required and applied internally during `fit()`. Without it, the first principal component would point toward the dataset mean rather than capturing maximum variance directions.
>
> 4. **Pre-processing Steps**: Before PCA, three critical steps were required:
>    - (a) HAAR Cascade face extraction to remove background noise
>    - (b) Centre-crop fallback for failed detections to avoid NaN values
>    - (c) Resizing all images to 100×100 so every flattened vector is exactly 30,000-dimensional

**transform 阶段**（训练集和测试集都用同一个 fit 的基底）：

```python
def transform(self, X):
    return self.pca.transform(self._flatten(X))
    # (N, 30000) → (N, 50)
```

本质是把每张图投影到这 50 个 Eigenface 上，得到 50 个坐标值：

$$\text{feature}_i = [x - \mu] \cdot V^T$$

其中 $V$ 是 (50, 30000) 的特征向量矩阵，$x$ 是展平后的图像。

> **关键约束**：`fit` 只在 `train_X_clean` 上调用一次，测试集用同样的均值和特征向量进行 `transform`，不能重新 fit——否则就是数据泄漏。

---

## 第二阶段：StandardScaler（标准化）

```python
# experiments.py: make_pca_svm_pipeline()
Pipeline([
    ('pca',    skPCA()),
    ('scaler', StandardScaler()),   # ← 这一步
    ('svm',    SVC(kernel='rbf', random_state=42)),
])
```

PCA 输出的 50 个坐标，不同主成分的量纲差异巨大——第 1 主成分（方差最大）的值可能是第 50 主成分的几十倍。SVM 的 RBF 核计算的是欧氏距离，对尺度非常敏感，必须先标准化。

StandardScaler 对每个维度 $j$：

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

其中 $\mu_j$ 和 $\sigma_j$ 也只从训练集计算。放在 sklearn `Pipeline` 里，这个过程是自动保证的。

---

## 第三阶段：SVM（分类）

### SVM 的基本思想

SVM 在特征空间中寻找能最大化类别间隔的超平面。对于 2 分类，超平面是 $w \cdot x + b = 0$，间隔是 $\frac{2}{\|w\|}$，目标是最大化它。

本任务是 **3 分类**，sklearn 的 SVC 用 **one-vs-one** 策略：构造 $\binom{3}{2}=3$ 个二分类器（0 vs 1、0 vs 2、1 vs 2），每个样本投票，得票多的类别获胜。

### RBF 核

线性 SVM 在 50 维 PCA 空间中可能找不到好的分割，RBF 核把数据隐式映射到无穷维空间：

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

| $\gamma$ | 核函数宽度 | 决策边界 | 风险 |
|----------|-----------|---------|------|
| 小 | 宽，每个支持向量影响范围大 | 平滑 | 欠拟合 |
| 大 | 窄，每个支持向量影响范围小 | 复杂 | 过拟合 |

```python
# classifiers.py
class SVMClassifier:
    def __init__(self, C=10, gamma='scale', kernel='rbf'):
        self.model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)
```

`gamma='scale'` 是 sklearn 的自动设置：$\gamma = \frac{1}{n\_\text{features} \times \text{Var}(X)}$，根据数据自适应调整。

### 参数 C 的作用

C 是软间隔惩罚系数，控制允许多少训练样本落在错误一侧：

| C 值 | 效果 |
|------|------|
| 小 | 允许更多误分类，决策边界更平滑，泛化更好 |
| 大 | 强迫正确分类所有训练样本，可能过拟合 |

---

## GridSearchCV：自动找最优超参

```python
# experiments.py: run_grid_search()
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, refit=True)
gs.fit(features, labels)
```

搜索空间（共 4×5×4 = **80 种组合**）：

```python
{
    'pca__n_components': [30, 50, 80, 100],
    'svm__C':            [0.1, 1, 10, 100, 500],
    'svm__gamma':        ['scale', 'auto', 1e-3, 1e-2],
}
```

每种组合用 5-fold CV 评估，等于训练 400 个模型，自动选最高 CV 准确率的组合。

**StratifiedKFold** 的作用：普通 KFold 随机分割可能导致某折 val 里全是一个类，StratifiedKFold 保证每折的类别比例与整体一致——对 80 张这种小数据集尤其重要。

---

## 5-fold CV 的工作方式

```
80 张 train_X_clean 按类别分层打乱

Fold 1: [ 1-16 张] val  |  [17-80 张] train → fit PCA+Scaler+SVM → accuracy
Fold 2: [17-32 张] val  |  其余 64 张 train  → fit PCA+Scaler+SVM → accuracy
Fold 3: ...
Fold 4: ...
Fold 5: ...
                                               ──────────────────────────────
                                               mean(5 scores) = CV accuracy
```

> Pipeline 保证每个 fold 的 PCA 和 Scaler 都只在该 fold 的训练部分上 fit，val 部分用训练部分的参数做 transform——这才是无泄漏的正确评估。

---

## 数字总结

| 阶段 | 输入维度 | 输出维度 | 说明 |
|------|---------|---------|------|
| 展平 | (80, 100, 100, 3) | (80, 30000) | 像素向量 |
| PCA | (80, 30000) | (80, 50) | 压缩 600 倍 |
| StandardScaler | (80, 50) | (80, 50) | 均值 0 方差 1 |
| SVM fit | (80, 50) + 标签 | 3 个二分类器 | one-vs-one |
| SVM predict | (1816, 50) | (1816,) | 0 / 1 / 2 |
