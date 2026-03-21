"""
PCA-based Face Recognition Pipeline  (KUL H02A5a Computer Vision — Group Assignment 1)
========================================================================================

Pipeline overview  /  流程概览:
  Raw Images  →  HAAR Face Detection  →  Crop & Resize  →  Flatten
  →  PCA (fit on train only)  →  50-dim feature vector  →  SVM  →  Prediction

Key concept / 核心概念:
  PCA (Principal Component Analysis) finds the directions in pixel space that
  capture the most variance across all training faces.  Each direction is
  visualised as an "Eigenface".  We keep only the top-k directions, compressing
  a 30 000-dim raw image into a 50-dim feature vector without losing much
  information.
  PCA 找到训练人脸像素空间中方差最大的方向（称为"特征脸"），
  将 30000 维原始图像压缩为 50 维特征向量，同时保留主要信息。
"""

import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

# Use a font that supports Chinese characters on Windows.
# 使用支持中文的字体，避免方块乱码。
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False   # fix minus sign rendering

# ──────────────────────────────────────────────────────────────────────────────
# Configuration / 全局配置
# ──────────────────────────────────────────────────────────────────────────────

# Dataset root relative to this script / 数据集根目录（相对于本脚本）
DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'kul-computer-vision-ga-1-2026')
TRAIN_CSV    = os.path.join(DATA_DIR, 'train_set.csv')
TEST_CSV     = os.path.join(DATA_DIR, 'test_set.csv')
TRAIN_DIR    = os.path.join(DATA_DIR, 'train')
TEST_DIR     = os.path.join(DATA_DIR, 'test')

# OpenCV ships haarcascades alongside its Python package; no download needed.
# OpenCV 安装包自带 haarcascade 文件，无需手动下载。
HAAR_PATH    = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

FACE_SIZE    = (100, 100)   # all detected faces are resized to this / 人脸统一缩放尺寸
N_COMPONENTS = 50           # number of PCA components (eigenfaces) to keep
                            # 保留的主成分数量（即保留多少个特征脸）

# Class metadata used across several plots / 类别元信息，多处复用
CLASS_IDS    = [0,           1,       2     ]
CLASS_COLORS = ['green',     'blue',  'red' ]
CLASS_LABELS = ['Look-alikes (0)', 'Jesse (1)', 'Mila (2)']


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — Data Loading  /  步骤0：数据加载
# ──────────────────────────────────────────────────────────────────────────────

def _load_images(df, img_dir, prefix):
    """Load .npy image files for every row in df and store in df['img'].

    每行图片以 BGR 格式存储在 .npy 文件中；
    这里转换为 RGB，方便 matplotlib 直接显示。

    Parameters
    ----------
    df      : pd.DataFrame  — must have an integer index matching file names
    img_dir : str           — directory containing the .npy files
    prefix  : str           — filename prefix, e.g. 'train' or 'test'
    """
    df['img'] = [
        cv2.cvtColor(                                          # BGR → RGB
            np.load(os.path.join(img_dir, f'{prefix}_{i}.npy'), allow_pickle=False),
            cv2.COLOR_BGR2RGB
        )
        for i in df.index
    ]


def load_data():
    """Read CSVs and attach the corresponding image arrays.
    读取 CSV 元数据并附加对应图像数组。

    Returns
    -------
    train : pd.DataFrame  — columns: name, class, img
    test  : pd.DataFrame  — columns: img
    """
    train = pd.read_csv(TRAIN_CSV, index_col=0)
    train.index = train.index.rename('id')

    test  = pd.read_csv(TEST_CSV,  index_col=0)
    test.index  = test.index.rename('id')

    _load_images(train, TRAIN_DIR, 'train')
    _load_images(test,  TEST_DIR,  'test')

    print(f"Loaded — Train: {len(train)} samples | Test: {len(test)} samples")
    return train, test


# ──────────────────────────────────────────────────────────────────────────────
# Step 0.3 — Preprocessing: HAAR Face Detector  /  步骤0.3：人脸检测预处理
# ──────────────────────────────────────────────────────────────────────────────

class HAARPreprocessor:
    """Locate the face in each image, crop it, and resize to FACE_SIZE.

    为什么要裁剪人脸？
    直接使用全图会引入大量背景噪声（衣服、场景等），
    裁剪到人脸区域后，每个像素都与分类任务直接相关。

    Why crop faces?
    Using the full image introduces background noise (clothes, scene, etc.).
    Cropping to the face region ensures every pixel is task-relevant.
    """

    def __init__(self, face_size=FACE_SIZE):
        self.face_size  = face_size
        self.classifier = cv2.CascadeClassifier(HAAR_PATH)
        # Fail early if the XML file could not be read.
        assert not self.classifier.empty(), \
            f"Could not load HAAR cascade from:\n  {HAAR_PATH}"

    def _extract_face(self, img):
        """Return a single cropped-and-resized face from img (H×W×3, RGB).

        1. Convert to grayscale — HAAR works on intensity, not colour.
           转为灰度图：HAAR 特征基于亮度梯度，不需要颜色信息。
        2. Run multiscale detection — tries the detector at various scales
           so it finds both large and small faces in the image.
           多尺度检测：以不同缩放比例反复扫描，找到大小不同的人脸。
        3. If no face is found, fall back to a centre-crop of the image
           so that every sample still has a valid array (avoids NaN).
           若未检测到人脸，则中心裁剪作为备选，确保不出现空值。
        """
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.2,    # shrink image by 20% at each scale level
            minNeighbors=5,     # require 5 overlapping detections to confirm a face
            minSize=(30, 30),   # ignore tiny detections
        )

        if len(faces) == 0:
            # Fallback: square centre-crop  /  备选：正方形中心裁剪
            h, w  = img.shape[:2]
            side  = min(h, w)
            y0    = (h - side) // 2
            x0    = (w - side) // 2
            face  = img[y0:y0+side, x0:x0+side]
        else:
            # Take the first detected face (usually the most prominent one).
            # 取第一个检测到的人脸（通常是最显著的）
            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]

        # Resize to a fixed resolution so all feature vectors have the same length.
        # 缩放到统一分辨率，确保所有特征向量等长。
        return cv2.resize(face, self.face_size, interpolation=cv2.INTER_AREA)

    def __call__(self, data):
        """Process a whole DataFrame; returns ndarray (N, H, W, 3)."""
        return np.stack([self._extract_face(row['img']) for _, row in data.iterrows()])


# ──────────────────────────────────────────────────────────────────────────────
# Step 1.0 — Base Feature Extractor  /  步骤1.0：基础特征提取器（基类）
# ──────────────────────────────────────────────────────────────────────────────

class IdentityFeatureExtractor:
    """Pass-through extractor — returns the input unchanged.
    恒等映射：直接返回输入，作为其他特征提取器的基类。
    """
    def transform(self, X):
        return X
    def __call__(self, X):
        return self.transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1.2 — PCA Feature Extractor  /  步骤1.2：PCA 特征提取器（特征脸）
# ──────────────────────────────────────────────────────────────────────────────

class PCAFeatureExtractor(IdentityFeatureExtractor):
    """Eigenface-based dimensionality reduction via PCA.

    How it works / 工作原理：
      1. Flatten each (H, W, 3) face image into a single vector of length H*W*3.
         将每张 (H, W, 3) 人脸图像展平为长度 H*W*3 的一维向量。
      2. PCA computes the mean face and the top-k eigenvectors of the
         covariance matrix.  These eigenvectors are the "Eigenfaces".
         PCA 计算平均脸和协方差矩阵的前 k 个特征向量，即"特征脸"。
      3. Each image is represented by its projection coefficients onto the
         k eigenfaces — a k-dimensional feature vector.
         每张图像用其在 k 个特征脸上的投影系数表示，得到 k 维特征向量。

    Critical rule / 关键规则：
      fit() must be called ONLY on training data to prevent data leakage.
      fit() 只能在训练集上调用，严禁使用测试集数据，否则导致数据泄露。

    Usage:
        extractor = PCAFeatureExtractor(n_components=50)
        extractor.fit(train_X)                          # learn eigenfaces
        train_feat = extractor.transform(train_X)       # → (N_train, 50)
        test_feat  = extractor.transform(test_X)        # → (N_test,  50)
        reconstructed = extractor.inverse_transform(train_feat)  # → (N, H, W, 3)
    """

    def __init__(self, n_components=N_COMPONENTS):
        self.n_components = n_components
        # sklearn's PCA wraps an efficient SVD decomposition.
        # sklearn 的 PCA 底层使用 SVD 分解，高效可靠。
        self.pca         = PCA(n_components=n_components)
        self.image_shape = None   # recorded in fit(); needed by inverse_transform

    def _flatten(self, X):
        """Flatten (N, H, W, C) → (N, H*W*C) as float64.

        Why float64? PCA's SVD computation is numerically sensitive;
        higher precision avoids accumulated rounding errors.
        为什么用 float64？SVD 对数值精度敏感，高精度可避免舍入误差累积。
        """
        return X.reshape(X.shape[0], -1).astype(np.float64)

    def fit(self, X):
        """Compute eigenfaces from training images.  Call once, on train only.
        从训练图像中计算特征脸。只调用一次，且只能用训练集。
        """
        self.image_shape = X.shape[1:]          # save (H, W, C) for reconstruction
        self.pca.fit(self._flatten(X))
        return self                              # enables chaining: extractor.fit(X).transform(X)

    def transform(self, X):
        """Project images onto the eigenface basis.  Returns (N, n_components).
        将图像投影到特征脸空间，返回 (N, n_components) 的低维表示。
        """
        return self.pca.transform(self._flatten(X))

    def inverse_transform(self, X_pca):
        """Reconstruct images from PCA coefficients.  Returns (N, H, W, C).
        从 PCA 系数重建图像，用于可视化重建质量。
        """
        return self.pca.inverse_transform(X_pca).reshape((-1,) + self.image_shape)

    def __call__(self, X):
        return self.transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers  /  可视化辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _norm_for_display(arr):
    """Min-max normalise an array to [0, 1] for matplotlib display.
    将数组归一化到 [0,1] 以便 matplotlib 正确显示（避免颜色截断）。
    """
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def plot_flowchart():
    """Draw the full pipeline as a flowchart using matplotlib patches.
    用 matplotlib 绘制整个流水线的流程图。
    """
    fig, ax = plt.subplots(figsize=(5, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis('off')

    # Each step: (y_centre, label_line1, label_line2, box_color)
    steps = [
        (26, '原始图像',           'Raw Images  (400×300×3)',        '#AED6F1'),
        (22, 'HAAR 人脸检测',      'Face Detection (HAAR Cascade)',   '#A9DFBF'),
        (18, '裁剪 & 缩放',        'Crop & Resize  →  100×100×3',    '#A9DFBF'),
        (14, '图像展平',            'Flatten  →  30 000-dim vector',   '#FAD7A0'),
        (10, 'PCA 降维',            f'PCA  →  {N_COMPONENTS}-dim vector'
                                    '\n(fit on train only)',           '#F1948A'),
        ( 6, 'SVM 分类器',          'SVM (RBF kernel)',                '#D7BDE2'),
        ( 2, '预测结果',            'Predicted Class  (0 / 1 / 2)',    '#AED6F1'),
    ]

    box_w, box_h = 7.0, 2.4
    x0 = 1.5   # left edge of every box

    for y_c, line1, line2, color in steps:
        # Draw rounded rectangle
        rect = FancyBboxPatch(
            (x0, y_c - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.15',
            facecolor=color, edgecolor='#555555', linewidth=1.2,
        )
        ax.add_patch(rect)
        # Two-line label inside the box
        ax.text(x0 + box_w / 2, y_c + 0.35, line1,
                ha='center', va='center', fontsize=10, fontweight='bold', color='#1A1A1A')
        ax.text(x0 + box_w / 2, y_c - 0.45, line2,
                ha='center', va='center', fontsize=8,  color='#333333')

    # Draw arrows between consecutive boxes
    for i in range(len(steps) - 1):
        y_top  = steps[i][0]   - box_h / 2        # bottom of upper box
        y_bot  = steps[i+1][0] + box_h / 2        # top of lower box
        x_mid  = x0 + box_w / 2
        ax.annotate(
            '', xy=(x_mid, y_bot + 0.05), xytext=(x_mid, y_top - 0.05),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
        )

    # Annotate the "data leakage prevention" note next to the PCA box
    ax.annotate(
        '⚠ fit() 仅在训练集调用\n   (prevents data leakage)',
        xy=(x0 + box_w, 10), xytext=(x0 + box_w + 0.2, 10),
        fontsize=7, color='#7B241C',
        va='center',
    )

    plt.title('PCA Face Recognition Pipeline', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('pipeline_flowchart.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved → pipeline_flowchart.png")


def plot_explained_variance(extractor):
    """Show how many components are needed to explain 95% of variance.
    展示需要多少主成分才能解释 95% 的方差。
    累积解释方差曲线是选择 n_components 超参数的主要依据。
    """
    # cumsum of individual explained variance ratios → cumulative curve
    cumvar = np.cumsum(extractor.pca.explained_variance_ratio_)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o', markersize=3)
    plt.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components  /  主成分数量')
    plt.ylabel('Cumulative Explained Variance  /  累积解释方差')
    plt.title('PCA: Explained Variance vs. Number of Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('explained_variance.png', dpi=100)
    plt.show()
    # cumvar is already computed above — no need to recompute
    print(f"First {extractor.n_components} PCs explain {cumvar[-1]*100:.1f}% of variance")
    print("Saved → explained_variance.png")


def plot_eigenfaces(extractor, n=16):
    """Visualise the top-n eigenfaces (principal components).

    每个特征脸是一个与原图同维度的向量，代表所有训练人脸中的某个"变化方向"：
      - 前几个特征脸捕捉全局光照变化和姿态差异（低频信息）。
      - 后面的特征脸捕捉细微的局部差异（高频信息）。
    Each eigenface is a vector the same size as the original image.
    Early eigenfaces capture global illumination/pose variation (low-freq).
    Later ones capture finer local differences (high-freq).
    """
    # components_ has shape (n_components, H*W*C); reshape back to image form
    eigenfaces = extractor.pca.components_[:n].reshape((n,) + extractor.image_shape)

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))

    for i, ax in enumerate(axes.flat):
        if i < n:
            # Normalise to [0,1] so the full colour range is used for display.
            # 归一化到 [0,1]，充分利用显示色域，使细节更清晰。
            ax.imshow(_norm_for_display(eigenfaces[i].astype(np.float64)))
            ax.set_title(f'PC {i+1}', fontsize=8)
        ax.axis('off')

    plt.suptitle('Eigenfaces  /  特征脸 (Top Principal Components)', fontsize=12)
    plt.tight_layout()
    plt.savefig('eigenfaces.png', dpi=100)
    plt.show()
    print("Saved → eigenfaces.png")


def plot_reconstruction(train_X, sample_idx=0):
    """Show how reconstruction quality improves with more components.

    通过对比不同 n_components 下的重建图像，直观展示：
      - 主成分太少 → 图像模糊（信息丢失）
      - 主成分足够多 → 重建接近原图
    This plot justifies our choice of N_COMPONENTS.
    """
    # Can't use more components than training samples
    max_n  = train_X.shape[0]
    n_vals = [n for n in [5, 10, 20, 50, 75] if n < max_n]

    fig, axes = plt.subplots(1, len(n_vals) + 1, figsize=(18, 3))
    axes[0].imshow(np.clip(train_X[sample_idx], 0, 255).astype(np.uint8))
    axes[0].set_title('Original')
    axes[0].axis('off')

    for ax, n in zip(axes[1:], n_vals):
        # Each reconstruction requires a separately fitted PCA with n components.
        # 每个重建需要用对应 n 单独拟合一个 PCA，因为 sklearn 的 PCA 在 fit 时
        # 就固定了保留的主成分数量，无法事后截断。
        extractor = PCAFeatureExtractor(n_components=n).fit(train_X)
        rec = extractor.inverse_transform(extractor.transform(train_X[sample_idx:sample_idx+1]))
        ax.imshow(np.clip(rec[0], 0, 255).astype(np.uint8))
        ax.set_title(f'n={n}')
        ax.axis('off')

    plt.suptitle('Reconstruction Quality vs. Number of Components  /  不同主成分数量下的重建质量')
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=100)
    plt.show()
    print("Saved → reconstruction.png")


def plot_feature_space(train_features, train_y, n_components):
    """t-SNE projection of PCA features to 2D for visual separability check.

    PCA 降维后的特征仍有 50 维，人眼无法直接观察。
    t-SNE 进一步将其投影到 2D 平面，帮助我们判断：
      - 三类样本在特征空间中是否形成可分离的簇？
      - 若各类点已明显分开，分类器应能取得较好效果。

    t-SNE reduces 50-dim PCA features to 2D for visual inspection.
    Well-separated clusters indicate the features carry discriminative information.

    n_components : int — used in the plot title to show which PCA was used
    """
    # perplexity≈15 works well for small datasets (~80 points)
    # 对于小数据集（约80个点），perplexity=15 是合适的选择
    tsne     = TSNE(n_components=2, random_state=42, perplexity=15)
    embedded = tsne.fit_transform(train_features)

    plt.figure(figsize=(7, 5))
    for cls, color, label in zip(CLASS_IDS, CLASS_COLORS, CLASS_LABELS):
        mask = train_y == cls
        plt.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=color, label=label, alpha=0.75, s=60,
            edgecolors='k', linewidths=0.3,
        )
    plt.title(f't-SNE of PCA Features  (n_components={n_components})\n'
              f't-SNE：PCA 特征的二维投影')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_space_tsne.png', dpi=100)
    plt.show()
    print("Saved → feature_space_tsne.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point  /  主程序
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Step 0: load raw data ──────────────────────────────────────────────────
    train, test = load_data()

    # ── Step 0.3: detect & crop faces ─────────────────────────────────────────
    preprocessor = HAARPreprocessor(face_size=FACE_SIZE)
    train_X = preprocessor(train)          # shape: (80,  100, 100, 3)
    train_y = train['class'].values        # labels: 0=look-alike, 1=Jesse, 2=Mila
    test_X  = preprocessor(test)           # shape: (1816, 100, 100, 3)
    print(f"Preprocessed — train_X: {train_X.shape} | test_X: {test_X.shape}")

    # ── Step 1.2: fit PCA on training set only ─────────────────────────────────
    # fit() learns the eigenfaces; transform() applies the projection.
    # 仅在训练集上调用 fit()，学习特征脸基；transform() 应用投影。
    pca_extractor  = PCAFeatureExtractor(n_components=N_COMPONENTS)
    pca_extractor.fit(train_X)

    train_features = pca_extractor.transform(train_X)   # (80,   N_COMPONENTS)
    test_features  = pca_extractor.transform(test_X)    # (1816, N_COMPONENTS)

    # ── Visualisations ─────────────────────────────────────────────────────────
    plot_flowchart()
    plot_explained_variance(pca_extractor)
    plot_eigenfaces(pca_extractor, n=16)
    plot_reconstruction(train_X, sample_idx=0)
    plot_feature_space(train_features, train_y, N_COMPONENTS)

    # ── Step 3: train SVM classifier on PCA features ──────────────────────────
    # RBF kernel handles non-linear boundaries in the 50-dim feature space.
    # RBF 核可以处理 50 维特征空间中的非线性决策边界。
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(train_features, train_y)

    train_pred = svm.predict(train_features)
    print(f"\nSVM train accuracy (PCA {N_COMPONENTS} components): "
          f"{accuracy_score(train_y, train_pred):.4f}")

    # ── Step 5: generate submission file ──────────────────────────────────────
    test_pred        = svm.predict(test_features)
    submission       = test.copy().drop('img', axis=1)
    submission['class'] = test_pred
    submission.to_csv('submission_pca.csv')
    print("Saved → submission_pca.csv")
