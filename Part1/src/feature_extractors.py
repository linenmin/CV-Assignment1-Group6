"""
feature_extractors.py
---------------------
Feature extraction classes and functions:
  - IdentityFeatureExtractor   (pass-through baseline)
  - PCAFeatureExtractor        (Eigenfaces via sklearn PCA)
  - extract_hog()              (OpenCV HOG, 1764-dim)
  - extract_lbp()              (vectorised LBP histogram, 256-dim)
  - extract_combined()         (HOG + LBP concatenated, 2020-dim)
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA


# ── Base extractor ────────────────────────────────────────────────────────────

class IdentityFeatureExtractor:
    """A simple pass-through: returns the input unchanged."""

    def transform(self, X):
        return X

    def __call__(self, X):
        return self.transform(X)


# ── PCA / Eigenfaces ──────────────────────────────────────────────────────────

class PCAFeatureExtractor(IdentityFeatureExtractor):
    """Eigenface-based dimensionality reduction via PCA.

   

    Usage
    -----
    extractor = PCAFeatureExtractor(n_components=50)
    extractor.fit(train_X_clean)                     # learn eigenfaces from train only
    train_feat = extractor.transform(train_X_clean)  # (N_train, 50)
    test_feat  = extractor.transform(test_X)         # (N_test,  50)
    recon      = extractor.inverse_transform(train_feat)  # (N, H, W, C)
    """

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.image_shape = None  # set in fit(); needed by inverse_transform

    def _flatten(self, X):
        """(N, H, W, C) -> (N, H*W*C) as float64 for numerical stability."""
        return X.reshape(X.shape[0], -1).astype(np.float64)

    def fit(self, X):
        """Learn eigenfaces from training images. Call ONCE, on train only."""
        self.image_shape = X.shape[1:]
        self.pca.fit(self._flatten(X))
        return self  # allows chaining: extractor.fit(X).transform(X)

    def transform(self, X):
        """Project images onto eigenface basis -> (N, n_components)."""
        return self.pca.transform(self._flatten(X))

    def inverse_transform(self, X_pca):
        """Reconstruct images from PCA coefficients -> (N, H, W, C)."""
        flat = self.pca.inverse_transform(X_pca)
        return flat.reshape((-1,) + self.image_shape)

    def __call__(self, X):
        return self.transform(X)


# ── HOG descriptor ────────────────────────────────────────────────────────────

_hog_descriptor = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)  # 1764-dim per image


def extract_hog(X):
    """(N, H, W, 3) RGB -> (N, 1764) HOG features."""
    feats = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        feats.append(_hog_descriptor.compute(resized).ravel())
    return np.array(feats, dtype=np.float64)


# ── LBP descriptor ───────────────────────────────────────────────────────────

def _lbp_image(gray):
    """Compute radius-1, 8-neighbour LBP for a single grayscale image."""
    c = gray[1:-1, 1:-1].astype(np.int16)
    shifts = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 2:],   gray[2:,   2:],   gray[2:,   1:-1],
        gray[2:,   0:-2], gray[1:-1, 0:-2],
    ]
    code = np.zeros_like(c, dtype=np.uint8)
    for bit, nb in enumerate(shifts):
        code |= ((nb.astype(np.int16) >= c).astype(np.uint8) << bit)
    return code


def extract_lbp(X, n_bins=256):
    """(N, H, W, 3) RGB -> (N, 256) normalised LBP histogram."""
    feats = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        lbp = _lbp_image(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 256))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
        feats.append(hist)
    return np.array(feats, dtype=np.float64)


# ── Combined HOG + LBP ────────────────────────────────────────────────────────

def extract_combined(X):
    """(N, H, W, 3) RGB -> (N, 2020) concatenated HOG + LBP features."""
    return np.hstack([extract_hog(X), extract_lbp(X)])





"""
 Technical Notes
    ---------------
    1. 2D Matrix Conversion: We converted the 4D image tensor into a 2D matrix by 
       flattening each 100 x 100 x 3 image into a 1D vector of length 30,000. 
       Stacking the N=80 training samples results in a 80 x 30,000 data matrix.

    2. Exploiting Dimensionality & SVD vs. Eigenvalue Decomposition: A standard 
       eigenvalue decomposition would require computing a 30,000 x 30,000 
       covariance matrix, which is computationally prohibitive. Because our 
       number of samples is much smaller than the number of features (N << D), 
       we exploited this dimensionality by using Singular Value Decomposition (SVD) 
       provided by sklearn. SVD efficiently computes the principal components 
       by implicitly operating on an 80 x 80 matrix, vastly improving effectiveness.

    3. Mean Subtraction: Yes, mean subtraction is strictly required and applied 
       internally during fit(). If the data is not mean-centered, the first 
       principal component would simply point toward the mean of the dataset 
       rather than capturing the direction of maximum variance.

    4. Pre-processing Steps: Before PCA, three critical pre-processing steps 
       were required:
       (a) HAAR Cascade face extraction to remove background noise, ensuring 
           PCA focuses on facial variance.
       (b) Center-cropping fallback for failed detections to avoid NaN values 
           that would break SVD.
       (c) Resizing all images to 100 x 100 to ensure every flattened vector 
           has the exact same dimension (30,000).
"""