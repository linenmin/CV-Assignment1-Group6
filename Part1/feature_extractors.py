"""
feature_extractors.py
---------------------
Feature extraction classes and functions:
  - IdentityFeatureExtractor   (pass-through baseline)
  - HOGFeatureExtractor        (stub / for extension)
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


# ── HOG stub (for extension) ──────────────────────────────────────────────────

class HOGFeatureExtractor(IdentityFeatureExtractor):
    """Placeholder HOG extractor — use the module-level extract_hog() instead."""

    def __init__(self, **params):
        self.params = params

    def transform(self, X):
        raise NotImplementedError("Use extract_hog() from this module.")


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
