"""不依赖 insightface 包：仅测相似度与决策逻辑。"""
from __future__ import annotations

import numpy as np
import pytest

from dl_pipeline.inference.insightface_dual_verifier import (
    dual_verifier_predict,
    top_k_mean_cosine_similarity,
    youden_threshold,
)


def test_top_k_mean_cosine():
    g = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    p = np.array([1.0, 0.0], dtype=np.float32)
    p /= np.linalg.norm(p)
    m = top_k_mean_cosine_similarity(p, g, k=2)
    assert m == pytest.approx(1.0, abs=1e-5)


def test_youden_simple():
    pos = np.array([0.9, 0.8, 0.85])
    neg = np.array([0.1, 0.2, 0.15])
    t, meta = youden_threshold(pos, neg)
    assert t > 0.15
    assert t <= 0.9
    assert "youden_j" in meta


def test_dual_verifier_branches():
    assert dual_verifier_predict(0.9, 0.1, 0.5, 0.5) == 1
    assert dual_verifier_predict(0.1, 0.9, 0.5, 0.5) == 2
    assert dual_verifier_predict(0.1, 0.1, 0.5, 0.5) == 0
    assert dual_verifier_predict(0.9, 0.9, 0.5, 0.5) == 1
    assert dual_verifier_predict(0.6, 0.9, 0.5, 0.5) == 2
