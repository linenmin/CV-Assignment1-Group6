import numpy as np

from experiments import _sanitize_param_grid, format_honest_candidate_label


def test_sanitize_param_grid_removes_illegal_pca_components():
    features = np.random.RandomState(0).randn(80, 120)
    labels = np.array([0] * 20 + [1] * 30 + [2] * 30)
    param_grid = {
        "pca__n_components": [30, 50, 80, 100],
        "svm__C": [1, 10],
        "svm__gamma": ["scale"],
    }

    cleaned, removed = _sanitize_param_grid(features, labels, param_grid, n_splits=5)

    assert cleaned["pca__n_components"] == [30, 50]
    assert removed == [80, 100]


def test_format_honest_candidate_label_preserves_winner_prefix():
    assert (
        format_honest_candidate_label("[H] HOG+PCA+SVM")
        == "[H] HOG+PCA+SVM (aug-aware, honest)"
    )


def test_format_honest_candidate_label_supports_i_pipeline():
    assert (
        format_honest_candidate_label("[I] HOG+LBP+PCA+SVM")
        == "[I] HOG+LBP+PCA+SVM (aug-aware, honest)"
    )
