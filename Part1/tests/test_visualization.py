import matplotlib
matplotlib.use("Agg")

import numpy as np

from feature_extractors import PCAFeatureExtractor
from visualization import (
    plot_pca_component_scatter,
    plot_reconstruction_progression,
)


def _fit_demo_extractor():
    rng = np.random.RandomState(0)
    images = rng.randint(0, 255, size=(10, 8, 8, 3), dtype=np.uint8)
    extractor = PCAFeatureExtractor(n_components=5)
    extractor.fit(images)
    return extractor, images


def test_plot_reconstruction_progression_returns_expected_axes():
    extractor, images = _fit_demo_extractor()

    fig, axes = plot_reconstruction_progression(
        extractor,
        images[0],
        k_values=[1, 3, 5],
    )

    assert len(axes) == 4
    assert axes[0].get_title() == "Original"
    assert axes[-1].get_title() == "k = 5"
    fig.canvas.draw()


def test_plot_pca_component_scatter_uses_pc_axis_labels():
    rng = np.random.RandomState(1)
    features = rng.randn(10, 5)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])

    fig, ax = plot_pca_component_scatter(features, labels)

    assert ax.get_xlabel() == "PC1"
    assert ax.get_ylabel() == "PC2"
    assert "Principal Components" in ax.get_title()
    fig.canvas.draw()
