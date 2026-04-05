"""
visualization.py
----------------
Plotting helpers used throughout the notebook.

Functions
---------
plot_image_sequence(data, n, imgs_per_row=7)
    Display a grid of images from an array.

norm_for_display(arr)
    Min-max normalise an array to [0, 1] for matplotlib rendering.

plot_reconstruction_progression(pca_extractor, image, k_values)
    Show one face reconstructed with gradually more eigenfaces.

plot_pca_component_scatter(features, labels)
    Plot the first two PCA coordinates with class-coloured markers.
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_image_sequence(data, n, imgs_per_row=7):
    """Plot the first *n* images from *data* in a grid.

    Parameters
    ----------
    data : array-like, shape (N, H, W, C)
    n : int
        Number of images to display.
    imgs_per_row : int
    """
    n_rows = 1 + int(n / (imgs_per_row + 1))
    n_cols = min(imgs_per_row, n)
    f, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    for i in range(n):
        if n == 1:
            ax.imshow(data[i])
            ax.axis('off')
        elif n_rows > 1:
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].imshow(data[i])
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].axis('off')
        else:
            ax[int(i % n)].imshow(data[i])
            ax[int(i % n)].axis('off')
    plt.tight_layout()
    plt.show()


def norm_for_display(arr):
    """Min-max normalise *arr* to [0, 1] so matplotlib renders without clipping."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _imshow(ax, image):
    """Display RGB or grayscale image arrays without matplotlib warnings."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 1:
        ax.imshow(image[..., 0], cmap='gray')
    elif image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)


def plot_eigenfaces(pca_extractor, n_show=16):
    """Display the top *n_show* eigenfaces from a fitted PCAFeatureExtractor."""
    eigenfaces = pca_extractor.pca.components_[:n_show].reshape(
        (n_show,) + pca_extractor.image_shape
    )
    fig, axes = plt.subplots(2, n_show // 2, figsize=(n_show, 4.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(norm_for_display(eigenfaces[i].astype(np.float64)))
        ax.set_title(f'PC {i + 1}', fontsize=8)
        ax.axis('off')
    plt.suptitle('Top Eigenfaces (Principal Components)', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_explained_variance(pca_extractor, train_X_clean, n_components_chosen):
    """Plot cumulative explained variance curve for PCA."""
    from sklearn.decomposition import PCA
    pca_full = PCA(n_components=min(train_X_clean.shape[0], 80)).fit(
        train_X_clean.reshape(train_X_clean.shape[0], -1).astype(np.float64)
    )
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o', markersize=3)
    plt.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    plt.axvline(n_components_chosen, color='g', linestyle=':',
                label=f'n={n_components_chosen} chosen')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Explained Variance vs. Number of Components')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_progression(pca_extractor, image, k_values=(1, 5, 10, 20, 50)):
    """Show one face reconstructed with progressively more principal components."""
    image_batch = np.asarray(image)[None, ...]
    coeffs = pca_extractor.transform(image_batch)
    k_values = [int(k) for k in k_values]

    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(2.5 * (len(k_values) + 1), 3.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    _imshow(axes[0], norm_for_display(image.astype(np.float64)))
    axes[0].set_title('Original')
    axes[0].axis('off')

    for ax, k in zip(axes[1:], k_values):
        truncated = np.zeros_like(coeffs)
        truncated[:, :k] = coeffs[:, :k]
        recon = pca_extractor.inverse_transform(truncated)[0]
        _imshow(ax, norm_for_display(recon.astype(np.float64)))
        ax.set_title(f'k = {k}')
        ax.axis('off')

    plt.suptitle('Face Reconstruction with Increasing Eigenfaces', fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_pca_component_scatter(features, labels):
    """Plot the first two PCA coordinates of the dataset."""
    CLASS_IDS = [0, 1, 2]
    CLASS_COLORS = ['green', 'blue', 'red']
    CLASS_LABELS = ['Look-alikes (0)', 'Jesse (1)', 'Mila (2)']

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, color, label in zip(CLASS_IDS, CLASS_COLORS, CLASS_LABELS):
        mask = labels == cls
        ax.scatter(
            features[mask, 0], features[mask, 1],
            c=color, label=label, alpha=0.75, s=70,
            edgecolors='k', linewidths=0.3,
        )
    ax.set_title('Faces Projected onto the First Two Principal Components')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_tsne(features, labels, n_components_label=''):
    """Plot a 2-D t-SNE embedding of *features* coloured by *labels*."""
    from sklearn.manifold import TSNE
    CLASS_IDS    = [0, 1, 2]
    CLASS_COLORS = ['green', 'blue', 'red']
    CLASS_LABELS = ['Look-alikes (0)', 'Jesse (1)', 'Mila (2)']

    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(7, 5))
    for cls, color, label in zip(CLASS_IDS, CLASS_COLORS, CLASS_LABELS):
        mask = labels == cls
        plt.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=color, label=label, alpha=0.75, s=70,
            edgecolors='k', linewidths=0.3,
        )
    plt.title(f't-SNE of PCA Features  {n_components_label}')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
