"""
augmentation.py
---------------
Data augmentation utilities and augmentation-aware cross-validation.

Functions
---------
augment_dataset(X, y)
    Returns original + h-flip + ±10° rotated copies (4x dataset size).

run_aug_aware_cv(extractor_fn, make_pipe_fn, param_grid, X_images, y, ...)
    Leakage-free CV: augment only inside each training fold, validate on
    original images. Avoids the inflated CV scores caused by augmenting
    the full dataset before splitting.
"""

import numpy as np
from scipy.ndimage import rotate as nd_rotate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold


def augment_dataset(X, y):
    """Return original + h-flip + rotate+10° + rotate-10° copies.

    Parameters
    ----------
    X : ndarray, shape (N, H, W, C)  uint8 images
    y : ndarray, shape (N,)          labels

    Returns
    -------
    X_aug : ndarray, shape (4*N, H, W, C)
    y_aug : ndarray, shape (4*N,)
    """
    parts_X, parts_y = [X], [y]

    # Horizontal flip
    parts_X.append(X[:, :, ::-1, :].copy())
    parts_y.append(y)

    # ±10° rotation (reshape=False keeps size the same)
    for angle in (10, -10):
        rotated = np.array([
            np.clip(nd_rotate(img, angle, axes=(0, 1), reshape=False), 0, 255
                    ).astype(np.uint8)
            for img in X
        ])
        parts_X.append(rotated)
        parts_y.append(y)

    return np.concatenate(parts_X, axis=0), np.concatenate(parts_y, axis=0)


def run_aug_aware_cv(extractor_fn, make_pipe_fn, param_grid,
                     X_images, y, n_splits=5, random_state=42):
    """Leakage-free cross-validation with in-fold augmentation.

    Augments ONLY the training fold, validates on original (non-augmented)
    images. This prevents the CV-vs-Kaggle gap caused by augmenting before
    splitting (where val fold may contain augmented versions of train images).

    Parameters
    ----------
    extractor_fn : callable
        Function that maps (N, H, W, 3) -> (N, d) feature array.
    make_pipe_fn : callable
        Returns a fresh sklearn Pipeline instance for each parameter combo.
    param_grid : dict
        Parameter grid passed to sklearn's ParameterGrid.
    X_images : ndarray, shape (N, H, W, C)
        Original (non-augmented) image arrays.
    y : ndarray, shape (N,)
        Labels.
    n_splits : int
        Number of stratified folds (default 5).
    random_state : int

    Returns
    -------
    best_params : dict
    best_score  : float
    results     : list of dicts sorted by mean_cv descending
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score, best_params, results = -1, None, []

    for params in ParameterGrid(param_grid):
        fold_scores = []
        for tr_idx, val_idx in skf.split(X_images, y):
            X_tr_aug, y_tr_aug = augment_dataset(X_images[tr_idx], y[tr_idx])
            X_tr_feat  = extractor_fn(X_tr_aug)
            X_val_feat = extractor_fn(X_images[val_idx])

            pipe = make_pipe_fn()
            pipe.set_params(**params)
            pipe.fit(X_tr_feat, y_tr_aug)
            fold_scores.append(accuracy_score(y[val_idx], pipe.predict(X_val_feat)))

        mean_cv = np.mean(fold_scores)
        results.append({
            'params':  params,
            'mean_cv': mean_cv,
            'std_cv':  np.std(fold_scores),
        })
        if mean_cv > best_score:
            best_score, best_params = mean_cv, params

    return best_params, best_score, sorted(results, key=lambda x: -x['mean_cv'])
