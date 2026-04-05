"""
experiments.py
--------------
Pipeline factory functions and helpers for GridSearchCV experiments.

Functions / Classes
-------------------
make_pca_svm_pipeline()
    Returns a fresh PCA -> StandardScaler -> SVM(RBF) sklearn Pipeline.

run_grid_search(features, labels, param_grid, cv=5)
    Fit a GridSearchCV on *features* and print/return the best result.

HonestResult
    Lightweight wrapper that gives honest (aug-aware) models the same
    .best_score_ / .best_estimator_ / .best_params_ interface as
    sklearn GridSearchCV objects, for unified model selection in cell 64.

select_best_honest_model(candidates, leaky_keys)
    Pick the best non-leaky model from the candidates dict.
"""

import pandas as pd
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ── Pipeline factories ────────────────────────────────────────────────────────

def make_pca_svm_pipeline():
    """Return a fresh PCA -> StandardScaler -> SVM(RBF) pipeline."""
    return Pipeline([
        ('pca',    skPCA()),
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', random_state=42)),
    ])


def make_scaler_svm_pipeline():
    """Return a fresh StandardScaler -> SVM(RBF) pipeline (no PCA)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', random_state=42)),
    ])


# ── GridSearch helper ─────────────────────────────────────────────────────────

def _sanitize_param_grid(features, labels, param_grid, n_splits):
    """Drop PCA component counts that are impossible inside CV folds."""
    if 'pca__n_components' not in param_grid:
        return param_grid, []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    min_train_samples = min(len(tr_idx) for tr_idx, _ in cv.split(features, labels))
    max_components = min(min_train_samples, features.shape[1])

    cleaned = dict(param_grid)
    original_values = list(cleaned['pca__n_components'])
    valid_values = [value for value in original_values if value <= max_components]
    removed_values = [value for value in original_values if value > max_components]

    if not valid_values:
        raise ValueError(
            "No valid PCA component counts remain for CV: "
            f"max allowed is {max_components}, requested {original_values}"
        )

    cleaned['pca__n_components'] = valid_values
    return cleaned, removed_values

def run_grid_search(features, labels, param_grid, label='', n_splits=5):
    """Fit GridSearchCV and return the fitted object.

    Parameters
    ----------
    features : ndarray  pre-extracted feature matrix
    labels   : ndarray
    param_grid : dict
    label    : str   printed in the summary line
    n_splits : int

    Returns
    -------
    gs : fitted GridSearchCV
    """
    cleaned_grid, removed_values = _sanitize_param_grid(
        features, labels, param_grid, n_splits
    )

    pipe = make_pca_svm_pipeline() if any('pca' in k for k in cleaned_grid) \
        else make_scaler_svm_pipeline()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, cleaned_grid, cv=cv,
                      scoring='accuracy', n_jobs=-1, refit=True)
    gs.fit(features, labels)
    if removed_values:
        print(
            f"{label:40s} removed invalid pca__n_components={removed_values} "
            f"for {n_splits}-fold CV"
        )
    print(f'{label:40s} CV={gs.best_score_:.4f}  {gs.best_params_}')
    return gs


# ── Honest-result wrapper ─────────────────────────────────────────────────────

class HonestResult:
    """Wraps aug-aware CV results to match the sklearn GridSearchCV interface.

    Lets HonestResult objects participate in the same _candidates dict as
    GridSearchCV objects so model selection code is uniform.
    """

    def __init__(self, score, estimator, params):
        self.best_score_     = score
        self.best_estimator_ = estimator
        self.best_params_    = params


# ── Model selection ───────────────────────────────────────────────────────────

def select_best_honest_model(candidates: dict, leaky_keys: set):
    """Return the name of the best non-leaky model in *candidates*.

    Parameters
    ----------
    candidates : dict
        Keys are model names; values are (gs_or_honest_result, ...) tuples.
    leaky_keys : set
        Model names to exclude (their CV scores are inflated by data leakage).

    Returns
    -------
    best_name : str
    """
    return max(
        (k for k in candidates if k not in leaky_keys),
        key=lambda k: candidates[k][0].best_score_,
    )


def format_honest_candidate_label(best_honest_name: str) -> str:
    """Return the display label used for the saved honest-model candidate."""
    return f"{best_honest_name} (aug-aware, honest)"


def print_model_comparison(candidates: dict, leaky_keys: set):
    """Print a sorted table comparing all candidate models."""
    rows = []
    for name, (gs, _, _) in candidates.items():
        rows.append({
            'Pipeline':     name,
            'CV Accuracy':  gs.best_score_,
            'Leakage-free': 'NO' if name in leaky_keys else 'Yes',
        })
    df = (pd.DataFrame(rows)
            .sort_values('CV Accuracy', ascending=False)
            .reset_index(drop=True))
    print(df.to_string(index=False))
    return df
