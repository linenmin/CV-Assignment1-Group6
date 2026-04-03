"""
classifiers.py
--------------
Classifier wrappers used in the pipeline:
  - RandomClassificationModel  (baseline random classifier)
  - SVMClassifier              (SVM with RBF kernel)
"""

import numpy as np
from sklearn.svm import SVC


class RandomClassificationModel:
    """Random classifier, draws a random sample based on the class distribution
    observed during training."""

    def fit(self, X, y):
        """Learn the class distribution from y.

        Parameters
        ----------
        X : array-like
            Training features (unused — kept for API compatibility).
        y : array-like
            Training labels.

        Returns
        -------
        self
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_ratio_ = counts / counts.sum()
        return self

    def predict(self, X):
        """Sample labels according to the fitted class distribution.

        Parameters
        ----------
        X : array-like, shape (N, ...)

        Returns
        -------
        y_star : ndarray, shape (N,)
        """
        np.random.seed(0)
        return np.random.choice(self.classes_, size=X.shape[0], p=self.class_ratio_)

    def __call__(self, X):
        return self.predict(X)


class SVMClassifier:
    """SVM with RBF kernel, wrapping sklearn's SVC for use in the pipeline."""

    def __init__(self, C=10, gamma='scale', kernel='rbf'):
        self.model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.predict(X)
