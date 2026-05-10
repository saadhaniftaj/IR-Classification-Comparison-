"""
rocchio.py
----------
Manual implementation of the Rocchio Classification algorithm.

Algorithm:
    For each class c, compute a centroid vector:
        μ(c) = (1/|Dc|) * Σ d ∈ Dc  (TF-IDF vector of document d)

    At prediction time, assign document d to the class whose centroid
    is closest under cosine similarity:
        class(d) = argmax_c  cos(d, μ(c))

The classic Rocchio formula also supports a negative centroid term:
        μ̃(c) = α * μ(c) - β * (1/(|D| - |Dc|)) * Σ d ∉ Dc  d_vec
    
    Here α=1, β=0 (query expansion disabled) for pure classification.
"""

import numpy as np
from sklearn.preprocessing import normalize


class RocchioClassifier:
    """
    Rocchio centroid-based text classifier.

    Parameters
    ----------
    alpha : float  — weight on positive class centroid (default 1.0)
    beta  : float  — weight on negative class centroid (default 0.0)
                     Set > 0 to enable Rocchio with negative feedback.
    metric: str    — 'cosine' (default) or 'euclidean'
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, metric: str = "cosine"):
        self.alpha    = alpha
        self.beta     = beta
        self.metric   = metric
        self.centroids_ = None   # shape: (n_classes, n_features)
        self.classes_   = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Compute per-class centroids from training data.
        X_train : sparse or dense matrix (n_samples, n_features)
        y_train : array-like of int labels
        """
        self.classes_ = np.unique(y_train)
        n_classes     = len(self.classes_)

        # Convert sparse to dense for centroid arithmetic
        X = X_train.toarray() if hasattr(X_train, "toarray") else np.array(X_train)

        positive_centroids = np.zeros((n_classes, X.shape[1]))

        for idx, c in enumerate(self.classes_):
            mask = (y_train == c)
            if mask.sum() > 0:
                positive_centroids[idx] = X[mask].mean(axis=0)

        if self.beta > 0:
            # Full Rocchio with negative centroid subtraction
            global_mean = X.mean(axis=0)
            n_total = X.shape[0]
            self.centroids_ = np.zeros_like(positive_centroids)
            for idx, c in enumerate(self.classes_):
                mask = (y_train == c)
                n_c  = mask.sum()
                n_nc = n_total - n_c
                neg_centroid = ((global_mean * n_total) - positive_centroids[idx] * n_c) / max(n_nc, 1)
                self.centroids_[idx] = (self.alpha * positive_centroids[idx]
                                        - self.beta  * neg_centroid)
        else:
            self.centroids_ = positive_centroids

        # L2-normalize centroids for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(self.centroids_, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.centroids_ = self.centroids_ / norms

        return self

    # ------------------------------------------------------------------
    def predict(self, X_test):
        """Assign each document to its nearest centroid."""
        X = X_test.toarray() if hasattr(X_test, "toarray") else np.array(X_test)

        if self.metric == "cosine":
            # L2-normalize documents
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X_norm = X / norms
            # Similarity = dot product (since both sides are L2-normalized)
            sims = X_norm @ self.centroids_.T       # (n_test, n_classes)
            pred_idx = np.argmax(sims, axis=1)
        else:
            # Euclidean distance
            dists = np.array([
                np.linalg.norm(X - self.centroids_[i], axis=1)
                for i in range(len(self.classes_))
            ]).T                                    # (n_test, n_classes)
            pred_idx = np.argmin(dists, axis=1)

        return self.classes_[pred_idx]

    @property
    def name(self) -> str:
        return f"Rocchio (α={self.alpha}, β={self.beta}, metric={self.metric})"
