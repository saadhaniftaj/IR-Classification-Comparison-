"""
knn.py
------
k-Nearest Neighbors classifier for text using cosine similarity.

sklearn's KNeighborsClassifier supports cosine metric only via
'algorithm=brute'. This wrapper exposes a clean interface and
also demonstrates manual cosine-kNN for report explanation.

Algorithm:
    For a query document q:
        1. Compute cosine_sim(q, d) for all d in training set
        2. Select k documents with highest similarity
        3. Assign the majority class label among those k neighbors

    cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import issparse


class KNNClassifier:
    """
    Cosine-similarity kNN wrapper around sklearn's KNeighborsClassifier.

    Parameters
    ----------
    k         : int   — number of neighbors (default 5)
    weights   : str   — 'uniform' or 'distance' weighting of neighbors
    """

    def __init__(self, k: int = 5, weights: str = "distance"):
        self.k       = k
        self.weights = weights
        self.model   = KNeighborsClassifier(
            n_neighbors = k,
            metric      = "cosine",
            algorithm   = "brute",    # required for cosine metric
            weights     = weights,
            n_jobs      = -1          # use all CPU cores
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    @property
    def name(self) -> str:
        return f"kNN (k={self.k}, weights={self.weights})"
