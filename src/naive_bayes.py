"""
naive_bayes.py
--------------
Naive Bayes classifier wrapper using sklearn's MultinomialNB.
Also includes a BernoulliNB variant for comparison.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes for text classification.
    ComplementNB is used as it performs better on imbalanced datasets.

    Math:
        P(c|d) ∝ P(c) * ∏ P(t|c)^(tf_t,d)
    where:
        P(c)    = prior probability of class c
        P(t|c)  = smoothed term likelihood given class c (Laplace smoothing)
    """

    def __init__(self, alpha: float = 1.0, variant: str = "complement"):
        """
        Parameters
        ----------
        alpha   : Laplace smoothing parameter (default 1.0)
        variant : 'multinomial' or 'complement' (ComplementNB)
        """
        self.alpha   = alpha
        self.variant = variant

        if variant == "complement":
            self.model = ComplementNB(alpha=alpha)
        else:
            self.model = MultinomialNB(alpha=alpha)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    @property
    def name(self) -> str:
        return f"Naive Bayes ({self.variant}, α={self.alpha})"
