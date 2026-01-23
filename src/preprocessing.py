import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from skbio.stats.composition import clr


# =========================
# Zero replacement + CLR
# =========================

def multiplicative_replacement(X, delta=1e-6):
    """
    Replace zeros using multiplicative replacement.
    X must be a 2D numpy array (samples x features).
    """
    X = np.asarray(X, dtype=float)
    X_out = X.copy()

    for i in range(X.shape[0]):
        row = X_out[i]
        zero_mask = (row == 0)
        k = int(zero_mask.sum())

        if k == 0:
            continue

        non_zero_mask = ~zero_mask
        non_zero_sum = row[non_zero_mask].sum()

        # Edge case: all zeros
        if non_zero_sum == 0:
            row[:] = delta
        else:
            row[zero_mask] = delta
            row[non_zero_mask] *= (1 - k * delta) / non_zero_sum

        X_out[i] = row

    return X_out


class CLRTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer:
      - multiplicative zero replacement
      - CLR transformation
    """

    def __init__(self, delta=1e-6):
        self.delta = delta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = multiplicative_replacement(X, delta=self.delta)
        return clr(X)


# =========================
# Pipeline builder
# =========================

def build_preprocessing_pipeline(
        use_clr=True,
        use_scaler=True,
        k_features=None
):
    """
    Build a preprocessing pipeline.

    Parameters
    ----------
    use_clr : bool
        Apply CLR + multiplicative replacement
    use_scaler : bool
        Apply StandardScaler
    k_features : int or None
        Number of features for SelectKBest.
        If None, feature selection is skipped.
    """

    steps = []

    if use_clr:
        steps.append(("clr", CLRTransformer()))

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if k_features is not None:
        steps.append(
            ("feature_selection", SelectKBest(score_func=f_classif, k=k_features))
        )

    return Pipeline(steps)
