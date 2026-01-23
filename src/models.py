from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def build_model(model_name: str, params: Dict[str, Any] | None = None):
    """
    Build and return a classifier.

    Parameters
    ----------
    model_name : str
        One of: 'logreg', 'svm', 'rf', 'xgboost'
    params : dict
        Model-specific parameters

    Returns
    -------
    model : sklearn-compatible estimator
    """

    params = params or {}

    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            **params
        )

    if model_name == "svm":
        # probability=True needed for ROC AUC
        return SVC(
            probability=True,
            **params
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_jobs=-1,
            **params
        )

    if model_name == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            eval_metric="mlogloss",
            use_label_encoder=False,
            **params
        )

    raise ValueError(f"Unknown model name: {model_name}")
