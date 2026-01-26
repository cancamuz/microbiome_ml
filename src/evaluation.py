from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =========================
# Metrics utilities
# =========================

def _get_probabilities(model, X):
    """Return probabilities or decision scores if available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            return scores
        return scores
    return None


def compute_metrics(
    y_true,
    y_pred,
    y_proba=None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    Works for binary and multiclass.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average=average
                )
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics


# =========================
# Core CV engines
# =========================

def run_nested_cv_experiment(
    pipeline,
    X,
    y,
    param_grid: Optional[Dict[str, List[Any]]],
    experiment_metadata: Dict[str, Any],
    n_splits_outer: int = 5,
    n_splits_inner: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run nested cross-validation.
    Outer CV: unbiased evaluation
    Inner CV: hyperparameter tuning (only if param_grid is provided)
    """

    outer_cv = StratifiedKFold(
        n_splits=n_splits_outer,
        shuffle=True,
        random_state=random_state,
    )

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Decide ROC scoring based on number of classes
        is_binary = len(np.unique(y_train)) == 2
        scoring = "roc_auc" if is_binary else "roc_auc_ovr"

        # -------------------------
        # Inner loop (optional)
        # -------------------------
        if param_grid is not None and len(param_grid) > 0:
            inner_cv = StratifiedKFold(
                n_splits=n_splits_inner,
                shuffle=True,
                random_state=random_state,
            )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_model = pipeline
            best_model.fit(X_train, y_train)
            best_params = {}

        # -------------------------
        # Outer test evaluation
        # -------------------------
        y_pred = best_model.predict(X_test)
        y_proba = _get_probabilities(best_model, X_test)

        # Prepare scores for ROC plotting
        y_score = None
        if y_proba is not None:
            if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2:
                # binary → use positive class probability
                if y_proba.shape[1] == 2:
                    y_score = y_proba[:, 1]
                else:
                    # multiclass → keep full matrix
                    y_score = y_proba
            else:
                # decision_function output
                y_score = y_proba

        metrics = compute_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        fold_rows.append({
            **experiment_metadata,
            "fold": fold_idx,
            **metrics,
            "best_params": best_params,
            "confusion_matrix": cm,
            # raw fold data for reports.py
            "y_true": np.asarray(y_test),
            "y_pred": np.asarray(y_pred),
            "y_score": y_score,
        })

    fold_df = pd.DataFrame(fold_rows)

    # -------------------------
    # Summary (mean ± std)
    # -------------------------
    summary = {**experiment_metadata}

    for col in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        if col in fold_df.columns:
            summary[f"{col}_mean"] = fold_df[col].mean()
            summary[f"{col}_std"] = fold_df[col].std()

    summary_df = pd.DataFrame([summary])

    return summary_df, fold_df



def run_cv_experiment(
    pipeline,
    X,
    y,
    param_grid: Optional[Dict[str, List[Any]]],
    experiment_metadata: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    outer_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if param_grid:
            inner_cv = StratifiedKFold(
                n_splits=max(2, n_splits - 1),
                shuffle=True,
                random_state=random_state,
            )

            scoring = "roc_auc" if len(np.unique(y_train)) == 2 else "roc_auc_ovr"

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model = pipeline
            model.fit(X_train, y_train)
            best_params = {}

        y_pred = model.predict(X_test)
        y_proba = _get_probabilities(model, X_test)

        # Store y_score for ROC plotting
        y_score = None
        if y_proba is not None:
            if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2:
                if y_proba.shape[1] == 2:
                    y_score = y_proba[:, 1]
                else:
                    y_score = y_proba
            else:
                y_score = y_proba

        metrics = compute_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        fold_rows.append({
            **experiment_metadata,
            "fold": fold_idx,
            **metrics,
            "best_params": best_params,
            "confusion_matrix": cm,
            "y_true": np.asarray(y_test),
            "y_pred": np.asarray(y_pred),
            "y_score": y_score,
            # optional:
            # "y_proba": y_proba,
        })

    fold_df = pd.DataFrame(fold_rows)

    summary = {**experiment_metadata}
    for col in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
        if col in fold_df:
            summary[f"{col}_mean"] = fold_df[col].mean()
            summary[f"{col}_std"] = fold_df[col].std()

    summary_df = pd.DataFrame([summary])

    return summary_df, fold_df

