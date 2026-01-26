from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


# =========================
# Utilities
# =========================

def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Load a results CSV produced by experiments.py."""
    return pd.read_csv(csv_path)


def load_folds(pkl_path: str | Path) -> pd.DataFrame:
    """Load fold-level results saved by experiments.py."""
    return pd.read_pickle(pkl_path)


def filter_results(
    df: pd.DataFrame,
    datasets: Optional[Iterable[str]] = None,
    scenarios: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Filter experiment results by dataset, scenario, model."""
    out = df.copy()
    if datasets is not None:
        out = out[out["dataset"].isin(datasets)]
    if scenarios is not None:
        out = out[out["scenario"].isin(scenarios)]
    if models is not None:
        out = out[out["model"].isin(models)]
    return out


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


# =========================
# Confusion matrices
# =========================

def plot_confusion_matrix_from_cm(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    normalize: bool = False,
    save_path: Optional[str | Path] = None,
):
    """
    Plot a confusion matrix from a precomputed matrix.
    """
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =========================
# ROC curves (binary only)
# =========================

def plot_roc_curve_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    save_path: Optional[str | Path] = None,
):
    """
    Plot ROC curve for binary classification.
    """
    if y_score is None or np.ndim(y_score) != 1:
        raise ValueError(
            "plot_roc_curve_binary expects 1D y_score for binary classification."
        )

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =========================
# Summary tables
# =========================

def save_summary_table(
    df: pd.DataFrame,
    metrics: List[str],
    save_path: str | Path,
):
    """
    Save a compact summary table with selected metrics.
    """
    cols = ["dataset", "scenario", "model"]
    for m in metrics:
        cols.append(f"{m}_mean")
        cols.append(f"{m}_std")

    summary = df[cols].copy()
    summary.to_csv(save_path, index=False)
