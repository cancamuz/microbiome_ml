from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def merge_classes(y: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """
    Merge/relabel classes using a mapping {old_label: new_label}.
    Labels not in mapping stay unchanged.
    """
    y = y.astype(str).copy()
    return y.map(lambda v: mapping.get(v, v))


def drop_classes(y: pd.Series, drop: Iterable[str]) -> pd.Series:
    """
    Drop samples whose label is in drop list.
    Returns filtered y (caller should filter X with y.index).
    """
    y = y.astype(str).copy()
    drop_set = set(map(str, drop))
    return y[~y.isin(drop_set)]


def keep_only_classes(y: pd.Series, keep: Iterable[str]) -> pd.Series:
    """
    Keep only samples whose label is in keep list.
    Returns filtered y (caller should filter X with y.index).
    """
    y = y.astype(str).copy()
    keep_set = set(map(str, keep))
    return y[y.isin(keep_set)]


def make_binary(
    y: pd.Series,
    positive: Iterable[str],
    negative: Iterable[str],
    pos_name: str = "positive",
    neg_name: str = "negative",
) -> pd.Series:
    """
    Convert multiclass labels to binary labels.

    Any label not in positive or negative is dropped.
    Returns filtered y with values {pos_name, neg_name}.
    """
    y = y.astype(str).copy()
    pos = set(map(str, positive))
    neg = set(map(str, negative))

    out = pd.Series(index=y.index, dtype="object")
    out[y.isin(pos)] = pos_name
    out[y.isin(neg)] = neg_name
    return out.dropna()


def encode_labels(y: pd.Series, classes=None):
    le = LabelEncoder()

    if classes is not None:
        le.fit([str(c) for c in classes])
        unknown = sorted(set(y.unique()) - set(le.classes_))
        if unknown:
            raise ValueError(f"Found labels not in 'classes': {unknown}")
        y_enc = pd.Series(
            le.transform(y.astype(str)),
            index=y.index,          # ← THIS IS THE FIX
            name=y.name,
        )
        return y_enc, le

    y_enc = pd.Series(
        le.fit_transform(y.astype(str)),
        index=y.index,              # ← AND HERE
        name=y.name,
    )
    return y_enc, le


def class_counts(y: pd.Series) -> pd.Series:
    """Convenience: class counts."""
    return y.value_counts()
