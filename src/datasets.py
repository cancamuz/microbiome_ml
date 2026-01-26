from pathlib import Path
from typing import Dict

import pandas as pd


def load_picrust_tabular(file_path: str | Path) -> pd.DataFrame:
    """
    Load a PICRUSt2 Galaxy tabular file.
    Returns a DataFrame with:
      - rows = samples
      - columns = features
    """
    file_path = Path(file_path)
    df = pd.read_csv(
        file_path,
        sep="\t",
        index_col=0,
        comment="#"
    ).T

    df.index = df.index.astype(str)
    return df


def load_datasets(dataset_paths: Dict[str, str | Path]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple datasets.
    Returns: {dataset_name: DataFrame}
    """
    return {
        name: load_picrust_tabular(path)
        for name, path in dataset_paths.items()
    }


def load_labels(labels_path: str | Path) -> pd.Series:
    """
    Load labels.csv with columns: sample, label (no header).
    Returns a Series indexed by sample.
    """
    labels_path = Path(labels_path)
    labels_df = pd.read_csv(
        labels_path,
        header=None,
        names=["sample", "label"]
    )
    labels_df["sample"] = labels_df["sample"].astype(str)
    labels_df = labels_df.set_index("sample")
    return labels_df["label"].astype(str)
