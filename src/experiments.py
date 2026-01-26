from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from src.label_classes import merge_classes, drop_classes, make_binary, encode_labels, class_counts
from src.preprocessing import build_preprocessing_pipeline
from src.models import build_model
from src.evaluation import run_cv_experiment, run_nested_cv_experiment


# =========================
# Label scenarios (fixed 3)
# =========================

@dataclass(frozen=True)
class Scenario:
    name: str
    classes: List[str]

    def apply(self, y: pd.Series) -> pd.Series:
        raise NotImplementedError


class Scenario4Classes(Scenario):
    def apply(self, y: pd.Series) -> pd.Series:
        return y.astype(str).copy()


class Scenario3Classes(Scenario):
    def apply(self, y: pd.Series) -> pd.Series:
        y2 = y.astype(str).copy()
        y2 = merge_classes(y2, {"AIP": "irreversible", "SIP": "irreversible"})
        return y2


class ScenarioBinary(Scenario):
    def apply(self, y: pd.Series) -> pd.Series:
        y2 = y.astype(str).copy()
        y2 = merge_classes(y2, {"AIP": "irreversible", "SIP": "irreversible"})
        y2 = drop_classes(y2, ["NP"])
        y2 = make_binary(
            y2,
            positive=["irreversible"],
            negative=["RP"],
            pos_name="irreversible",
            neg_name="reversible",
        )
        return y2


def get_scenarios() -> List[Scenario]:
    return [
        Scenario4Classes(name="4_classes_all", classes=["NP", "RP", "AIP", "SIP"]),
        Scenario3Classes(name="3_classes_np_rp_irrev", classes=["NP", "RP", "irreversible"]),
        ScenarioBinary(name="binary_rev_vs_irrev_drop_np", classes=["reversible", "irreversible"]),
    ]


# =========================
# Model specs
# =========================

@dataclass(frozen=True)
class ModelSpec:
    name: str
    mode: Literal["fixed", "grid"]
    fixed_params: Optional[Dict[str, Any]] = None
    param_grid: Optional[Dict[str, List[Any]]] = None


# =========================
# Experiment runner
# =========================

def run_experiment_suite(
    datasets: Dict[str, pd.DataFrame],
    y_global: pd.Series,
    dataset_names: List[str],
    model_specs: List[ModelSpec],
    output_dir: str | Path,
    preprocess_cfg: Optional[Dict[str, Any]] = None,
    use_nested_cv: bool = True,
) -> Dict[str, pd.DataFrame]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = preprocess_cfg or {
        "use_clr": True,
        "use_scaler": True,
        "k_features": 100,
    }

    preprocess = build_preprocessing_pipeline(**preprocess_cfg)
    scenarios = get_scenarios()

    scenario_results: Dict[str, pd.DataFrame] = {}

    for scenario in scenarios:
        rows = []

        # ----- label handling -----
        y_s = scenario.apply(y_global)
        y_enc, le = encode_labels(y_s, classes=scenario.classes)
        counts = class_counts(y_s)

        for ds_name in dataset_names:
            X = datasets[ds_name]
            X_s = X.loc[y_enc.index]

            for spec in model_specs:
                model = build_model(spec.name, params=(spec.fixed_params or {}))

                pipe = Pipeline([
                    ("preprocess", preprocess),
                    ("model", model),
                ])

                metadata = {
                    "dataset": ds_name,
                    "scenario": scenario.name,
                    "model": spec.name,
                    "mode": spec.mode,
                    "preprocess": preprocess_cfg,
                    "classes": list(le.classes_),
                    "class_counts": counts.to_dict(),
                }

                # ----- choose evaluation engine -----
                if use_nested_cv:
                    param_grid = (
                        _prefix_grid(spec.param_grid)
                        if spec.mode == "grid"
                        else None
                    )

                    if spec.mode == "grid" and not spec.param_grid:
                        raise ValueError(
                            f"ModelSpec '{spec.name}' is mode='grid' but param_grid is empty."
                        )

                    summary_df, fold_df = run_nested_cv_experiment(
                        pipeline=pipe,
                        X=X_s,
                        y=y_enc,
                        param_grid=param_grid,
                        experiment_metadata=metadata,
                    )

                else:
                    param_grid = (
                        _prefix_grid(spec.param_grid)
                        if spec.mode == "grid"
                        else None
                    )

                    summary_df, fold_df = run_cv_experiment(
                        pipeline=pipe,
                        X=X_s,
                        y=y_enc,
                        param_grid=param_grid,
                        experiment_metadata=metadata,
                    )

                # ----- save fold-level data for reports.py -----
                fold_path = (
                    output_dir
                    / f"folds__{scenario.name}__{ds_name}__{spec.name}.pkl"
                )
                fold_df.to_pickle(fold_path)

                rows.append(summary_df)

        results_df = pd.concat(rows, ignore_index=True)
        results_df.to_csv(
            output_dir / f"results__{scenario.name}.csv",
            index=False,
        )

        scenario_results[scenario.name] = results_df

    return scenario_results


# =========================
# Helpers
# =========================

def _prefix_grid(param_grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    out = {}
    for k, v in param_grid.items():
        if "__" in k:
            out[k] = v
        else:
            out[f"model__{k}"] = v
    return out
