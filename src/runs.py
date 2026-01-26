from pathlib import Path

from src.datasets import load_datasets, load_labels
from src.experiments import ModelSpec, run_experiment_suite


def main():
    # -------------------------
    # 1) Paths
    # -------------------------
    dataset_paths = {
        "pathway2235": r"C:\Users\ccz\Desktop\Project with Public Data\Tables for ML\Galaxy2235-[PICRUSt2 Full pipeline on data 2226 and data 2231_ Pathway abundances].tabular.txt",
    }
    labels_path = r"C:\Users\ccz\Desktop\Project with Public Data\labels.csv"

    # -------------------------
    # 2) Load data
    # -------------------------
    datasets = load_datasets(dataset_paths)
    y = load_labels(labels_path)

    # -------------------------
    # 3) Model specification
    # -------------------------
    model_specs = [
        ModelSpec(
            name="rf",
            mode="grid",
            param_grid={
                "n_estimators": [200, 500],
                "max_depth": [None, 10],
                "max_features": ["sqrt", "log2"],
            },
        )
    ]

    # -------------------------
    # 4) Output directory
    # -------------------------
    output_dir = Path("outputs") / "debug_rf_2235"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 5) Run experiment (nested CV ON)
    # -------------------------
    run_experiment_suite(
        datasets=datasets,
        y_global=y,
        dataset_names=["pathway2235"],
        model_specs=model_specs,
        output_dir=output_dir,
        preprocess_cfg={
            "use_clr": True,
            "use_scaler": True,
            "k_features": 50,
        },
        use_nested_cv=True,
    )

    print(f"Experiment finished. Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
