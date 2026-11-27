from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (PROJECT_ROOT / "reports" / "project_A_SVM" / "tuned_svm_evaluation_results.csv") # Baseline or Tuned

OUTPUT_DIR = (PROJECT_ROOT / "reports" / "project_A_SVM" / "svm_learning_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CONFIG

KERNELS = ["linear", "polynomial", "rbf", "combined"]

CLASSIFICATION_METRICS = ["accuracy", "macro_f1"]

REGRESSION_METRICS = ["r2", "mae", "rmse"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "r2": "R^2",
    "mae": "MAE",
    "rmse": "RMSE"
}

DATASET_LABELS = {
    "adaptive_difficulty": "Adaptive Difficulty",
    "threat_encounter": "Threat Encounter"
}

def load_results():

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)

    required_columns = [
        "dataset",
        "task",
        "model",
        "kernel",
        "fold",
        "training_size",
        "training_time_seconds"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns in results CSV: {missing_columns}")

    return df

def create_aggregated_summary(df):

    metrics = (CLASSIFICATION_METRICS + REGRESSION_METRICS + ["training_time_seconds"])

    grouping_columns = ["dataset", "task", "model", "kernel", "training_size"]

    summary = (df.groupby(grouping_columns)[metrics].agg(["mean", "std"]).reset_index())

    # flatten multi level columns
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    summary.to_csv(OUTPUT_DIR / "learning_curves_summary.csv", index=False)

    return summary

# PLOT LEARNING CURVES

def plot_metric(summary, dataset, task, metric):

    fig, ax = plt.subplots(figsize=(9, 6))

    dataset_data = summary[(summary["dataset"] == dataset) & (summary["task"] == task)]

    for kernel in KERNELS:

        kernel_data = dataset_data[
            dataset_data["kernel"] == kernel
        ].sort_values("training_size")

        if kernel_data.empty:
            continue
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        if mean_column not in kernel_data.columns:
            continue

        means = kernel_data[mean_column]

        stds = (
            kernel_data[std_column].fillna(0)
            if std_column in kernel_data.columns
            else 0
        )

        ax.errorbar(
            kernel_data["training_size"],
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=2,
            label=kernel.upper()
        )

    ax.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)} "
        f"- {task.capitalize()} - {METRIC_LABELS[metric]}"
    )

    ax.set_xlabel("Training samples")
    ax.set_ylabel(METRIC_LABELS[metric])

    ax.set_xticks(
        sorted(dataset_data["training_size"].unique())
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename = f"{dataset}_{task}_{metric}.png"

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# PLOT TRAINING TIME

def plot_training_time(summary, dataset, task):

    fig, ax = plt.subplots(figsize=(9, 6))

    dataset_data = summary[
        (summary["dataset"] == dataset)
        & (summary["task"] == task)
    ]

    for kernel in KERNELS:

        kernel_data = dataset_data[
            dataset_data["kernel"] == kernel
        ].sort_values("training_size")

        if kernel_data.empty:
            continue

        means = kernel_data["training_time_seconds_mean"]

        stds = kernel_data["training_time_seconds_std"].fillna(0)

        ax.errorbar(
            kernel_data["training_size"],
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=2,
            label=kernel.upper(),
        )

    ax.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)} "
        f"- {task.capitalize()} - Training Time"
    )

    ax.set_xlabel("Training samples")
    ax.set_ylabel("Training time (seconds)")

    ax.set_xticks(
        sorted(dataset_data["training_size"].unique())
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename = f"{dataset}_{task}_training_time.png"

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# MAIN

def main():

    print("=" * 70)
    print("GENERATING LEARNING CURVES")
    print("=" * 70)

    results = load_results()

    summary = create_aggregated_summary(results)

    datasets = results["dataset"].unique()
    tasks = results["task"].unique()

    for dataset in datasets:

        for task in tasks:

            if task == "classification":
                metrics = CLASSIFICATION_METRICS

            elif task == "regression":
                metrics = REGRESSION_METRICS
            
            else:
                continue
            
            for metric in metrics:

                plot_metric(
                    summary=summary,
                    dataset=dataset,
                    task=task,
                    metric=metric
                )
            plot_training_time(
                summary=summary,
                dataset=dataset,
                task=task
            )
    
    print("\nLearning curves generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary CSV: {OUTPUT_DIR / 'learning_curves_summary.csv'}")

    print("=" * 70)

if __name__ == "__main__":
    main()