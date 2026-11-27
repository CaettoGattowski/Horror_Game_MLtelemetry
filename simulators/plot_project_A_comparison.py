
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PATHS 


BASE_DIR = Path("reports/project_A_SVM")

BASELINE_FILE = BASE_DIR / "baseline_svm_evaluation_summary.csv"
TUNED_FILE = BASE_DIR / "tuned_svm_evaluation_summary.csv"
NN_FILE = BASE_DIR / "nearest_neighbor_evaluation_summary.csv"

OUTPUT_FILE = BASE_DIR / "project_A_final_comparison.png"

# LOAD SUMMARIES

def load_summary(path):
    df = pd.read_csv(path)

    df["dataset"] = (
        df["dataset"]
        .astype(str)
        .str.lower()
        .str.replace("addaptive", "adaptive", regex=False)
    )

    df["task"] = df["task"].astype(str).str.lower()
    df["model"] = df["model"].astype(str)

    if "kernel" in df.columns:
        df["kernel"] = df["kernel"].astype(str).str.lower()

    df["training_size"] = pd.to_numeric(
        df["training_size"], errors="coerce"
    )

    # Convert metric columns to numeric
    metric_columns = [
        "accuracy", "accuracy.1",
        "macro_f1", "macro_f1.1",
        "r2", "r2.1",
        "mae", "mae.1",
        "rmse", "rmse.1",
        "training_time_seconds",
        "training_time_seconds.1"
    ]

    for col in metric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


baseline_df = load_summary(BASELINE_FILE)
tuned_df = load_summary(TUNED_FILE)
nn_df = load_summary(NN_FILE)

def mean_column(df, metric):
    return metric


def best_model(df, dataset, task, metric, model_filter=None):
    subset = df[(df["dataset"] == dataset) & (df["task"] == task)].copy()

    if model_filter is not None:
        subset = subset[subset["model"].str.lower().isin([m.lower() for m in model_filter])]

    if subset.empty:
        return None

    # Largest available training size for this dataset/task
    max_size = subset["training_size"].max()
    subset = subset[subset["training_size"] == max_size]

    metric_col = mean_column(subset, metric)

    if metric_col not in subset.columns:
        return None

    subset = subset.dropna(subset=[metric_col])

    if subset.empty:
        return None

    # For R2, Accuracy and F1, larger is better
    best_row = subset.loc[subset[metric_col].idxmax()]

    return best_row


def get_score(row, metric):
    if row is None:
        return np.nan
    return row[metric]


def get_best_label(row, default_label):
    if row is None:
        return default_label

    kernel = row.get("kernel", None)

    if pd.isna(kernel):
        return default_label

    return f"{default_label} ({kernel})"

# CLASSIFICATION DATA

def classification_comparison(dataset, dataset_title):

    # Best baseline SVM per metric
    baseline_acc = best_model(baseline_df, dataset, "classification", "accuracy", model_filter=["SVM"])

    baseline_f1 = best_model(baseline_df, dataset, "classification", "macro_f1", model_filter=["SVM"])

    # Best tuned SVM per metric
    tuned_acc = best_model(tuned_df, dataset, "classification", "accuracy", model_filter=["SVM"])

    tuned_f1 = best_model(tuned_df, dataset, "classification", "macro_f1", model_filter=["SVM"])

    # NN and NCC
    ncc_acc = best_model(nn_df, dataset, "classification", "accuracy", model_filter=["Nearest Class Centroid"])

    ncc_f1 = best_model(nn_df, dataset, "classification", "macro_f1", model_filter=["Nearest Class Centroid"])

    nn_acc = best_model(nn_df, dataset, "classification", "accuracy", model_filter=["Nearest Neighbor"])

    nn_f1 = best_model(nn_df, dataset, "classification", "macro_f1", model_filter=["Nearest Neighbor"])

    results = {
        "Baseline SVM": [get_score(baseline_acc, "accuracy") * 100, get_score(baseline_f1, "macro_f1") * 100],
        "Tuned SVM": [get_score(tuned_acc, "accuracy") * 100, get_score(tuned_f1, "macro_f1") * 100],
        "NCC": [get_score(ncc_acc, "accuracy") * 100, get_score(ncc_f1, "macro_f1") * 100],
        "1-NN": [get_score(nn_acc, "accuracy") * 100, get_score(nn_f1, "macro_f1") * 100]
    }

    labels = {
        "Baseline SVM": get_best_label(baseline_acc, "Baseline"),
        "Tuned SVM": get_best_label(tuned_acc, "Tuned"),
        "NCC": "Nearest Class Centroid",
        "1-NN": "Nearest Neighbor",
    }

    return results, labels

# REGRESSION DATA

def regression_comparison(dataset):

    baseline = best_model(baseline_df, dataset, "regression", "r2", model_filter=["SVR"])

    tuned = best_model(tuned_df, dataset, "regression", "r2", model_filter=["SVR"])

    results = {"Baseline SVR": get_score(baseline, "r2"), "Tuned SVR": get_score(tuned, "r2")}

    labels = {"Baseline SVR": get_best_label(baseline, "Baseline"), "Tuned SVR": get_best_label(tuned, "Tuned")}

    return results, labels

# PLOT

def plot_classification(ax, results, title):

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.35

    accuracy = [results[m][0] for m in models]
    macro_f1 = [results[m][1] for m in models]

    bar1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy")
    bar2 = ax.bar(x + width / 2, macro_f1, width, label="Macro F1")

    # Add percentage labels above each classification bar
    for bars in (bar1, bar2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)


def plot_regression(ax, results, title):

    models = list(results.keys())
    values = [float(results[m]) for m in models]

    bars = ax.bar(models, values)

    ax.set_ylabel("$R^2$")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    min_value = min(0, min(values))
    max_value = max(values)

    ax.set_ylim(min_value - 0.05, max_value + 0.10)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

# FIGURE

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Adaptive Difficulty
adaptive_cls, _ = classification_comparison("adaptive_difficulty", "Adaptive Difficulty")

adaptive_reg, _ = regression_comparison("adaptive_difficulty")

plot_classification(axes[0, 0], adaptive_cls, "Adaptive Difficulty - Classification")

plot_regression(axes[1, 0], adaptive_reg, "Adaptive Difficulty - Regression")

# Threat Encounter
threat_cls, _ = classification_comparison("threat_encounter", "Threat Encounter")

threat_reg, _ = regression_comparison("threat_encounter")

plot_classification(axes[0, 1], threat_cls,"Threat Encounter - Classification")

plot_regression(axes[1, 1], threat_reg,"Threat Encounter - Regression")


fig.suptitle("Project A - Baseline, Tuned and Nearest Neighbor Models", fontsize=16)

plt.tight_layout()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

plt.show()

print(f"Figure saved to: {OUTPUT_FILE}")
