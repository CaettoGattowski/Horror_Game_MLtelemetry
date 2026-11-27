import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from preprocessing.adaptive_difficulty_classification_preprocessing import (get_folds as get_adaptive_classification_folds)
from preprocessing.threat_encounter_classification_preprocessing import (get_folds as get_threat_classification_folds)

# CONFIG

TRAINING_SIZES = [250, 500, 1000, 2000, 10000, 20000, 30000]

RANDOM_STATE = 42

MODELS = ["Nearest Neighbor", "Nearest Class Centroid"]

# DATASET / FOLDS

DATASETS = {
    "addaptive_difficulty": get_adaptive_classification_folds,
    "threat_encounter": get_threat_classification_folds
}

# TRAINING SUBSET + PREPROCESSING

def prepare_training_subset(fold_data, training_size, fold, random_state=RANDOM_STATE):

    X_train_raw = fold_data["X_train_raw"]
    X_validate_raw = fold_data["X_validate_raw"]

    y_train = fold_data["y_train"]
    y_validate = fold_data["y_validate"]

    actual_training_size = min(training_size, len(X_train_raw))

    rng = np.random.default_rng(random_state + fold)

    shuffled_indices = rng.permutation(len(X_train_raw))
    selected_indices = shuffled_indices[:actual_training_size]

    X_train_subset = (X_train_raw.iloc[selected_indices].reset_index(drop=True))
    y_train_subset = (y_train.iloc[selected_indices].reset_index(drop=True))

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_validate_scaled = scaler.transform(X_validate_raw)

    return (X_train_scaled, X_validate_scaled, y_train_subset, y_validate.reset_index(drop=True), actual_training_size)

# METRICS

def classification_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

# CLASSIFICATION EVAL

def evaluate_classifiers(dataset_name, folds):

    results = []

    for fold_data in folds:

        fold = fold_data["fold"]

        for training_size in TRAINING_SIZES:

            (X_train, X_validate, y_train, y_validate, actual_training_size) = prepare_training_subset(fold_data, training_size, fold)

            for model_name in MODELS:

                print(
                    f"{dataset_name} | "
                    f"Fold {fold} | "
                    f"Training size {actual_training_size} | "
                    f"Model {model_name}"
                )

                # select classifier
                if model_name == "Nearest Neighbor":

                    model = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)

                elif model_name == "Nearest Class Centroid":

                    model = NearestCentroid(metric="euclidean")

                # training time
                start_time = time.perf_counter()

                model.fit(X_train, y_train)

                training_time = time.perf_counter() - start_time

                # prediction
                y_pred = model.predict(X_validate)

                # evaluation
                metrics = classification_metrics(y_validate, y_pred)

                results.append({
                    "dataset": dataset_name,
                    "task": "classification",
                    "model": model_name,
                    "fold": fold,
                    "training_size": actual_training_size,
                    "training_time_seconds": training_time,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "confusion_matrix": metrics["confusion_matrix"]
                })
    return results          

# RESULTS SUMMARY

def create_summary(results_df):

    metric_columns = [
        "accuracy",
        "macro_f1",
        "training_time_seconds"
    ]

    summary_df = (
        results_df.groupby(
            [
                "dataset",
                "task",
                "model",
                "training_size"
            ],
            dropna=False
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    return summary_df

# MAIN

def run_evaluation():

    all_results = []

    for dataset_name, get_folds in DATASETS.items():

        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        folds = get_folds()

        dataset_results = evaluate_classifiers(
            dataset_name,
            folds
        )

        all_results.extend(dataset_results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Output directory
    output_directory = Path("reports/project_A_SVM")
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_output_path = (
        output_directory / "nearest_neighbor_evaluation_results.csv"
    )

    summary_output_path = (
        output_directory / "nearest_neighbor_evaluation_summary.csv"
    )

    # Save raw results
    results_df.to_csv(
        raw_output_path,
        index=False
    )

    # Save summary
    summary_df = create_summary(results_df)

    summary_df.to_csv(
        summary_output_path,
        index=False
    )

    # Final output
    print("\n" + "=" * 70)
    print("NEAREST NEIGHBOR EVALUATION COMPLETED")
    print("=" * 70)

    print("\nRaw results preview:")
    print(results_df.head(16).to_string(index=False))

    print("\nSummary preview:")
    print(summary_df.head(16).to_string(index=False))

    print(f"\nRaw results saved to: {raw_output_path}")
    print(f"Summary saved to: {summary_output_path}")

    return results_df, summary_df


if __name__ == "__main__":
    run_evaluation()