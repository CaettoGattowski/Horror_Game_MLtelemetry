import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)

from models.svmc import SVCmodel
from models.svmr import SVRmodel

from preprocessing.adaptive_difficulty_regression_preprocessing import (get_folds as get_adaptive_regression_folds)
from preprocessing.adaptive_difficulty_classification_preprocessing import (get_folds as get_adaptive_classification_folds)
from preprocessing.threat_encounter_regression_preprocessing import (get_folds as get_threat_regression_folds)
from preprocessing.threat_encounter_classification_preprocessing  import (get_folds as get_threat_classification_folds)

# CONFIG
# controlled training size

TRAINING_SIZES = [250, 500, 1000, 2000, 10000, 200000, 300000]
RANDOM_STATE=42

KERNELS = ["linear", "polynomial", "rbf", "combined"]
SVM_PARAMETERS = {
    "C": 1.0,
    "gamma": 1.0,
    "degree": 3,
    "coef0": 0.0,
    "alpha": 0.5,
    "tol":1e-3,
    "max_iter": 5000
}

SVR_PARAMETERS = {
    "C": 1.0,
    "gamma": 1.0,
    "degree": 3,
    "coef0": 0.0,
    "alpha": 0.5,
    "epsilon": 0.1,
    "tol":1e-3,
    "max_iter": 5000
}

# DATASETS / FOLDS

DATASETS = {
    "adaptive_difficulty": {
        "classification": get_adaptive_classification_folds,
        "regression": get_adaptive_regression_folds
    },
    "threat_encounter": {
        "classification": get_threat_classification_folds,
        "regression": get_threat_regression_folds
    }
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

    X_train_subset = X_train_raw.iloc[selected_indices].reset_index(drop=True)
    y_train_subset = y_train.iloc[selected_indices].reset_index(drop=True)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_validate_scaled = scaler.transform(X_validate_raw)

    return (
        X_train_scaled,
        X_validate_scaled,
        y_train_subset,
        y_validate.reset_index(drop=True),
        actual_training_size,
    )

# METRICS

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5
    }

# CLASSIFICATION EVALUATION

def evaluate_svm(dataset_name, folds):

    results = []

    for fold_data in folds:

        fold = fold_data["fold"]

        for training_size in TRAINING_SIZES:

            (X_train, X_validate, y_train, y_validate, actual_training_size) = prepare_training_subset(fold_data, training_size, fold)

            for kernel in KERNELS:

                print(f"{dataset_name} | Classification |" 
                f"Fold {fold} | Training size {actual_training_size} |" 
                f"Kernel {kernel}")

                model = SVCmodel(kernel=kernel, **SVM_PARAMETERS)

                start_time = time.perf_counter()
                model.fit(X_train, y_train)
                training_time = time.perf_counter() - start_time

                y_pred = model.predict(X_validate)

                metrics = classification_metrics(y_validate, y_pred)

                results.append({
                    "dataset": dataset_name,
                    "task": "classification",
                    "model": "SVM",
                    "kernel": kernel,
                    "fold": fold,
                    "training_size": actual_training_size,
                    "training_time_seconds": training_time,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"])
                })

    return results

# REGRESSION EVALUATION

def evaluate_svr(dataset_name, folds):

    results = []

    for fold_data in folds:

        fold = fold_data["fold"]

        for training_size in TRAINING_SIZES:

            (X_train, X_validate, y_train, y_validate, actual_training_size) = prepare_training_subset(fold_data, training_size, fold)

            for kernel in KERNELS:

                print(
                    f"{dataset_name} | Regression |"
                    f"Fold {fold} | Training size {actual_training_size} |"
                    f"Kernel {kernel}")

                model = SVRmodel(kernel=kernel,**SVR_PARAMETERS)

                start_time = time.perf_counter()
                model.fit(X_train, y_train)
                training_time = time.perf_counter() - start_time

                y_pred = model.predict(X_validate)

                metrics = regression_metrics(y_validate, y_pred)

                results.append({
                    "dataset": dataset_name,
                    "task": "regression",
                    "model": "SVR",
                    "kernel": kernel,
                    "fold": fold,
                    "training_size": actual_training_size,
                    "training_time_seconds": training_time,
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"]
                })

    return results

# RESULTS SUMMARY

def create_summary(results_df):

    metric_columns = [
        "accuracy",
        "macro_f1",
        "r2",
        "mae",
        "rmse",
        "training_time_seconds"
    ]

    available_metrics = [
        metric for metric in metric_columns
        if metric in results_df.columns
    ]

    summary_df = (
        results_df.groupby([
            "dataset",
            "task",
            "model",
            "kernel",
            "training_size"
        ],
        dropna=False)[available_metrics].agg(["mean", "std"]).reset_index())

    return summary_df
# MAIN

def run_evaluation():

    all_results = []

    for dataset_name, tasks in DATASETS.items():

        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)

        # Classification
        classification_folds = tasks["classification"]()

        classification_results = evaluate_svm(dataset_name, classification_folds)

        all_results.extend(classification_results)

        # Regression
        regression_folds = tasks["regression"]()

        regression_results = evaluate_svr(dataset_name, regression_folds)

        all_results.extend(regression_results)

    # convert results to dataframe
    results_df = pd.DataFrame(all_results)

    
    # output paths
    output_directory = Path("reports/project_A_SVM")
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_output_path = output_directory / "svm_evaluation_results.csv"

    summary_output_path = output_directory / "svm_evaluation_summary.csv"

    # save raw results
    results_df.to_csv(raw_output_path, index=False)

    # save summary
    summary_df = create_summary(results_df)

    # final output
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)

    print("\nRaw results preview:")
    print(results_df.head(16).to_string(index=False))

    print("\nSummary preview:")
    print(summary_df.head(16).to_string(index=False))

    print(f"\nRaw results saved to: {raw_output_path}")
    print(f"Summary saved to: {summary_output_path}")

    return results_df, summary_df


if __name__ == "__main__":
    run_evaluation() 