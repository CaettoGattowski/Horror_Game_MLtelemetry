import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
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

TRAINING_SIZES = [250, 500, 1000, 2000, 10000, 20000, 30000]
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

# HYPERPARAMETER TUNING
# SOS
TUNE = False # false = baseline fixed, True = tuned

TUNING_SIZE = {
    "adaptive_difficulty": 10000,
    "threat_encounter": 2400
}

TUNING_FOLDS = 3

PARAMETER_GRID = {
    "linear": [
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0}
    ],

    "polynomial": [
        {"C": 1.0, "gamma": 0.01, "degree": 2, "coef0": 0.0},
        {"C": 1.0, "gamma": 0.1, "degree": 3, "coef0": 1.0},
        {"C": 10.0, "gamma": 0.01, "degree": 3, "coef0": 1.0},
    ],
    "rbf": [
        {"C": 1.0, "gamma": 0.01},
        {"C": 10.0, "gamma": 0.01},
        {"C": 10.0, "gamma": 0.1},
    ],
    "combined": [
        {"C": 1.0, "gamma": 0.01, "degree": 2,"coef0": 0.0, "alpha": 0.25},
        {"C": 1.0, "gamma": 0.1, "degree": 3, "coef0": 1.0, "alpha": 0.5},
        {"C": 10.0, "gamma": 0.01, "degree": 3, "coef0": 1.0, "alpha": 0.75},
    ]
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

# HYPERPARAMETER TUNING

def tune_parameters(X_raw, y, task, dataset_name, fold):

    tuning_size = min(TUNING_SIZE[dataset_name], len(X_raw))

    rng = np.random.default_rng(RANDOM_STATE + fold)
    indices = rng.permutation(len(X_raw))[:tuning_size]

    X_raw = X_raw.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)

    inner_cv = KFold(n_splits = TUNING_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_parameters = {}

    for kernel in KERNELS:

        best_score = -np.inf
        best_params = None

        for candidate in PARAMETER_GRID[kernel]:

            fold_scores = []

            for train_idx, val_idx in inner_cv.split(X_raw):

                X_inner_train = X_raw.iloc[train_idx]
                X_inner_val = X_raw.iloc[val_idx]

                y_inner_train = y.iloc[train_idx]
                y_inner_val = y.iloc[val_idx]

                scaler = StandardScaler()

                X_inner_train_scaled = scaler.fit_transform(X_inner_train)
                X_inner_val_scaled = scaler.transform(X_inner_val)

                # default params + candidate params
                if task == "classification":
                    params = SVM_PARAMETERS.copy()
                else:
                    params = SVR_PARAMETERS.copy()
                
                params.update(candidate)

                model = (
                    SVCmodel(kernel=kernel, **params)
                    if task == "classification"
                    else SVRmodel(kernel=kernel, **params))

                model.fit(X_inner_train_scaled, y_inner_train)

                predictions = model.predict(X_inner_val_scaled)

                if task == "classification":
                    score = f1_score(y_inner_val, predictions, average="macro", zero_division=0)
                else:
                    score = r2_score(y_inner_val, predictions)

                fold_scores.append(score)
            
            mean_score = np.mean(fold_scores)

            print(
                f"Tuning | {dataset_name} | {task} | "
                f"Fold {fold} | {kernel} | "
                f"{candidate} | Score: {mean_score:.4f}"
            )

            if mean_score > best_score:
                best_score = mean_score
                best_params = candidate.copy()

        best_parameters[kernel] = {**best_params, "tuning_score": best_score}

        print(f"\nBEST {kernel}: {best_parameters[kernel]}\n")

    return best_parameters

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

        if TUNE:
            tuned_parameters = tune_parameters(
                X_raw=fold_data["X_train_raw"],
                y=fold_data["y_train"],
                task="classification",
                dataset_name=dataset_name,
                fold=fold
            )

        sizes_to_run = (
            [TUNING_SIZE[dataset_name]]
            if TUNE
            else TRAINING_SIZES
            )

        for training_size in TRAINING_SIZES:

            (X_train, X_validate, y_train, y_validate, actual_training_size) = prepare_training_subset(fold_data, training_size, fold)

            for kernel in KERNELS:

                print(f"{dataset_name} | Classification |" 
                f"Fold {fold} | Training size {actual_training_size} |" 
                f"Kernel {kernel}")

                if TUNE:
                    params = SVM_PARAMETERS.copy()
                    params.update(tuned_parameters[kernel])

                    
                    params.pop("tuning_score", None)

                    model = SVCmodel(kernel=kernel, **params)
                else:
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
                    "tuned": TUNE,
                    "hyperparameters": json.dumps(params if TUNE else SVM_PARAMETERS),
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

        if TUNE:
            tuned_parameters = tune_parameters(
                X_raw=fold_data["X_train_raw"],
                y=fold_data["y_train"],
                task="regression",
                dataset_name=dataset_name,
                fold=fold
            )

        sizes_to_run = (
            [TUNING_SIZE[dataset_name]]
            if TUNE
            else TRAINING_SIZES
            )

        for training_size in TRAINING_SIZES:

            (X_train, X_validate, y_train, y_validate, actual_training_size) = prepare_training_subset(fold_data, training_size, fold)

            for kernel in KERNELS:

                print(
                    f"{dataset_name} | Regression |"
                    f"Fold {fold} | Training size {actual_training_size} |"
                    f"Kernel {kernel}")

                if TUNE:
                    params = SVR_PARAMETERS.copy()
                    params.update(tuned_parameters[kernel])

                    params.pop("tuning_score", None)

                    model = SVRmodel(kernel=kernel, **params)
                else:
                    model = SVRmodel(kernel=kernel, **SVR_PARAMETERS)

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
                    "tuned": TUNE,
                    "hyperparameters": json.dumps(params if TUNE else SVR_PARAMETERS),
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
            "training_size",
            "tuned"
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
    summary_df.to_csv(summary_output_path, index=False)

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