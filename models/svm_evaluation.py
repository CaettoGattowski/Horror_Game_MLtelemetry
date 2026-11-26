import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)

from models.svmc import SVCmodel
from models.svmr import SVRmodel

from preprocessing.adaptive_difficulty_regression_preprocessing import (get_folds as get_adaptive_regression_folds)
from preprocessing.adaptive_difficulty_classification_preprocessing import (get_folds as get_adaptive_classification_folds)
from preprocessing.threat_encounter_regression_preprocessing import (get_folds as get_threat_regression_folds)
from preprocessing.threat_encounter_classification_preprocessing  import (get_folds as get_threat_classification_folds)

# CONFIG
# controlled training size

TRAINING_SIZE_LIMIT = 500
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

# PREPROCESSING FOLDS

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


def limit_training_data(X_train, y_train, training_size_limit=TRAINING_SIZE_LIMIT, random_state=RANDOM_STATE):
    if training_size_limit is None or len(X_train) <= training_size_limit:
        return X_train, y_train

    rng = np.random.default_rng(random_state)

    selected_indices = rng.choice(len(X_train),size=training_size_limit, replace=False)

    X_train_limited = X_train[selected_indices]
    y_train_limited = y_train.iloc[selected_indices].reset_index(drop=True)

    return X_train_limited, y_train_limited

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

        X_train = fold_data["X_train"]
        X_validate = fold_data["X_validate"]

        y_train = fold_data["y_train"]
        y_validate = fold_data["y_validate"]

        X_train, y_train = limit_training_data(X_train, y_train, random_state=RANDOM_STATE + fold)

        for kernel in KERNELS:

            print(f"{dataset_name} | Classification |" 
            f"Fold {fold} | Kernel {kernel}")

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
                "training_size": len(X_train),
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

        X_train = fold_data["X_train"]
        X_validate = fold_data["X_validate"]

        y_train = fold_data["y_train"]
        y_validate = fold_data["y_validate"]

        X_train, y_train = limit_training_data(X_train, y_train, random_state=RANDOM_STATE + fold)

        for kernel in KERNELS:

            print(
                f"{dataset_name} | Regression |"
                f"Fold {fold} | Kernel {kernel}")

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
                "training_size": len(X_train),
                "training_time_seconds": training_time,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"]
            })

    return results

# MAIN

def run_evaluation():

    all_results = []

    for dataset_name, tasks in DATASETS.items():

        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

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

    
    # save results
    output_path = "reports/project_A_SVM/svm_evaluation_results.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print("\nResults preview:")
    print(results_df.head(10).to_string(index=False))

    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    run_evaluation() 