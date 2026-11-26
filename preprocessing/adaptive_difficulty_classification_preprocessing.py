from preprocessing.fold_preprocessor import create_folds


def get_folds():
    return create_folds(
        dataset_path="datasets/adaptive_difficulty.csv",
        target_column="difficulty_category",
        excluded_target_columns=[
            "recommended_difficulty",
            "difficulty_category"
        ]
    )


if __name__ == "__main__":
    folds = get_folds()
    print(f"\nCreated {len(folds)} folds for adaptive difficulty classification.")
