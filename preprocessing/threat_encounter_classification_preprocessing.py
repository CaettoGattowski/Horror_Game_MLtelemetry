from preprocessing.fold_preprocessor import create_folds


def get_folds():
    return create_folds(
        dataset_path="datasets/threat_encounter.csv",
        target_column="encounter_category",
        excluded_target_columns=[
            "time_to_next_threat_event",
            "encounter_category"
        ]
    )


if __name__ == "__main__":
    folds = get_folds()
    print(f"\nCreated {len(folds)} folds for threat encounter classification.")
