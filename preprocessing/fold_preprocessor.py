import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def create_folds(dataset_path, target_column, excluded_target_columns, n_splits=5, shuffle=True, random_state=42):
    # loads the dataset
    df = pd.read_csv(dataset_path)

    # seperates features and target
    X = df.drop(columns=excluded_target_columns)
    y = df[target_column]

    # verify shapes
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Features:", X.columns.tolist())
    print("Target:", target_column)
    print("Target Preview")
    print(y.head())

    # create KFold
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = []

    for fold, (train_index, validate_index) in enumerate(kf.split(X), start=1):

        # split data
        X_train = X.iloc[train_index]
        X_validate = X.iloc[validate_index]

        y_train = y.iloc[train_index]
        y_validate = y.iloc[validate_index]

        # feature scaling
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_validate_scaled = scaler.transform(X_validate)

        # store fold data
        folds.append({
            "fold": fold,
            "X_train": X_train_scaled,
            "X_validate": X_validate_scaled,
            "X_train_raw": X_train.reset_index(drop=True),
            "X_validate_raw": X_validate.reset_index(drop=True),
            "y_train": y_train.reset_index(drop=True),
            "y_validate": y_validate.reset_index(drop=True),
            "train_indices": train_index,
            "validate_indices": validate_index,
            "scaler": scaler
        })

        print(f"\nFold {fold}")
        print("Training:", X_train_scaled.shape, y_train.shape)
        print("Validation:", X_validate_scaled.shape, y_validate.shape)
        print("Train mean:", X_train_scaled.mean(axis=0))
        print("Train std:", X_train_scaled.std(axis=0))
        print("NUmber of Features:", X_train_scaled.shape[1])

    return folds