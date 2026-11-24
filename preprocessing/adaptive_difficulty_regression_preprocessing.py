import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# loads the dataset
df = pd.read_csv("datasets/adaptive_difficulty.csv")

# seperates features and targets
X = df.drop(columns=["recommended_difficulty", "difficulty_category"])
y = df["recommended_difficulty"]

# verifies shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

# verifies selected features
print(X.columns.tolist())

# verifies target
print(y.head())

# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, validate_index) in enumerate(kf.split(X), start=1):
    X_train, X_validate = X.iloc[train_index], X.iloc[validate_index]
    y_train, y_validate = y.iloc[train_index], y.iloc[validate_index]

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)

    # verifies scaling
    print(f"Fold {fold}")
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_validate.shape, y_validate.shape)
    print("Train mean:", X_train_scaled.mean(axis=0))
    print("Train std:", X_train_scaled.std(axis=0))
    print("Number of features:", X_train_scaled.shape[1])
    print("Number of std values:", len(X_train_scaled.std(axis=0)))


