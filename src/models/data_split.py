from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def split_training_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Split labeled records into train/validation sets."""
    labeled = df.dropna(subset=[target_column]).copy()
    if labeled.empty:
        raise ValueError("No labeled records available after target creation.")

    available_features = [column for column in feature_columns if column in labeled.columns]
    if not available_features:
        raise ValueError("No configured feature columns were found in the dataset.")

    X = labeled[available_features]
    y = labeled[target_column].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]

    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        X_train[numeric_columns] = numeric_imputer.fit_transform(X_train[numeric_columns])
        X_val[numeric_columns] = numeric_imputer.transform(X_val[numeric_columns])

    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_columns] = categorical_imputer.fit_transform(X_train[categorical_columns])
        X_val[categorical_columns] = categorical_imputer.transform(X_val[categorical_columns])

    return {
        "features": available_features,
        "train_shape": (int(X_train.shape[0]), int(X_train.shape[1])),
        "test_shape": (int(X_val.shape[0]), int(X_val.shape[1])),
        "train_target_distribution": y_train.value_counts().to_dict(),
        "test_target_distribution": y_val.value_counts().to_dict(),
    }