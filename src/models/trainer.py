"""Training pipeline for loan approval models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pickle
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.features.feature_builder import build_features
from src.models.evaluator import evaluate_model, select_decision_threshold

from sklearn.pipeline import Pipeline as SklearnPipeline


def _make_classifier(model_type: str, random_state: int):
    model_type = (model_type or "").lower()
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                eval_metric="logloss",
            )
        except ImportError:
            pass
    return RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight="balanced_subsample")


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _smote_resample(X, y, random_state: int, k_neighbors: int = 5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2:
        raise ValueError("SMOTE resampling expects a binary target.")

    minority_label = classes[np.argmin(counts)]
    majority_count = int(counts.max())
    minority_indices = np.flatnonzero(y == minority_label)
    if minority_indices.size < 2:
        raise ValueError("SMOTE requires at least two samples in the minority class.")

    minority_X = X[minority_indices]
    k = min(k_neighbors, minority_indices.size - 1)
    rng = np.random.default_rng(random_state)
    synthetic_count = majority_count - minority_indices.size
    if synthetic_count <= 0:
        return X, y

    neighbor_lists = []
    for i, sample in enumerate(minority_X):
        distances = np.sqrt(np.sum((minority_X - sample) ** 2, axis=1))
        distances[i] = np.inf
        neighbor_lists.append(np.argpartition(distances, k)[:k])

    synthetic_samples = []
    for _ in range(synthetic_count):
        source_idx = int(rng.integers(minority_indices.size))
        neighbor_idx = int(rng.choice(neighbor_lists[source_idx]))
        source = minority_X[source_idx]
        neighbor = minority_X[neighbor_idx]
        gap = float(rng.random())
        synthetic_samples.append(source + gap * (neighbor - source))

    X_resampled = np.vstack([X, np.asarray(synthetic_samples, dtype=float)])
    y_resampled = np.concatenate([y, np.full(synthetic_count, minority_label, dtype=y.dtype)])
    return X_resampled, y_resampled


class ResampledClassifier:
    def __init__(self, preprocess, classifier, sampling_strategy: str = "smote", random_state: int = 42):
        self.preprocess = preprocess
        self.classifier = classifier
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.named_steps = {"preprocess": self.preprocess, "classifier": self.classifier}

    def fit(self, X, y):
        transformed = self.preprocess.fit_transform(X, y)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        if self.sampling_strategy == "smote":
            transformed, y = _smote_resample(transformed, y, random_state=self.random_state)

        self.classifier.fit(transformed, y)
        self.is_fitted_ = True
        return self

    def _transform(self, X):
        transformed = self.preprocess.transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return transformed

    def predict(self, X):
        return self.classifier.predict(self._transform(X))

    def predict_proba(self, X):
        return self.classifier.predict_proba(self._transform(X))

    def __getstate__(self):
        return {
            "preprocess": self.preprocess,
            "classifier": self.classifier,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
        }

    def __setstate__(self, state):
        self.preprocess = state["preprocess"]
        self.classifier = state["classifier"]
        self.sampling_strategy = state["sampling_strategy"]
        self.random_state = state["random_state"]
        self.named_steps = {"preprocess": self.preprocess, "classifier": self.classifier}


def train_model(
    df,
    feature_columns: List[str],
    target_column: str,
    model_type: str,
    numeric_columns: List[str],
    test_size: float,
    random_state: int,
    validation_size: float = 0.15,
    decision_values: Dict[str, float] | None = None,
    min_precision: float = 0.985,
    sampling_strategy: str | None = "smote",
    artifact_dir: str | Path | None = None,
) -> Dict[str, object]:
    """Train a real preprocessing + classification pipeline."""
    prepared = build_features(df)
    available_features = [column for column in feature_columns if column in prepared.columns]
    if not available_features:
        raise ValueError("No configured feature columns were found in the dataset.")

    labeled = prepared.dropna(subset=[target_column]).copy()
    X = labeled[available_features]
    y = labeled[target_column].astype(int)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must leave room for training data.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    validation_fraction = validation_size / (1 - test_size) if validation_size else 0
    if validation_fraction:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=y_train_val,
        )
    else:
        X_train, X_val, y_train, y_val = X_train_val, X_train_val.iloc[0:0], y_train_val, y_train_val.iloc[0:0]

    numeric_features = [column for column in numeric_columns if column in available_features]
    categorical_features = [column for column in available_features if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                SklearnPipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    classifier = _make_classifier(model_type, random_state)
    if sampling_strategy and sampling_strategy.lower() == "smote":
        model = ResampledClassifier(preprocess, classifier, sampling_strategy="smote", random_state=random_state)
    else:
        model = ResampledClassifier(preprocess, classifier, sampling_strategy="none", random_state=random_state)
    model.fit(X_train, y_train)

    if not X_val.empty:
        validation_probabilities = model.predict_proba(X_val)[:, 1]
        threshold_summary = select_decision_threshold(
            y_val,
            validation_probabilities,
            values=decision_values,
            min_precision=min_precision,
        )
        decision_threshold = float(threshold_summary["decision_threshold"])
        validation_metrics = evaluate_model(model, X_val, y_val, threshold=decision_threshold, values=decision_values)[
            "metrics"
        ]
    else:
        decision_threshold = 0.5
        threshold_summary = {
            "decision_threshold": decision_threshold,
            "precision_floor_met": True,
            "candidate_thresholds": 0,
            "expected_value": None,
        }
        validation_metrics = {}

    artifact_path = None
    if artifact_dir is not None:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_path / "loan_model.pkl"
        with artifact_path.open("wb") as handle:
            pickle.dump(model, handle)

    return {
        "model": model,
        "features": available_features,
        "sampling_strategy": sampling_strategy if sampling_strategy else "none",
        "train": {"X": X_train, "y": y_train},
        "validation": {"X": X_val, "y": y_val},
        "test": {"X": X_test, "y": y_test},
        "decision_threshold": decision_threshold,
        "threshold_summary": threshold_summary,
        "validation_metrics": validation_metrics,
        "split_summary": {
            "train_shape": (int(X_train.shape[0]), int(X_train.shape[1])),
            "validation_shape": (int(X_val.shape[0]), int(X_val.shape[1])),
            "test_shape": (int(X_test.shape[0]), int(X_test.shape[1])),
            "train_target_distribution": y_train.value_counts().to_dict(),
            "validation_target_distribution": y_val.value_counts().to_dict(),
            "test_target_distribution": y_test.value_counts().to_dict(),
            "sampling_strategy": sampling_strategy if sampling_strategy else "none",
        },
        "artifact_path": str(artifact_path) if artifact_path else None,
    }
