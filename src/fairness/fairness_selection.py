"""Strategy selection helpers for fairness mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.fairness.fairness_metrics import demographic_parity_difference, disparate_impact, equalized_odds_difference


@dataclass
class FairnessStrategyResult:
    name: str
    predictions: Any
    probabilities: Any
    postprocessor: Any | None
    metrics: dict[str, float]
    fairness_components: dict[str, float]
    fairness_score: float
    expected_value: float
    precision: float


def fairness_tradeoff_score(metrics: dict[str, float]) -> float:
    """Lower is better; 0 means parity across all three fairness metrics."""
    demographic_parity = float(metrics.get("demographic_parity_difference", 0.0))
    equalized_odds = float(metrics.get("equalized_odds_difference", 0.0))
    disparate_impact = float(metrics.get("disparate_impact", 0.0))
    return demographic_parity + equalized_odds + abs(1.0 - disparate_impact)


def cross_validated_fairness(
    model: Any,
    X,
    y,
    sensitive_feature,
    constraint: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Average fairness metrics across folds to reduce split-specific noise."""
    if X is None or y is None or len(np.unique(y)) < 2:
        return {
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "disparate_impact": 1.0,
        }

    allowed_values = []
    if hasattr(sensitive_feature, "astype"):
        safe_sensitive = sensitive_feature.astype(str).replace({"8888": "Unknown"})
        for value, group in y.groupby(safe_sensitive):
            if value == "Unknown":
                continue
            if len(np.unique(group)) >= 2 and len(group) >= 2:
                allowed_values.append(value)
        if not allowed_values:
            return {
                "demographic_parity_difference": 0.0,
                "equalized_odds_difference": 0.0,
                "disparate_impact": 1.0,
            }
        mask = safe_sensitive.isin(allowed_values)
        X = X.loc[mask] if hasattr(X, "loc") else X[mask]
        y = y.loc[mask] if hasattr(y, "loc") else y[mask]
        sensitive_feature = safe_sensitive.loc[mask] if hasattr(safe_sensitive, "loc") else safe_sensitive[mask]

    y_array = np.asarray(y)
    min_class_count = min(np.bincount(y_array.astype(int)).min(), n_splits)
    if min_class_count < 2:
        return {
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "disparate_impact": 1.0,
        }

    cv = StratifiedKFold(n_splits=min(n_splits, min_class_count), shuffle=True, random_state=random_state)
    fold_metrics = []
    for train_idx, val_idx in cv.split(X, y_array):
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_val = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        sensitive_train = sensitive_feature.iloc[train_idx] if hasattr(sensitive_feature, "iloc") else sensitive_feature[train_idx]
        sensitive_val = sensitive_feature.iloc[val_idx] if hasattr(sensitive_feature, "iloc") else sensitive_feature[val_idx]

        train_mask = sensitive_train.isin(allowed_values) if hasattr(sensitive_train, "isin") else np.isin(sensitive_train, allowed_values)
        val_mask = sensitive_val.isin(allowed_values) if hasattr(sensitive_val, "isin") else np.isin(sensitive_val, allowed_values)
        X_train = X_train.loc[train_mask] if hasattr(X_train, "loc") else X_train[train_mask]
        y_train = y_train.loc[train_mask] if hasattr(y_train, "loc") else y_train[train_mask]
        sensitive_train = sensitive_train.loc[train_mask] if hasattr(sensitive_train, "loc") else sensitive_train[train_mask]
        X_val = X_val.loc[val_mask] if hasattr(X_val, "loc") else X_val[val_mask]
        y_val = y_val.loc[val_mask] if hasattr(y_val, "loc") else y_val[val_mask]
        sensitive_val = sensitive_val.loc[val_mask] if hasattr(sensitive_val, "loc") else sensitive_val[val_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        from fairlearn.postprocessing import ThresholdOptimizer

        postprocessor = ThresholdOptimizer(
            estimator=model,
            constraints=constraint,
            objective="accuracy_score",
            prefit=True,
            predict_method="predict_proba",
        )
        postprocessor.fit(X_train, y_train, sensitive_features=sensitive_train)
        fold_predictions = postprocessor.predict(X_val, sensitive_features=sensitive_val, random_state=random_state)
        fold_metrics.append(
            {
                "demographic_parity_difference": demographic_parity_difference(y_val, fold_predictions, sensitive_val)["metric"],
                "equalized_odds_difference": equalized_odds_difference(y_val, fold_predictions, sensitive_val)["metric"],
                "disparate_impact": disparate_impact(fold_predictions, sensitive_val)["metric"],
            }
        )

    if not fold_metrics:
        return {
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "disparate_impact": 1.0,
        }

    agg = {
        "demographic_parity_difference": float(np.mean([metric["demographic_parity_difference"] for metric in fold_metrics])),
        "equalized_odds_difference": float(np.mean([metric["equalized_odds_difference"] for metric in fold_metrics])),
        "disparate_impact": float(np.mean([metric["disparate_impact"] for metric in fold_metrics])),
    }
    return agg


def choose_best_fairness_strategy(
    candidates: Iterable[FairnessStrategyResult],
    min_precision: float,
    precision_tolerance: float = 0.002,
) -> FairnessStrategyResult:
    """Pick the strategy with the best combined fairness score and acceptable precision."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("At least one fairness candidate is required.")

    compliant = [candidate for candidate in candidates if candidate.precision >= (min_precision - precision_tolerance)]
    ranked = compliant if compliant else candidates
    return sorted(
        ranked,
        key=lambda candidate: (
            candidate.fairness_score,
            -candidate.expected_value,
            -candidate.precision,
        ),
    )[0]
