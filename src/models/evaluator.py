"""Evaluation utilities for model performance."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


DEFAULT_EXPECTED_VALUE = {
    "tp": 1.0,
    "tn": 0.25,
    "fp": -3.0,
    "fn": -0.5,
}


def _confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def calculate_expected_value(confusion: list[list[int]], values: Dict[str, float] | None = None) -> float:
    """Compute expected value from a confusion matrix and outcome values."""
    values = {**DEFAULT_EXPECTED_VALUE, **(values or {})}
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    return float(
        tp * values["tp"]
        + tn * values["tn"]
        + fp * values["fp"]
        + fn * values["fn"]
    )


def _metrics_from_predictions(y, y_pred, probabilities=None, threshold: float = 0.5, values: Dict[str, float] | None = None) -> Dict[str, object]:
    values = {**DEFAULT_EXPECTED_VALUE, **(values or {})}
    tn, fp, fn, tp = _confusion_counts(y, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "decision_threshold": float(threshold),
        "approval_rate": float(np.mean(y_pred)),
        "false_approval_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_rejection_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    metrics["expected_value"] = calculate_expected_value(metrics["confusion_matrix"], values)

    if probabilities is not None and len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))

    return metrics


def select_decision_threshold(
    y,
    probabilities,
    values: Dict[str, float] | None = None,
    min_precision: float = 0.985,
    thresholds: Iterable[float] | None = None,
) -> Dict[str, object]:
    """Select a conservative approval threshold from probability scores."""
    values = {**DEFAULT_EXPECTED_VALUE, **(values or {})}
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)

    rows = []
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        metrics = _metrics_from_predictions(y, y_pred, probabilities, threshold=float(threshold), values=values)
        rows.append(metrics)

    ranked = sorted(
        rows,
        key=lambda row: (
            row["expected_value"],
            row["precision"],
            row["recall"],
            row["decision_threshold"],
        ),
        reverse=True,
    )

    compliant = [row for row in ranked if row["precision"] >= min_precision]
    chosen = compliant[0] if compliant else ranked[0]
    chosen = dict(chosen)
    chosen["precision_floor_met"] = bool(chosen["precision"] >= min_precision)
    chosen["candidate_thresholds"] = len(rows)
    return chosen


def evaluate_model(
    model,
    X,
    y,
    threshold: float = 0.5,
    values: Dict[str, float] | None = None,
    predictions=None,
    probabilities=None,
) -> Dict[str, object]:
    """Evaluate a fitted classifier on a labeled split."""
    if probabilities is None and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    if predictions is None:
        predictions = (probabilities >= threshold).astype(int) if probabilities is not None else model.predict(X)
    metrics = _metrics_from_predictions(y, predictions, probabilities, threshold=threshold, values=values)

    metrics = {
        **metrics,
        "predictions": np.asarray(predictions).tolist(),
    }
    return {"predictions": np.asarray(predictions).tolist(), "metrics": metrics}
