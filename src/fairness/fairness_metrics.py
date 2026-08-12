"""Fairness metrics for mortgage decision models."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate


def _group_rate_mask(values: np.ndarray, group_value) -> np.ndarray:
    return values == group_value


def demographic_parity_difference(y_true, y_pred, sensitive_feature) -> Dict[str, object]:
    """Measure the spread in positive prediction rates across groups."""
    sensitive = np.asarray(sensitive_feature)
    predictions = np.asarray(y_pred)
    metric_frame = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=predictions, sensitive_features=sensitive)
    rates = {str(group): float(rate) for group, rate in metric_frame.by_group.items()}
    if len(rates) < 2:
        return {"metric": 0.0, "group_rates": rates}

    return {"metric": float(metric_frame.difference(method="between_groups")), "group_rates": rates}


def equalized_odds_difference(y_true, y_pred, sensitive_feature) -> Dict[str, object]:
    """Measure the maximum spread in TPR/FPR across groups."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    sensitive = np.asarray(sensitive_feature)
    tpr_frame = MetricFrame(metrics=true_positive_rate, y_true=truth, y_pred=predictions, sensitive_features=sensitive)
    fpr_frame = MetricFrame(metrics=false_positive_rate, y_true=truth, y_pred=predictions, sensitive_features=sensitive)
    tpr_by_group = {str(group): float(rate) for group, rate in tpr_frame.by_group.items()}
    fpr_by_group = {str(group): float(rate) for group, rate in fpr_frame.by_group.items()}
    tpr_spread = float(tpr_frame.difference(method="between_groups")) if len(tpr_by_group) >= 2 else 0.0
    fpr_spread = float(fpr_frame.difference(method="between_groups")) if len(fpr_by_group) >= 2 else 0.0
    return {
        "metric": float(max(tpr_spread, fpr_spread)),
        "true_positive_rate_by_group": tpr_by_group,
        "false_positive_rate_by_group": fpr_by_group,
    }


def disparate_impact(y_pred, sensitive_feature) -> Dict[str, object]:
    """Return the ratio between the least and most favored group approval rates."""
    sensitive = np.asarray(sensitive_feature)
    predictions = np.asarray(y_pred)
    metric_frame = MetricFrame(metrics=selection_rate, y_true=np.zeros_like(predictions), y_pred=predictions, sensitive_features=sensitive)
    rates = [float(rate) for rate in metric_frame.by_group.tolist()]
    if len(rates) < 2 or max(rates) == 0:
        return {"metric": 0.0, "group_rates": rates}

    return {"metric": float(min(rates) / max(rates)), "group_rates": rates}
