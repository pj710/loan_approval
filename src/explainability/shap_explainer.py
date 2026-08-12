"""SHAP-based explainability helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _feature_name_map(model, feature_names: List[str] | None, X) -> List[str]:
    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(getattr(X, "shape", [0, 0])[1])]

    if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
        preprocess = model.named_steps["preprocess"]
        if hasattr(preprocess, "get_feature_names_out"):
            return list(preprocess.get_feature_names_out(feature_names))

    return feature_names


def explain_prediction(model, X, feature_names: List[str] | None = None, top_n: int = 10) -> Dict[str, object]:
    """Summarize the most influential features for the fitted model."""
    feature_names = _feature_name_map(model, feature_names, X)

    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        estimator = model.named_steps["classifier"]
    else:
        estimator = model

    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)

    if importances is None:
        return {"top_features": [], "note": "Model does not expose coefficients or feature importances."}

    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:top_n]
    return {
        "top_features": [{"feature": name, "importance": float(score)} for name, score in ranked],
    }


def explain_application(
    model,
    X: pd.DataFrame,
    feature_names: List[str] | None = None,
    top_n: int = 5,
) -> Dict[str, object]:
    """Return a local explanation and simple recommendations for a single application."""
    if X.empty:
        return {"contributions": [], "recommendations": [], "note": "No application data provided."}

    feature_names = _feature_name_map(model, feature_names, X)
    estimator = model.named_steps["classifier"] if hasattr(model, "named_steps") and "classifier" in model.named_steps else model
    transformed = model.named_steps["preprocess"].transform(X) if hasattr(model, "named_steps") and "preprocess" in model.named_steps else X

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)

    contributions = None
    try:
        import shap

        if hasattr(estimator, "get_booster") or estimator.__class__.__name__.lower().startswith("xgb"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(transformed)
            contributions = np.asarray(shap_values[0], dtype=float)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            coefficients = coef[0] if coef.ndim > 1 else coef
            contributions = transformed[0] * coefficients
    except Exception:
        contributions = None

    if contributions is None:
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
            contributions = importances * transformed[0]
        else:
            contributions = np.zeros(len(feature_names), dtype=float)

    local = sorted(zip(feature_names, contributions), key=lambda item: abs(item[1]), reverse=True)
    positive = [{"feature": name, "contribution": float(value)} for name, value in local if value > 0][:top_n]
    negative = [{"feature": name, "contribution": float(value)} for name, value in local if value < 0][:top_n]

    recommendation_map = {
        "numeric__debt_to_income_ratio": "Lower the debt-to-income ratio if possible.",
        "numeric__interest_rate": "Seek a lower interest rate or stronger pricing terms.",
        "numeric__loan_amount": "Reduce the requested loan amount if feasible.",
        "numeric__income": "Higher verified income would strengthen the application.",
        "numeric__property_value": "A higher property value can improve collateral strength.",
        "numeric__combined_loan_to_value_ratio": "Lower the loan-to-value ratio to reduce risk.",
        "numeric__loan_term": "A shorter loan term can sometimes improve risk profile.",
        "numeric__loan_to_income_ratio": "Lower the loan amount relative to income if possible.",
        "numeric__loan_to_property_value_ratio": "Increase collateral coverage or reduce the loan amount.",
        "numeric__high_dti_flag": "Bring debt-to-income below the underwriting review threshold.",
        "numeric__high_cltv_flag": "Bring combined loan-to-value below the underwriting review threshold.",
        "categorical__loan_type_1": "Conventional loan structure is generally favorable.",
        "categorical__occupancy_type_1": "Owner-occupied properties are usually viewed more favorably.",
    }

    recommendations = []
    for item in negative[:top_n]:
        recommendation = recommendation_map.get(item["feature"])
        if recommendation:
            recommendations.append(recommendation)

    if not recommendations and negative:
        recommendations.append("Focus on the strongest negative SHAP drivers to improve the approval score.")

    return {
        "contributions": [{"feature": name, "contribution": float(value)} for name, value in local[:top_n]],
        "positive_drivers": positive,
        "negative_drivers": negative,
        "recommendations": recommendations,
    }
