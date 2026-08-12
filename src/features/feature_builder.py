"""Create engineered features for underwriting models."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of stable, deterministic underwriting features."""
    features = df.copy()

    if "loan_amount" in features.columns and "income" in features.columns:
        loan_amount = pd.to_numeric(features["loan_amount"], errors="coerce")
        income = pd.to_numeric(features["income"], errors="coerce")
        features["loan_to_income_ratio"] = loan_amount.div(income.where(income != 0))

    if "property_value" in features.columns and "loan_amount" in features.columns:
        property_value = pd.to_numeric(features["property_value"], errors="coerce")
        loan_amount = pd.to_numeric(features["loan_amount"], errors="coerce")
        loan_to_property_value = loan_amount.div(property_value.where(property_value != 0))
        features["loan_to_property_value_ratio"] = loan_to_property_value

        if "combined_loan_to_value_ratio" in features.columns:
            existing_cltv = pd.to_numeric(features["combined_loan_to_value_ratio"], errors="coerce")
            features["combined_loan_to_value_ratio"] = existing_cltv.fillna(loan_to_property_value)
        else:
            features["combined_loan_to_value_ratio"] = loan_to_property_value

    if "debt_to_income_ratio" in features.columns:
        dti = pd.to_numeric(features["debt_to_income_ratio"], errors="coerce")
        dti_normalized = dti.where(dti <= 1, dti / 100.0)
        features["high_dti_flag"] = dti_normalized.gt(0.43).astype("float")

    if "combined_loan_to_value_ratio" in features.columns:
        cltv = pd.to_numeric(features["combined_loan_to_value_ratio"], errors="coerce")
        cltv_normalized = cltv.where(cltv <= 1, cltv / 100.0)
        features["high_cltv_flag"] = cltv_normalized.gt(0.80).astype("float")

    return features
