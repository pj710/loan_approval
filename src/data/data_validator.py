"""Validation utilities for incoming loan application datasets."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def validate_dataset(df: pd.DataFrame, required_columns: Iterable[str] | None = None) -> Dict[str, object]:
    """Return validation summary including missing required fields."""
    required_columns = list(required_columns or [])
    missing = [column for column in required_columns if column not in df.columns]

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing,
    }


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """Return the list of required columns missing from the dataframe."""
    return [column for column in required_columns if column not in df.columns]


def summarize_target(df: pd.DataFrame, target_column: str = "target") -> Dict[str, int]:
    """Summarize binary target values and unlabeled records."""
    value_counts = df[target_column].value_counts(dropna=True).to_dict()
    return {str(key): int(value) for key, value in value_counts.items()}
