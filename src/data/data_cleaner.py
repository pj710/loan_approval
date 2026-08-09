"""Helpers for cleaning and labeling mortgage application data."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


MISSING_TOKENS = {"NA", "", " ", "9999", "Exempt", "exempt"}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common HMDA missing values to pandas NA."""
    cleaned = df.copy()
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].where(~cleaned[column].isin(MISSING_TOKENS), pd.NA)
    return cleaned.infer_objects(copy=False)


def convert_numeric_columns(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected columns to numeric where possible."""
    converted = df.copy()
    for column in numeric_columns:
        if column in converted.columns:
            converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def create_binary_target(
    df: pd.DataFrame,
    action_column: str = "action_taken",
    approved_codes: Tuple[str, ...] = ("1", "2"),
    denied_codes: Tuple[str, ...] = ("3",),
    target_column: str = "target",
) -> pd.DataFrame:
    """Create a binary target from HMDA action codes, leaving other outcomes as NA."""
    labeled = df.copy()

    approved_set = set(approved_codes)
    denied_set = set(denied_codes)

    def map_action(value: object) -> object:
        if value in approved_set:
            return 1
        if value in denied_set:
            return 0
        return pd.NA

    labeled[target_column] = labeled[action_column].map(map_action)
    return labeled
