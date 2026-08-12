"""Utilities for loading HMDA mortgage underwriting datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def resolve_dataset_path(data_path: str | Path, project_root: Path) -> Path:
    """Resolve a dataset path from config to an absolute path."""
    path = Path(data_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_dataset(path: str | Path, delimiter: str = "|") -> pd.DataFrame:
    """Load a pipe-delimited HMDA dataset as strings for safe parsing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(
        path,
        sep=delimiter,
        dtype=str,
        low_memory=False,
        na_values=["NA", ""],
        keep_default_na=True,
    )


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a concise profile for quick sanity checks."""
    target_column = "target" if "target" in df.columns else "decision" if "decision" in df.columns else None
    target_dist = (
        {str(key): int(value) for key, value in df[target_column].value_counts(dropna=True).to_dict().items()}
        if target_column is not None
        else (
            {str(key): int(value) for key, value in df["action_taken"].value_counts(dropna=True).head(10).to_dict().items()}
            if "action_taken" in df.columns
            else {}
        )
    )

    top_states = (
        df["state_code"].value_counts(dropna=False).head(10).to_dict()
        if "state_code" in df.columns
        else {}
    )

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "sample_columns": df.columns[:12].tolist(),
        "target_distribution": target_dist,
        "top_states": top_states,
    }
