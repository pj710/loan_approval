"""Fairness-aware model bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FairnessAwareModel:
    """Bundle a fitted base model with a Fairlearn postprocessor."""

    base_model: Any
    postprocessor: Any | None = None
    fairness_feature: str | None = None
    fairness_strategy: str | None = None
    allowed_sensitive_values: set[str] | None = None

    @property
    def named_steps(self):
        return getattr(self.base_model, "named_steps", {})

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X, sensitive_features=None, random_state=None):
        if self.postprocessor is not None and sensitive_features is not None:
            sensitive = sensitive_features
            if hasattr(sensitive_features, "columns"):
                sensitive = sensitive_features.iloc[:, 0]
            if hasattr(sensitive, "astype"):
                sensitive = sensitive.astype(str)
            else:
                sensitive = [str(value) for value in sensitive]

            if self.allowed_sensitive_values:
                import numpy as np

                base_predictions = np.asarray(self.base_model.predict(X))
                valid_mask = np.asarray([value in self.allowed_sensitive_values for value in sensitive])
                if valid_mask.any():
                    fair_predictions = self.postprocessor.predict(
                        X.loc[valid_mask] if hasattr(X, "loc") else X[valid_mask],
                        sensitive_features=(
                            sensitive.loc[valid_mask]
                            if hasattr(sensitive, "loc")
                            else [value for value, keep in zip(sensitive, valid_mask) if keep]
                        ),
                        random_state=random_state,
                    )
                    base_predictions[valid_mask] = fair_predictions
                return base_predictions.astype(int)

            return self.postprocessor.predict(
                X,
                sensitive_features=sensitive_features,
                random_state=random_state,
            )
        return self.base_model.predict(X)
