#%%
"""Feature selection, engineering/extraction, training, and evaluation notebook."""

#%%
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import clean_dataset, convert_numeric_columns, create_binary_target, load_dataset
from src.features.feature_builder import build_features
from src.fairness.fairness_metrics import demographic_parity_difference, disparate_impact, equalized_odds_difference
from src.models import evaluate_model, train_model
from src.utils.config_loader import load_config
from src.utils.paths import resolve_path

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

os.chdir(project_root)

config = load_config(project_root / "config.yaml")
paths = config["paths"]
raw_data_path = resolve_path(paths["data_raw"], project_root)
reports_path = resolve_path(paths["reports"], project_root)
models_path = resolve_path(paths["models"], project_root)
reports_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

target_column = config["model"]["target"]
model_config = config["model"]
numeric_columns = config["preprocessing"]["numeric_columns"]
core_features = config["model"]["features"]
engineered_features = [
    "loan_to_income_ratio",
    "loan_to_property_value_ratio",
    "high_dti_flag",
]
candidate_features = list(dict.fromkeys(core_features + engineered_features))

#%%
print(f"Project root: {project_root}")
print(f"Raw data: {raw_data_path}")
print(f"Reports: {reports_path}")
print(f"Models: {models_path}")

raw_df = load_dataset(raw_data_path, delimiter=paths.get("data_delimiter", "|"))
print(raw_df.shape)
raw_df.head()

#%%
# Extraction and preparation

clean_df = clean_dataset(raw_df)
clean_df = convert_numeric_columns(clean_df, numeric_columns)
feature_df = build_features(clean_df)
feature_df = create_binary_target(
    feature_df,
    action_column=config["target_mapping"]["action_taken_column"],
    approved_codes=tuple(config["target_mapping"]["approved_codes"]),
    denied_codes=tuple(config["target_mapping"]["denied_codes"]),
    target_column=target_column,
)

available_features = [column for column in candidate_features if column in feature_df.columns]
protected_attributes = [column for column in config["fairness"]["protected_attributes"] if column in feature_df.columns]
analysis_df = feature_df[available_features + protected_attributes + [target_column]].copy()
analysis_df = analysis_df.dropna(subset=[target_column])

print(f"Prepared rows: {analysis_df.shape[0]}")
print(f"Prepared features: {available_features}")

#%%
# Feature selection: missingness and coverage

missingness = (
    analysis_df[available_features]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .rename("missing_rate")
    .to_frame()
)
selection_summary = missingness.copy()
selection_summary["coverage"] = 1 - selection_summary["missing_rate"]
selection_summary.sort_values("missing_rate", ascending=False, inplace=True)
selection_summary.to_csv(reports_path / "feature_selection_summary.csv")

print(selection_summary.head(15))

plt.figure(figsize=(10, 6))
selection_summary.head(15)["missing_rate"].sort_values().plot(kind="barh", color="steelblue")
plt.title("Top Feature Missingness Rates")
plt.xlabel("Missing rate")
plt.tight_layout()
plt.show()

#%%
# Feature engineering notes
#
# - loan_to_income_ratio: loan_amount / income
# - loan_to_property_value_ratio: loan_amount / property_value
# - high_dti_flag: debt_to_income_ratio > 0.43
#
# These features help capture affordability and leverage signals that are not
# always visible in the raw HMDA columns.

engineered_preview = analysis_df[[c for c in engineered_features if c in analysis_df.columns]].describe(include="all")
engineered_preview

#%%
# Baseline training with all usable features

baseline_result = train_model(
    feature_df,
    feature_columns=available_features,
    target_column=target_column,
    model_type=model_config["type"],
    numeric_columns=numeric_columns + ["loan_to_income_ratio", "loan_to_property_value_ratio", "high_dti_flag"],
    test_size=model_config["test_size"],
    random_state=model_config["random_seed"],
    artifact_dir=models_path / "notebook_baseline",
)

baseline_model = baseline_result["model"]
baseline_val = baseline_result["validation"]
baseline_eval = evaluate_model(baseline_model, baseline_val["X"], baseline_val["y"])
baseline_metrics = baseline_eval["metrics"]
baseline_metrics

#%%
# Permutation importance for raw feature selection

perm = permutation_importance(
    baseline_model,
    baseline_val["X"],
    baseline_val["y"],
    n_repeats=5,
    random_state=model_config["random_seed"],
    scoring="f1",
)

importance_df = pd.DataFrame(
    {
        "feature": available_features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_mean", ascending=False)

importance_df.to_csv(reports_path / "permutation_feature_importance.csv", index=False)
importance_df.head(15)

#%%
plt.figure(figsize=(10, 6))
top_importance = importance_df.head(15).sort_values("importance_mean")
plt.barh(top_importance["feature"], top_importance["importance_mean"], xerr=top_importance["importance_std"], color="darkorange")
plt.title("Top Permutation Importances")
plt.xlabel("Mean importance (F1 drop)")
plt.tight_layout()
plt.show()

#%%
# Select the strongest features and retrain

top_n = min(8, len(importance_df))
selected_features = importance_df.head(top_n)["feature"].tolist()

selected_result = train_model(
    feature_df,
    feature_columns=selected_features,
    target_column=target_column,
    model_type=model_config["type"],
    numeric_columns=numeric_columns + ["loan_to_income_ratio", "loan_to_property_value_ratio", "high_dti_flag"],
    test_size=model_config["test_size"],
    random_state=model_config["random_seed"],
    artifact_dir=models_path / "notebook_selected",
)

selected_model = selected_result["model"]
selected_val = selected_result["validation"]
selected_eval = evaluate_model(selected_model, selected_val["X"], selected_val["y"])
selected_metrics = selected_eval["metrics"]
selected_metrics

#%%
# Compare baseline vs selected feature set

comparison = pd.DataFrame(
    [
        {"model": "baseline_all_features", **baseline_metrics},
        {"model": "selected_top_features", **selected_metrics},
    ]
)

comparison.to_csv(reports_path / "model_comparison_summary.csv", index=False)
comparison

#%%
# Model evaluation plots for the selected feature set

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    selected_val["y"],
    selected_eval["predictions"],
    ax=axes[0],
    cmap="Blues",
    colorbar=False,
)
axes[0].set_title("Selected Model Confusion Matrix")

RocCurveDisplay.from_estimator(
    selected_model,
    selected_val["X"],
    selected_val["y"],
    ax=axes[1],
)
axes[1].set_title("Selected Model ROC Curve")

plt.tight_layout()
plt.show()

#%%
# Fairness evaluation on the validation split

selected_predictions = np.asarray(selected_eval["predictions"])
fairness_rows = []

for attribute in protected_attributes:
    sensitive_values = feature_df.loc[selected_val["X"].index, attribute]
    valid_mask = sensitive_values.notna()
    if valid_mask.sum() < 2:
        continue

    y_subset = selected_val["y"].loc[valid_mask]
    p_subset = selected_predictions[valid_mask.to_numpy()]
    s_subset = sensitive_values.loc[valid_mask]

    fairness_rows.append(
        {
            "attribute": attribute,
            "demographic_parity_difference": demographic_parity_difference(y_subset, p_subset, s_subset)["metric"],
            "equalized_odds_difference": equalized_odds_difference(y_subset, p_subset, s_subset)["metric"],
            "disparate_impact": disparate_impact(p_subset, s_subset)["metric"],
        }
    )

fairness_summary = pd.DataFrame(fairness_rows).sort_values("attribute")
fairness_summary.to_csv(reports_path / "fairness_summary.csv", index=False)
fairness_summary

#%%
# Final interpretation

print("Baseline metrics:", baseline_metrics)
print("Selected metrics:", selected_metrics)
print("Selected features:", selected_features)

# The selected-feature model should be the preferred candidate if it preserves
# or improves F1/ROC-AUC while reducing feature complexity.
