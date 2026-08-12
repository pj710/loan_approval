#%%
"""Fairlearn fairness mitigation analysis notebook."""

#%%
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import clean_dataset, convert_numeric_columns, create_binary_target, load_dataset
from src.fairness.fairness_metrics import demographic_parity_difference, disparate_impact, equalized_odds_difference
from src.fairness.fairness_selection import fairness_tradeoff_score
from src.models import evaluate_model, train_model
from src.models.fair_model import FairnessAwareModel
from src.utils.config_loader import load_config
from src.utils.paths import resolve_path

warnings.filterwarnings("ignore")
os.chdir(project_root)

config = load_config(project_root / "config.yaml")
paths = config["paths"]
reports_path = resolve_path(paths["reports"], project_root)
models_path = resolve_path(paths["models"], project_root)
reports_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

target_column = config["model"]["target"]
primary_attribute = config.get("fairness", {}).get("mitigation", {}).get("primary_attribute", "applicant_age")
decision_policy = config.get("decision_policy", {})

#%%
# Load and prepare the HMDA data.

raw_data_path = resolve_path(paths["data_raw"], project_root)
raw_df = load_dataset(raw_data_path, delimiter=paths.get("data_delimiter", "|"))
clean_df = clean_dataset(raw_df)
clean_df = convert_numeric_columns(clean_df, config["preprocessing"]["numeric_columns"])
feature_df = create_binary_target(
    clean_df,
    action_column=config["target_mapping"]["action_taken_column"],
    approved_codes=tuple(config["target_mapping"]["approved_codes"]),
    denied_codes=tuple(config["target_mapping"]["denied_codes"]),
    target_column=target_column,
)

available_features = [column for column in config["model"]["features"] if column in feature_df.columns]
prepared_df = feature_df[available_features + [primary_attribute, target_column]].copy()
prepared_df = prepared_df.dropna(subset=[target_column])

print(f"Rows: {prepared_df.shape[0]}")
print(f"Primary fairness attribute: {primary_attribute}")

#%%
# Train the base model used for all mitigation comparisons.

base_result = train_model(
    feature_df,
    feature_columns=config["model"]["features"],
    target_column=target_column,
    model_type=config["model"]["type"],
    numeric_columns=config["preprocessing"]["numeric_columns"],
    test_size=config["model"]["test_size"],
    random_state=config["model"]["random_seed"],
    validation_size=config["model"]["val_size"],
    decision_values=decision_policy.get("costs"),
    min_precision=decision_policy.get("min_precision", 0.985),
    sampling_strategy=config.get("sampling", {}).get("strategy", "smote"),
    artifact_dir=models_path / "fairlearn_analysis_base",
)

base_model = base_result["model"]
validation_split = base_result["validation"]
test_split = base_result["test"]

def _normalize_sensitive(series: pd.Series) -> pd.Series:
    return series.astype(str).replace({"8888": "Unknown"})


validation_sensitive = _normalize_sensitive(feature_df.loc[validation_split["X"].index, primary_attribute])
test_sensitive = _normalize_sensitive(feature_df.loc[test_split["X"].index, primary_attribute])

allowed_sensitive_values = []
for sensitive_value, group in validation_split["y"].groupby(validation_sensitive):
    if sensitive_value == "Unknown":
        continue
    if group.nunique() >= 2 and len(group) >= 2:
        allowed_sensitive_values.append(sensitive_value)

fit_mask = validation_sensitive.isin(allowed_sensitive_values)
base_test_probabilities = base_model.predict_proba(test_split["X"])[:, 1]
base_test_predictions = base_model.predict(test_split["X"])

#%%
# Compare baseline and Fairlearn mitigation strategies.

def _evaluate_strategy(name: str, predictions, probabilities):
    metrics = evaluate_model(
        base_model,
        test_split["X"],
        test_split["y"],
        threshold=base_result["decision_threshold"],
        predictions=predictions,
        probabilities=probabilities,
        values=decision_policy.get("costs"),
    )["metrics"]

    fairness = {
        "demographic_parity_difference": demographic_parity_difference(test_split["y"], predictions, test_sensitive),
        "equalized_odds_difference": equalized_odds_difference(test_split["y"], predictions, test_sensitive),
        "disparate_impact": disparate_impact(predictions, test_sensitive),
    }

    return {
        "strategy": name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "expected_value": metrics["expected_value"],
        "approval_rate": metrics["approval_rate"],
        "roc_auc": metrics.get("roc_auc"),
        "dpd_age": fairness["demographic_parity_difference"]["metric"],
        "eod_age": fairness["equalized_odds_difference"]["metric"],
        "di_age": fairness["disparate_impact"]["metric"],
    }


results = []
results.append(_evaluate_strategy("baseline", base_test_predictions, base_test_probabilities))

for constraint in ["demographic_parity", "equalized_odds"]:
    if not allowed_sensitive_values:
        continue

    postprocessor = ThresholdOptimizer(
        estimator=base_model,
        constraints=constraint,
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    postprocessor.fit(
        validation_split["X"].loc[fit_mask],
        validation_split["y"].loc[fit_mask],
        sensitive_features=validation_sensitive.loc[fit_mask],
    )
    fairness_model = FairnessAwareModel(
        base_model=base_model,
        postprocessor=postprocessor,
        fairness_feature=primary_attribute,
        fairness_strategy=constraint,
        allowed_sensitive_values=set(allowed_sensitive_values),
    )
    fair_predictions = fairness_model.predict(test_split["X"], sensitive_features=test_sensitive)
    fair_probabilities = fairness_model.predict_proba(test_split["X"])[:, 1]
    results.append(_evaluate_strategy(f"fairlearn_{constraint}", fair_predictions, fair_probabilities))

comparison_df = pd.DataFrame(results).sort_values(["dpd_age", "expected_value"], ascending=[True, False])
comparison_df["fairness_score"] = comparison_df.apply(
    lambda row: fairness_tradeoff_score(
        {
            "demographic_parity_difference": row["dpd_age"],
            "equalized_odds_difference": row["eod_age"],
            "disparate_impact": row["di_age"],
        }
    ),
    axis=1,
)
comparison_df = comparison_df.sort_values(["fairness_score", "precision", "expected_value"], ascending=[True, False, False])
comparison_df.to_csv(reports_path / "fairlearn_strategy_comparison.csv", index=False)
comparison_df

#%%
# Strategy selection note.
#
# The project uses the strategy with the lowest combined fairness score across
# demographic parity difference, equalized odds difference, and disparate impact,
# while keeping precision and expected value near the baseline.

selected_strategy = comparison_df.iloc[0].to_dict()
selected_strategy

#%%
# Protected-attribute fairness summary for the selected strategy.

selected_predictions = None
selected_probabilities = None
if selected_strategy["strategy"] == "baseline":
    selected_predictions = base_test_predictions
    selected_probabilities = base_test_probabilities
else:
    selected_constraint = selected_strategy["strategy"].replace("fairlearn_", "")
    selected_postprocessor = ThresholdOptimizer(
        estimator=base_model,
        constraints=selected_constraint,
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    selected_postprocessor.fit(
        validation_split["X"].loc[fit_mask],
        validation_split["y"].loc[fit_mask],
        sensitive_features=validation_sensitive.loc[fit_mask],
    )
    selected_model = FairnessAwareModel(
        base_model=base_model,
        postprocessor=selected_postprocessor,
        fairness_feature=primary_attribute,
        fairness_strategy=selected_constraint,
        allowed_sensitive_values=set(allowed_sensitive_values),
    )
    selected_predictions = selected_model.predict(test_split["X"], sensitive_features=test_sensitive)
    selected_probabilities = selected_model.predict_proba(test_split["X"])[:, 1]

protected_attributes = [column for column in config["fairness"]["protected_attributes"] if column in feature_df.columns]
fairness_rows = []
for attribute in protected_attributes:
    sensitive = _normalize_sensitive(feature_df.loc[test_split["X"].index, attribute])
    fairness_rows.append(
        {
            "attribute": attribute,
            "dpd": demographic_parity_difference(test_split["y"], selected_predictions, sensitive)["metric"],
            "eod": equalized_odds_difference(test_split["y"], selected_predictions, sensitive)["metric"],
            "di": disparate_impact(selected_predictions, sensitive)["metric"],
        }
    )

fairness_summary = pd.DataFrame(fairness_rows)
fairness_summary.to_csv(reports_path / "fairlearn_protected_attribute_fairness.csv", index=False)
fairness_summary

#%%
print("Fairlearn analysis complete.")
print(f"Comparison written to: {reports_path / 'fairlearn_strategy_comparison.csv'}")
print(f"Protected-attribute summary written to: {reports_path / 'fairlearn_protected_attribute_fairness.csv'}")
