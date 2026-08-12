from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

from fairlearn.postprocessing import ThresholdOptimizer

from src.data import (
    clean_dataset,
    convert_numeric_columns,
    create_binary_target,
    load_dataset,
    resolve_dataset_path,
    summarize_dataset,
    summarize_target,
    validate_dataset,
)
from src.explainability.shap_explainer import explain_prediction
from src.fairness.fairness_metrics import demographic_parity_difference, disparate_impact, equalized_odds_difference
from src.fairness.fairness_selection import FairnessStrategyResult, choose_best_fairness_strategy, fairness_tradeoff_score
from src.models import evaluate_model, train_model
from src.models.fair_model import FairnessAwareModel
from src.utils.config_loader import load_config
from src.utils.paths import find_project_root


def run_training_pipeline() -> Dict[str, Any]:
    """Run the current preprocessing and split scaffold."""
    project_root = find_project_root()
    config = load_config()

    data_path = resolve_dataset_path(config["paths"]["data_raw"], project_root)
    delimiter = config["paths"].get("data_delimiter", "|")
    required_columns = config.get("data_validation", {}).get("required_columns", [])
    numeric_columns = config.get("preprocessing", {}).get("numeric_columns", [])
    target_config = config.get("target_mapping", {})
    feature_columns = config.get("model", {}).get("features", [])
    target_column = config.get("model", {}).get("target", "target")

    df = load_dataset(data_path, delimiter=delimiter)
    dataset_validation = validate_dataset(df, required_columns=required_columns)
    cleaned = clean_dataset(df)
    cleaned = convert_numeric_columns(cleaned, numeric_columns)
    labeled = create_binary_target(
        cleaned,
        action_column=target_config.get("action_taken_column", "action_taken"),
        approved_codes=tuple(target_config.get("approved_codes", ["1", "2"])),
        denied_codes=tuple(target_config.get("denied_codes", ["3"])),
        target_column=target_column,
    )

    summary = summarize_dataset(labeled)
    target_summary = summarize_target(labeled, target_column=target_column)

    model_config = config.get("model", {})
    decision_policy = config.get("decision_policy", {})
    sampling_config = config.get("sampling", {})
    fairness_config = config.get("fairness", {})
    mitigation_config = fairness_config.get("mitigation", {})
    fairness_mode = mitigation_config.get("strategy", "multi_metric")
    primary_attribute = mitigation_config.get("primary_attribute", "applicant_age")
    training_result = train_model(
        labeled,
        feature_columns=feature_columns,
        target_column=target_column,
        model_type=model_config.get("type", "random_forest"),
        numeric_columns=config.get("preprocessing", {}).get("numeric_columns", []),
        test_size=model_config.get("test_size", 0.2),
        random_state=model_config.get("random_seed", 42),
        validation_size=model_config.get("val_size", 0.15),
        decision_values=decision_policy.get("costs"),
        min_precision=decision_policy.get("min_precision", 0.985),
        sampling_strategy=sampling_config.get("strategy", "smote"),
        artifact_dir=project_root / config["paths"]["models"],
    )

    base_model = training_result["model"]
    split_result = training_result["test"]
    validation_result = training_result["validation"]
    if primary_attribute not in labeled.columns:
        raise ValueError(f"Fairness mitigation attribute '{primary_attribute}' is missing from the dataset.")

    validation_sensitive = labeled.loc[validation_result["X"].index, primary_attribute].astype(str).replace({"8888": "Unknown"})
    test_sensitive = labeled.loc[split_result["X"].index, primary_attribute].astype(str).replace({"8888": "Unknown"})
    valid_sensitive_values = []
    for sensitive_value, group in validation_result["y"].groupby(validation_sensitive):
        if sensitive_value == "Unknown":
            continue
        if group.nunique() >= 2 and len(group) >= 2:
            valid_sensitive_values.append(sensitive_value)

    mitigation_mask = validation_sensitive.isin(valid_sensitive_values)
    candidate_constraints = mitigation_config.get("candidate_constraints", ["demographic_parity", "equalized_odds"])
    candidate_results = []
    candidate_names = ["baseline"]
    for constraint in candidate_constraints:
        candidate_names.append(f"fairlearn_{constraint}")

    for candidate_name in candidate_names:
        postprocessor = None
        if candidate_name != "baseline" and valid_sensitive_values:
            constraint = candidate_name.replace("fairlearn_", "")
            postprocessor = ThresholdOptimizer(
                estimator=base_model,
                constraints=constraint,
                objective="accuracy_score",
                prefit=True,
                predict_method="predict_proba",
            )
            postprocessor.fit(
                validation_result["X"].loc[mitigation_mask],
                validation_result["y"].loc[mitigation_mask],
                sensitive_features=validation_sensitive.loc[mitigation_mask],
            )

        candidate_model = FairnessAwareModel(
            base_model=base_model,
            postprocessor=postprocessor,
            fairness_feature=primary_attribute,
            fairness_strategy=candidate_name,
            allowed_sensitive_values=set(valid_sensitive_values) if valid_sensitive_values else None,
        )
        validation_predictions = candidate_model.predict(
            validation_result["X"],
            sensitive_features=validation_sensitive,
            random_state=model_config.get("random_seed", 42),
        )
        validation_probabilities = candidate_model.predict_proba(validation_result["X"])[:, 1]
        validation_metrics = evaluate_model(
            candidate_model,
            validation_result["X"],
            validation_result["y"],
            threshold=training_result["decision_threshold"],
            values=decision_policy.get("costs"),
            predictions=validation_predictions,
            probabilities=validation_probabilities,
        )["metrics"]
        validation_fairness = {
            "demographic_parity_difference": demographic_parity_difference(
                validation_result["y"], validation_predictions, validation_sensitive
            )["metric"],
            "equalized_odds_difference": equalized_odds_difference(
                validation_result["y"], validation_predictions, validation_sensitive
            )["metric"],
            "disparate_impact": disparate_impact(validation_predictions, validation_sensitive)["metric"],
        }
        candidate_results.append(
            FairnessStrategyResult(
                name=candidate_name,
                predictions=validation_predictions,
                probabilities=validation_probabilities,
                postprocessor=postprocessor,
                metrics=validation_metrics,
                fairness_components=validation_fairness,
                fairness_score=fairness_tradeoff_score(validation_fairness),
                expected_value=float(validation_metrics["expected_value"]),
                precision=float(validation_metrics["precision"]),
            )
        )

    selected_candidate = choose_best_fairness_strategy(candidate_results, min_precision=decision_policy.get("min_precision", 0.985))
    selected_strategy = selected_candidate.name
    fairness_postprocessor = selected_candidate.postprocessor
    candidate_summaries = [
        {
            "strategy": candidate.name,
            "fairness_score": candidate.fairness_score,
            "precision": candidate.precision,
            "expected_value": candidate.expected_value,
            "demographic_parity_difference": candidate.fairness_components["demographic_parity_difference"],
            "equalized_odds_difference": candidate.fairness_components["equalized_odds_difference"],
            "disparate_impact": candidate.fairness_components["disparate_impact"],
        }
        for candidate in candidate_results
    ]

    fair_model = FairnessAwareModel(
        base_model=base_model,
        postprocessor=fairness_postprocessor,
        fairness_feature=primary_attribute,
        fairness_strategy=selected_strategy,
        allowed_sensitive_values=set(valid_sensitive_values) if valid_sensitive_values else None,
    )
    fair_predictions = fair_model.predict(
        split_result["X"],
        sensitive_features=test_sensitive,
        random_state=model_config.get("random_seed", 42),
    )
    fair_probabilities = fair_model.predict_proba(split_result["X"])[:, 1]
    fair_validation_predictions = fair_model.predict(
        validation_result["X"],
        sensitive_features=validation_sensitive,
        random_state=model_config.get("random_seed", 42),
    )
    fair_validation_probabilities = fair_model.predict_proba(validation_result["X"])[:, 1]
    fair_validation_metrics = evaluate_model(
        fair_model,
        validation_result["X"],
        validation_result["y"],
        threshold=training_result["decision_threshold"],
        values=decision_policy.get("costs"),
        predictions=fair_validation_predictions,
        probabilities=fair_validation_probabilities,
    )["metrics"]
    evaluation = evaluate_model(
        fair_model,
        split_result["X"],
        split_result["y"],
        threshold=training_result["decision_threshold"],
        values=decision_policy.get("costs"),
        predictions=fair_predictions,
        probabilities=fair_probabilities,
    )

    protected_attributes = fairness_config.get("protected_attributes", [])
    fairness_results: Dict[str, Any] = {}
    for attribute in protected_attributes:
        if attribute not in labeled.columns:
            continue

        sensitive_values = labeled.loc[split_result["X"].index, attribute]
        valid_mask = sensitive_values.notna()
        if not valid_mask.any():
            continue

        y_subset = split_result["y"].loc[valid_mask]
        sensitive_subset = sensitive_values.loc[valid_mask]
        prediction_subset = [fair_predictions[index] for index, keep in enumerate(valid_mask.to_numpy()) if keep]

        fairness_results[attribute] = {
            "demographic_parity_difference": demographic_parity_difference(
                y_subset,
                prediction_subset,
                sensitive_subset,
            ),
            "equalized_odds_difference": equalized_odds_difference(
                y_subset,
                prediction_subset,
                sensitive_subset,
            ),
            "disparate_impact": disparate_impact(
                prediction_subset,
                sensitive_subset,
            ),
        }

    explainability = explain_prediction(
        base_model,
        split_result["X"],
        feature_names=training_result["features"],
        top_n=config.get("explainability", {}).get("top_features", 10),
    )

    reports_dir = project_root / config["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "pipeline_report.json"
    pipeline_report = {
        "status": "ok",
        "validation": dataset_validation,
        "dataset_summary": summary,
        "target_summary": target_summary,
        "training_summary": {
            "model_type": model_config.get("type", "random_forest"),
            "features": training_result["features"],
            "train_shape": training_result["split_summary"]["train_shape"],
            "validation_shape": training_result["split_summary"]["validation_shape"],
            "test_shape": training_result["split_summary"]["test_shape"],
            "train_target_distribution": training_result["train"]["y"].value_counts().to_dict(),
            "validation_target_distribution": training_result["validation"]["y"].value_counts().to_dict(),
            "test_target_distribution": split_result["y"].value_counts().to_dict(),
            "decision_threshold": training_result["decision_threshold"],
            "threshold_summary": training_result["threshold_summary"],
            "decision_policy": decision_policy,
            "sampling_strategy": training_result["sampling_strategy"],
            "fairness_strategy": selected_strategy,
            "fairness_attribute": primary_attribute,
            "fairness_mode": fairness_mode,
            "artifact_path": training_result["artifact_path"],
        },
        "evaluation": evaluation["metrics"],
        "validation_metrics": training_result["validation_metrics"],
        "fair_validation_metrics": fair_validation_metrics,
        "fairness_mitigation": {
            "mode": fairness_mode,
            "strategy": selected_strategy,
            "objective": "composite_fairness",
            "primary_attribute": primary_attribute,
            "enabled": fairness_postprocessor is not None,
            "allowed_sensitive_values": valid_sensitive_values,
            "candidate_results": candidate_summaries,
        },
        "fairness": fairness_results,
        "explainability": explainability,
        "artifact_path": training_result["artifact_path"],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline_report, handle, indent=2)

    with open(training_result["artifact_path"], "wb") as handle:
        pickle.dump(fair_model, handle)

    return {
        "status": "ok",
        "project_root": str(project_root),
        "data_path": str(Path(data_path)),
        "validation": dataset_validation,
        "dataset_summary": summary,
        "target_summary": target_summary,
        "training_summary": {
            "model_type": model_config.get("type", "random_forest"),
            "features": training_result["features"],
            "train_shape": training_result["split_summary"]["train_shape"],
            "validation_shape": training_result["split_summary"]["validation_shape"],
            "test_shape": training_result["split_summary"]["test_shape"],
            "train_target_distribution": training_result["train"]["y"].value_counts().to_dict(),
            "validation_target_distribution": training_result["validation"]["y"].value_counts().to_dict(),
            "test_target_distribution": split_result["y"].value_counts().to_dict(),
            "decision_threshold": training_result["decision_threshold"],
            "threshold_summary": training_result["threshold_summary"],
            "decision_policy": decision_policy,
            "sampling_strategy": training_result["sampling_strategy"],
            "fairness_strategy": selected_strategy,
            "fairness_attribute": primary_attribute,
            "fairness_mode": fairness_mode,
            "artifact_path": training_result["artifact_path"],
        },
        "evaluation": evaluation["metrics"],
        "validation_metrics": training_result["validation_metrics"],
        "fair_validation_metrics": fair_validation_metrics,
        "fairness_mitigation": {
            "mode": fairness_mode,
            "strategy": selected_strategy,
            "objective": "composite_fairness",
            "primary_attribute": primary_attribute,
            "enabled": fairness_postprocessor is not None,
            "allowed_sensitive_values": valid_sensitive_values,
            "candidate_results": candidate_summaries,
        },
        "fairness": fairness_results,
        "explainability": explainability,
        "artifacts": {
            "model": training_result["artifact_path"],
            "report": str(report_path),
        },
    }
