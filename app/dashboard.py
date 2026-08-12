from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

try:
    import pandas as pd
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency guard
    pd = None
    st = None

CACHE_DATA = st.cache_data if st is not None else (lambda **kwargs: (lambda fn: fn))
CACHE_RESOURCE = st.cache_resource if st is not None else (lambda **kwargs: (lambda fn: fn))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import run_training_pipeline
from src.data import clean_dataset, convert_numeric_columns, load_dataset
from src.explainability.shap_explainer import explain_application
from src.features.feature_builder import build_features
from src.utils.config_loader import load_config
from src.utils.paths import find_project_root, resolve_path


def _load_report(report_path: Path) -> dict | None:
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _flatten_fairness(report: dict) -> pd.DataFrame:
    fairness = report.get("fairness", {})
    rows = []
    for attribute, metrics in fairness.items():
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict) and "metric" in metric_value:
                rows.append(
                    {
                        "attribute": attribute,
                        "metric": metric_name,
                        "metric_label": _fairness_metric_label(metric_name),
                        "value": metric_value["metric"],
                    }
                )
    return pd.DataFrame(rows)


def _top_features(report: dict) -> pd.DataFrame:
    explainability = report.get("explainability", {})
    top_features = explainability.get("top_features", [])
    if not top_features:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame(top_features).rename(columns={"importance": "importance"})


def _target_counts(report: dict) -> pd.DataFrame:
    target_summary = report.get("target_summary", {})
    rows = []
    for label, count in target_summary.items():
        if label in {"<NA>", "nan", "None"}:
            continue
        rows.append({"class": str(label), "count": int(count)})
    return pd.DataFrame(rows)


def _dict_table(data: dict, key_name: str = "field", value_name: str = "value") -> pd.DataFrame:
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            rows.append({key_name: str(key), value_name: json.dumps(value, sort_keys=True)})
        elif isinstance(value, list):
            rows.append({key_name: str(key), value_name: ", ".join(map(str, value))})
        else:
            rows.append({key_name: str(key), value_name: value})
    return pd.DataFrame(rows)


def _fairness_metric_label(metric_name: str) -> str:
    labels = {
        "demographic_parity_difference": "DPD [<0.05]",
        "equalized_odds_difference": "EOD [<0.05]",
        "disparate_impact": "DI [>=0.80]",
    }
    return labels.get(metric_name, metric_name)


def _display_name(feature: str) -> str:
    return feature.replace("_", " ").title()


DERIVED_FEATURES = {"loan_to_income_ratio", "loan_to_property_value_ratio", "high_dti_flag", "high_cltv_flag"}


def _safe_float(series: pd.Series, default: float = 0.0) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return default
    return float(numeric.median())


def _normalize_ratio(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    numeric = float(value)
    if 1.0 < numeric <= 100.0:
        return numeric / 100.0
    return numeric


def _risk_flags(application: pd.Series) -> list[str]:
    flags = []
    dti = _normalize_ratio(application.get("debt_to_income_ratio", 0) or 0)
    cltv = _normalize_ratio(application.get("combined_loan_to_value_ratio", 0) or 0)
    ltv = _normalize_ratio(application.get("loan_to_property_value_ratio", 0) or 0)

    if dti <= 0 and "loan_amount" in application and "income" in application:
        loan_amount = float(application.get("loan_amount", 0) or 0)
        income = float(application.get("income", 0) or 0)
        dti = loan_amount / income if income else 0.0

    if cltv <= 0 and "loan_amount" in application and "property_value" in application:
        loan_amount = float(application.get("loan_amount", 0) or 0)
        property_value = float(application.get("property_value", 0) or 0)
        cltv = loan_amount / property_value if property_value else 0.0

    if ltv <= 0 and "loan_amount" in application and "property_value" in application:
        loan_amount = float(application.get("loan_amount", 0) or 0)
        property_value = float(application.get("property_value", 0) or 0)
        ltv = loan_amount / property_value if property_value else 0.0

    if dti > 0.43:
        flags.append("Debt-to-income ratio exceeds 43%.")
    if cltv > 0.80:
        flags.append("Combined loan-to-value ratio exceeds 80%.")
    if ltv > 0.90:
        flags.append("Loan-to-property-value ratio exceeds 90%.")
    return flags


CACHE_DATA(show_spinner=False)
def _prediction_metadata(raw_data_path: str, delimiter: str, feature_columns: tuple[str, ...], numeric_columns: tuple[str, ...]) -> dict:
    df = load_dataset(raw_data_path, delimiter=delimiter)
    df = clean_dataset(df)
    df = convert_numeric_columns(df, numeric_columns)

    metadata = {}
    for feature in feature_columns:
        if feature in numeric_columns:
            series = pd.to_numeric(df[feature], errors="coerce") if feature in df.columns else pd.Series(dtype=float)
            clean_series = series.dropna()
            metadata[feature] = {
                "kind": "numeric",
                "default": float(clean_series.median()) if not clean_series.empty else 0.0,
                "min": float(clean_series.min()) if not clean_series.empty else 0.0,
                "max": float(clean_series.max()) if not clean_series.empty else 1.0,
            }
        else:
            options = []
            if feature in df.columns:
                options = df[feature].dropna().astype(str).value_counts().head(20).index.tolist()
            metadata[feature] = {
                "kind": "categorical",
                "options": options or ["Unknown"],
                "default": options[0] if options else "Unknown",
            }
    return metadata


CACHE_RESOURCE(show_spinner=False)
def _load_model(model_path: str):
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def _style_fairness_table(table: pd.DataFrame) -> pd.io.formats.style.Styler:
    def style_row(row: pd.Series) -> list[str]:
        styles = []
        for column, value in row.items():
            if column == "attribute" or pd.isna(value):
                styles.append("")
                continue

            violated = False
            if column in {"DPD [<0.05]", "EOD [<0.05]"}:
                violated = float(value) > 0.05
            elif column == "DI [>=0.80]":
                violated = float(value) < 0.80

            if violated:
                styles.append("background-color: #f8d7da; color: #842029; font-weight: 600;")
            else:
                styles.append("background-color: #d1e7dd; color: #0f5132;")
        return styles

    return table.style.apply(style_row, axis=1)


def _metric_columns(report: dict) -> None:
    training = report.get("training_summary", {})
    evaluation = report.get("evaluation", {})
    dataset = report.get("dataset_summary", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{dataset.get('row_count', 0):,}")
    c2.metric("Features", f"{len(training.get('features', [])):,}")
    c3.metric("Test F1", f"{evaluation.get('f1', 0):.4f}")
    c4.metric("ROC AUC", f"{evaluation.get('roc_auc', 0):.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Accuracy", f"{evaluation.get('accuracy', 0):.4f}")
    c6.metric("Precision", f"{evaluation.get('precision', 0):.4f}")
    c7.metric("Recall", f"{evaluation.get('recall', 0):.4f}")
    c8.metric("Expected value", f"{evaluation.get('expected_value', 0):.2f}")

    threshold = training.get("decision_threshold", 0.5)
    sampling = training.get("sampling_strategy", "none")
    fairness_strategy = training.get("fairness_strategy", "none")
    fairness_attribute = training.get("fairness_attribute", "none")
    st.caption(
        f"Auto-approval threshold: {threshold:.0%}. Training sampling: {sampling}. "
        f"Fairlearn strategy: {fairness_strategy} on {fairness_attribute}. "
        "Applications near the cutoff are routed to manual review."
    )


def _render_prediction_tab(report: dict, project_root: Path, config: dict) -> None:
    st.subheader("Online prediction")
    st.caption("Enter an application profile to score approval probability, then inspect the Fairlearn-adjusted decision and SHAP drivers.")

    training_summary = report.get("training_summary", {})
    feature_columns = tuple(training_summary.get("features", config.get("model", {}).get("features", [])))
    raw_feature_columns = tuple(feature for feature in feature_columns if feature not in DERIVED_FEATURES)
    numeric_columns = tuple(config.get("preprocessing", {}).get("numeric_columns", []))
    metadata = _prediction_metadata(
        str(resolve_path(config["paths"]["data_raw"], project_root)),
        config["paths"].get("data_delimiter", "|"),
        raw_feature_columns,
        numeric_columns,
    )

    model_path = training_summary.get("artifact_path") or report.get("artifact_path")
    if not model_path or not Path(model_path).exists():
        st.warning("Trained model artifact not found. Run the pipeline first.")
        return

    model_bundle = _load_model(model_path)
    base_model = getattr(model_bundle, "base_model", model_bundle)
    fairness_feature = getattr(model_bundle, "fairness_feature", None)

    def _derived_cltv_value(current_inputs: dict[str, object]) -> float:
        loan_amount = current_inputs.get("loan_amount")
        property_value = current_inputs.get("property_value")
        if loan_amount in (None, "") or property_value in (None, ""):
            return 0.0
        try:
            loan_amount_float = float(loan_amount)
            property_value_float = float(property_value)
        except (TypeError, ValueError):
            return 0.0
        if property_value_float == 0:
            return 0.0
        return loan_amount_float / property_value_float

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        inputs: dict[str, object] = {}
        for index, feature in enumerate(raw_feature_columns):
            target_col = col_a if index % 2 == 0 else col_b
            spec = metadata.get(feature, {})
            with target_col:
                if feature == "combined_loan_to_value_ratio":
                    computed_cltv = _derived_cltv_value(inputs)
                    default_value = computed_cltv if computed_cltv > 0 else float(spec.get("default", 0.0))
                    inputs[feature] = st.number_input(
                        _display_name(feature),
                        value=default_value,
                        min_value=0.0,
                        max_value=max(default_value * 2, 1.0),
                    )
                elif spec.get("kind") == "numeric":
                    inputs[feature] = st.number_input(
                        _display_name(feature),
                        value=float(spec.get("default", 0.0)),
                        min_value=float(spec.get("min", 0.0)),
                        max_value=float(spec.get("max", max(float(spec.get("default", 0.0)) * 2, 1.0))),
                    )
                else:
                    options = spec.get("options", ["Unknown"])
                    default = spec.get("default", options[0])
                    inputs[feature] = st.selectbox(_display_name(feature), options, index=options.index(default) if default in options else 0)

        submitted = st.form_submit_button("Score application", use_container_width=True)

    if not submitted:
        st.info("Complete the form and click Score application to generate a prediction.")
        return

    if inputs.get("combined_loan_to_value_ratio") in (None, ""):
        computed_cltv = _derived_cltv_value(inputs)
        if computed_cltv > 0:
            inputs["combined_loan_to_value_ratio"] = computed_cltv

    input_df = pd.DataFrame([inputs], columns=list(raw_feature_columns))
    enriched_input = build_features(input_df)
    model_input = enriched_input.reindex(columns=list(feature_columns))

    probability = float(model_bundle.predict_proba(model_input)[0, 1])
    threshold = float(training_summary.get("decision_threshold", 0.5))
    review_margin = float(config.get("decision_policy", {}).get("review_margin", 0.10))
    flags = _risk_flags(enriched_input.iloc[0])
    fair_decision = None
    if fairness_feature and fairness_feature in enriched_input.columns and hasattr(model_bundle, "predict"):
        fair_decision = int(
            model_bundle.predict(
                model_input,
                sensitive_features=enriched_input[[fairness_feature]],
                random_state=training_summary.get("random_seed", 42),
            )[0]
        )

    if flags or (probability >= threshold - review_margin and probability < threshold):
        decision = "Manual review"
    elif fair_decision is not None:
        decision = "Approved" if fair_decision == 1 else "Denied"
    else:
        decision = "Approved" if probability >= threshold else "Denied"

    c1, c2, c3 = st.columns(3)
    c1.metric("Approval probability", f"{probability:.1%}")
    c2.metric("Prediction", decision)
    c3.metric("Score threshold", f"{threshold:.0%}")

    st.subheader("Application inputs")
    st.dataframe(enriched_input.reindex(columns=list(feature_columns)), use_container_width=True, hide_index=True)

    if flags:
        st.warning("Manual review required: " + " ".join(flags))

    explanation = explain_application(base_model, model_input, feature_names=list(feature_columns), top_n=5)
    contrib_df = pd.DataFrame(explanation.get("contributions", []))

    left, right = st.columns(2)
    with left:
        st.subheader("Top SHAP drivers")
        if not contrib_df.empty:
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        else:
            st.info("No SHAP contribution data was returned.")

    with right:
        st.subheader("Recommendations")
        recommendations = explanation.get("recommendations", [])
        if recommendations:
            for recommendation in recommendations:
                st.write(f"- {recommendation}")
        else:
            st.write("No recommendation was generated for this application.")

    if not contrib_df.empty:
        st.subheader("Contribution breakdown")
        st.bar_chart(contrib_df.set_index("feature"))


def run_dashboard() -> None:
    """Render an interactive dashboard for the loan approval pipeline."""
    if st is None or pd is None:
        raise ImportError("Streamlit and pandas are required to run the dashboard. Install the project dependencies first.")

    project_root = find_project_root(PROJECT_ROOT)
    config = load_config(project_root / "config.yaml")
    reports_dir = resolve_path(config["paths"]["reports"], project_root)
    report_path = reports_dir / "pipeline_report.json"
    model_path = resolve_path(config["paths"]["models"], project_root) / "loan_model.pkl"

    st.set_page_config(
        page_title="Loan Approval Assistant",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "dashboard_report" not in st.session_state:
        st.session_state.dashboard_report = _load_report(report_path)
    if "dashboard_source" not in st.session_state:
        st.session_state.dashboard_source = "saved report" if st.session_state.dashboard_report else "none"

    st.sidebar.title("Controls")
    st.sidebar.caption("Use the controls below to refresh the model summary and inspect the latest artifacts.")

    if st.sidebar.button("Refresh pipeline", use_container_width=True):
        with st.spinner("Running the training pipeline..."):
            st.session_state.dashboard_report = run_training_pipeline()
            st.session_state.dashboard_source = "live pipeline run"
        st.rerun()

    if st.sidebar.button("Load saved report", use_container_width=True):
        st.session_state.dashboard_report = _load_report(report_path)
        st.session_state.dashboard_source = "saved report"
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Artifacts")
    st.sidebar.code(str(model_path), language="text")
    st.sidebar.code(str(report_path), language="text")
    st.sidebar.write(f"Source: **{st.session_state.dashboard_source}**")

    report = st.session_state.dashboard_report
    st.title("Loan Approval Assistant")
    st.caption("Interactive view of the mortgage underwriting pipeline, model quality, and fairness signals.")

    if not report:
        st.info("No saved report is available yet. Run the pipeline from the sidebar to generate one.")
        st.stop()

    _metric_columns(report)

    tabs = st.tabs(["Overview", "Data", "Model", "Fairness", "Predictions", "Artifacts"])

    with tabs[0]:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Dataset summary")
            st.dataframe(_dict_table(report.get("dataset_summary", {}), "metric", "value"), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Target distribution")
            target_df = _target_counts(report)
            if not target_df.empty:
                st.bar_chart(target_df.sort_values("class").set_index("class"))
            else:
                st.write("No target summary available.")

    with tabs[1]:
        st.subheader("Train, validation, and test summary")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Dataset validation")
            validation_df = _dict_table(report.get("validation", {}), "field", "value")
            if not validation_df.empty:
                st.dataframe(validation_df, use_container_width=True, hide_index=True)
        with c2:
            st.write("Training summary")
            training_df = _dict_table(report.get("training_summary", {}), "field", "value")
            if not training_df.empty:
                st.dataframe(training_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Model quality")
        validation_metrics = report.get("validation_metrics", {})
        if validation_metrics:
            st.write("Validation metrics used to select the approval threshold")
            st.dataframe(_dict_table(validation_metrics, "metric", "value"), use_container_width=True, hide_index=True)

        fair_validation_metrics = report.get("fair_validation_metrics", {})
        if fair_validation_metrics:
            st.write("Fairlearn validation metrics")
            st.dataframe(_dict_table(fair_validation_metrics, "metric", "value"), use_container_width=True, hide_index=True)

        fairness_mitigation = report.get("fairness_mitigation", {})
        if fairness_mitigation:
            st.write("Fairness mitigation strategy")
            st.dataframe(_dict_table(fairness_mitigation, "field", "value"), use_container_width=True, hide_index=True)

        st.write("Test metrics")
        evaluation_df = pd.DataFrame(
            [{"metric": k, "value": v} for k, v in report.get("evaluation", {}).items() if k != "confusion_matrix"]
        )
        st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

        top_features = _top_features(report)
        if not top_features.empty:
            st.subheader("Top feature explanations")
            st.dataframe(top_features, use_container_width=True, hide_index=True)
            chart_df = top_features.sort_values("importance", ascending=True).set_index("feature")
            st.bar_chart(chart_df)
        else:
            st.info("No explainability output was found in the report.")

    with tabs[3]:
        st.subheader("Fairness metrics")
        fairness_df = _flatten_fairness(report)
        if not fairness_df.empty:
            pivot = fairness_df.pivot(index="attribute", columns="metric_label", values="value").reset_index()
            metric_cols = [col for col in pivot.columns if col != "attribute"]
            pivot = pivot[["attribute"] + sorted(metric_cols)]
            st.dataframe(_style_fairness_table(pivot), use_container_width=True, hide_index=True)
        else:
            st.info("No fairness metrics were generated for the current report.")

    with tabs[4]:
        _render_prediction_tab(report, project_root, config)

    with tabs[5]:
        st.subheader("Saved artifacts")
        st.code(report.get("artifact_path", "No model artifact recorded."), language="text")
        st.code(report.get("artifacts", {}).get("report", str(report_path)), language="text")

        st.subheader("Report tables")
        top_level = _dict_table(
            {
                "status": report.get("status"),
                "project_root": report.get("project_root"),
                "data_path": report.get("data_path"),
                "model_artifact": report.get("artifact_path"),
                "report_artifact": report.get("artifacts", {}).get("report", str(report_path)),
            },
            "field",
            "value",
        )
        st.dataframe(top_level, use_container_width=True, hide_index=True)

        st.download_button(
            "Download pipeline report",
            data=json.dumps(report, indent=2),
            file_name="pipeline_report.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    run_dashboard()
