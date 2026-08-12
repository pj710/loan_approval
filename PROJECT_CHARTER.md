# Project Charter: AI-Powered Mortgage Underwriting Assistant

## 1. Project Overview

**Project Name:** AI-Powered Mortgage Underwriting Assistant  
**Repository:** [loan_approval](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project)  
**Primary Goal:** Build an ML-based decision support system that helps mortgage lenders evaluate applications more consistently, transparently, and fairly.

## 2. Purpose

This project modernizes mortgage underwriting with a data-driven workflow that:

- predicts loan approval likelihood,
- explains model reasoning,
- evaluates fairness across protected attributes,
- supports analyst review through a dashboard and API.

The system is designed for portfolio/demo use and not for production lending without additional legal, compliance, and governance review.

## 3. Business Problem

Traditional mortgage underwriting is slow, manual, and inconsistent. It is also vulnerable to bias and limited transparency. The project addresses:

- long decision cycles,
- variability in analyst judgment,
- fairness and compliance concerns,
- limited explainability for stakeholders.

## 4. Objectives

### Core Objectives

1. Deliver a working ML pipeline for HMDA-based underwriting analysis.
2. Provide test-set performance metrics for model evaluation.
3. Surface fairness metrics across protected attributes.
4. Provide SHAP-based explanations for global and local interpretation.
5. Offer an interactive dashboard and API for operational use.

### Success Targets

- ROC AUC: `>= 0.75`
- F1: `>= 0.80`
- Fairness thresholds:
  - DPD `< 0.05`
  - EOD `< 0.05`
  - DI `>= 0.80`
- API latency: `< 500ms`

## 5. Scope

### In Scope

- HMDA data loading and validation
- preprocessing and binary target creation
- feature engineering and selection
- train/test split and model training
- test-set evaluation
- fairness assessment
- SHAP-based explanations
- FastAPI endpoints
- Streamlit dashboard
- analysis notebook for feature engineering and model evaluation

### Out of Scope

- production deployment
- real-time underwriting automation
- model governance workflow
- compliance approval
- credit bureau integration
- human-in-the-loop case management

## 6. Key Deliverables

- [pipeline.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/pipeline.py)
- [app/main.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/app/main.py)
- [app/dashboard.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/app/dashboard.py)
- [src/models/trainer.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/src/models/trainer.py)
- [src/models/evaluator.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/src/models/evaluator.py)
- [src/fairness/fairness_metrics.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/src/fairness/fairness_metrics.py)
- [src/explainability/shap_explainer.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/src/explainability/shap_explainer.py)
- [notebooks/03_feature_selection_engineering_training.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/notebooks/03_feature_selection_engineering_training.py)
- Generated model artifact under [models/](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/models)
- Generated report artifact under [reports/](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/reports)

## 7. Stakeholders

- **Project Owner / Analyst:** Josiah Gordor
- **Primary Users:** mortgage underwriting analysts, model reviewers
- **Secondary Users:** stakeholders reviewing fairness and explainability outputs

## 8. Assumptions

- HMDA source data is available and structurally valid.
- Required Python dependencies can be installed in the working environment.
- A labeled approval/denial target can be derived from HMDA action codes.
- The project will remain a decision-support tool rather than an automated approval system.

## 9. Constraints

- Data quality and missingness may vary by HMDA field.
- Protected-attribute fairness reporting depends on available demographic columns.
- The model is constrained by available labeled records after filtering.
- Streamlit and API usage depend on local execution environment stability.

## 10. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Label imbalance | Model bias or inflated metrics | Stratified splitting and threshold review |
| Missing fields | Reduced feature coverage | Validation and fallback feature selection |
| Fairness violations | Compliance concerns | Monitor DPD, EOD, and DI metrics |
| Explainability gaps | Harder model interpretation | SHAP-based recommendation layer |
| Environment drift | Runtime failures | Pin dependencies and keep artifacts reproducible |

## 11. Governance

- Model outputs are advisory, not final underwriting decisions.
- Fairness metrics must be reviewed with every major model iteration.
- Any production use requires legal and compliance validation.
- Report artifacts should be retained for reproducibility and auditability.

## 12. Technical Approach

### Data Flow

1. Load HMDA raw records.
2. Validate required columns.
3. Clean missing tokens and convert numeric fields.
4. Create a binary target from action codes.
5. Engineer and select features.
6. Train a classification model.
7. Evaluate on the test split.
8. Compute fairness metrics.
9. Generate SHAP-based explanations.
10. Publish results in API/dashboard/report artifacts.

### Stack

- Python
- pandas
- scikit-learn
- XGBoost
- SHAP
- FastAPI
- Streamlit

## 13. Current Status

- Data loading, validation, and target creation are implemented.
- Real model training and test evaluation are implemented.
- Fairness and explainability outputs are implemented.
- The dashboard supports online prediction and reporting.
- The model has been reviewed for overfitting. The observed train/test gap is negligible, so there is no material evidence of overfitting in the current split. The primary remaining risk is model overconfidence and threshold calibration for underwriting decisions.

## 14. Risk Review and Controls

- Overfitting risk: low based on current train/test diagnostics.
- Overconfidence risk: moderate because the model can assign extremely high approval probabilities even when underwriting-risk checks are poor.
- Control measures: conservative approval threshold, manual-review escalation for risky ratios, and monitored fairness metrics across protected groups.

## 15. Acceptance Criteria

The project is considered complete for portfolio/demo purposes when:

- the pipeline runs end-to-end,
- the dashboard renders with all major tabs,
- model and report artifacts are generated,
- fairness metrics and explanations are visible,
- online scoring works from the dashboard.


**Prepared By:** Josiah Gordor  

**Date:** 2026-08-12

