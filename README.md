# AI-Powered Mortgage Underwriting Assistant

An ML-based decision support system for mortgage underwriting using HMDA data.

## Project structure

- `app/` - FastAPI and Streamlit entry points
- `src/` - data, feature, model, fairness, and explainability modules
- `notebooks/` - analysis notebooks for feature selection and modeling
- `data/raw/` - HMDA source extract
- `reports/` - generated analysis artifacts
- `config.yaml` - project paths, features, and validation rules

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install fastapi streamlit uvicorn pandas scikit-learn xgboost pyyaml
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/pipeline
```

The `/pipeline` response now includes:

- dataset validation and summary
- train/test split statistics
- model metrics, including expected value
- fairness checks
- top feature explanations
- artifact paths for the saved model and report
- conservative decision thresholding with manual-review fallback
- SMOTE-based resampling during model training
- Fairlearn postprocessing on the primary age attribute
- composite fairness selection across demographic parity, equalized odds, and disparate impact

## Run the dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard includes:

- pipeline refresh controls
- dataset, model, fairness, prediction, and artifact tabs
- metric cards and feature-importance charts
- table-based summaries instead of raw JSON blocks
- download access for the latest pipeline report
- online prediction scoring with manual review for risky applications
- Fairlearn validation and mitigation summaries
- composite fairness strategy selection details

## Pipeline check

```bash
python -c "from pipeline import run_training_pipeline; print(run_training_pipeline())"
```

## Analysis notebook

Open [notebooks/03_feature_selection_engineering_training.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/notebooks/03_feature_selection_engineering_training.py)
for feature selection, engineering, extraction, training, and evaluation analysis.

Open [notebooks/04_fairlearn_fairness_mitigation.py](/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval.worktrees/draw-an-architectural-diagram-for-this-project/notebooks/04_fairlearn_fairness_mitigation.py)
for the Fairlearn strategy comparison and mitigation rationale.

### Fairlearn strategy comparison

| Strategy | Focus | Accuracy | Precision | Recall | Expected value | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | No mitigation | 99.82% | 100.00% | 99.77% | 2805.50 | Highest raw utility, but fairness gaps remain. |
| Fairlearn demographic parity | Balance approval rates | 98.32% | 99.10% | 98.84% | 2690.00 | Strongest reduction in selection-rate gaps. |
| Fairlearn equalized odds | Balance error rates | 99.48% | 100.00% | 99.36% | 2789.00 | Best fit when you want tighter TPR/FPR parity with smaller utility loss. |

The live pipeline compares these strategies and selects the best option under the composite fairness objective and utility guardrails.

## Model fit and overfitting review

A train/test comparison was performed to check whether the current model is overfitting.

- Train accuracy: 0.9972
- Test accuracy: 0.9979
- Train F1: 0.9983
- Test F1: 0.9987
- Train ROC AUC: 1.0000
- Test ROC AUC: 0.9999

These results show a negligible train/test gap, so there is no meaningful evidence of material overfitting in the current model. The more relevant production risk is overconfidence and threshold calibration: the model is highly predictive, but approval decisions are still filtered by a conservative 95% threshold and manual-review rules for high-risk applications.

## Current status

- Data loading, validation, cleaning, labeling, and train/test splitting are implemented.
- The API exposes `/health` and `/pipeline` with real training/evaluation output.
- The dashboard shows Fairlearn mitigation status and a fairness-adjusted prediction path.
- The pipeline saves a trained model bundle under `models/` and a JSON report under `reports/`.
- The model has been checked for overfitting and shows no material train/test separation; threshold calibration and approval-risk guardrails remain the key operational focus.
