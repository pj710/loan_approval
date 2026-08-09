# Loan Approval Project

This repository now includes a starter package layout for an AI-powered mortgage underwriting assistant.

## Structure

- loan_approval/ : package modules and configuration
- src/ : reusable source modules for data, features, models, fairness, and explainability
- notebooks/ : exploratory analysis notebooks
- reports/ : generated reports and artifacts

## Run the API

```bash
uvicorn loan_approval.app.main:app --reload
```

## Run the dashboard

```bash
streamlit run loan_approval/app/dashboard.py
```

## Run the pipeline scaffold

```bash
python -c "from loan_approval.pipeline import run_training_pipeline; print(run_training_pipeline())"
```
