# Underwriter Guide

**Author:** Josiah Gordor  
**Last Updated:** February 2026

---

## Introduction

Welcome to the AI-Powered Mortgage Underwriting Assistant! This guide will help you use the dashboard effectively to make informed loan approval decisions.

### What This System Does

- Provides **instant approval recommendations** based on ML analysis
- Shows **confidence levels** and **risk assessments** for each application
- Explains **which factors** influenced the recommendation
- Monitors **fairness** across demographic groups
- Supports **multiple fair ML models** for comparison

### What This System Does NOT Do

- It does **NOT** make final approval decisions (that's your job!)
- It does **NOT** replace human judgment
- It does **NOT** have access to full credit bureau data

---

## Getting Started

### Accessing the Dashboard

1. Open your web browser
2. Navigate to: `http://localhost:8501` (or your organization's URL)
3. Wait for the dashboard to load

### Checking System Status

Look at the sidebar for **API Status**:
- ✅ **API Online** - System is ready
- ⚠️ **Models Loading** - Wait a moment
- ❌ **API Offline** - Contact IT support

---

## Using the Loan Application Form

### Required Fields

| Field | Description | Valid Range |
|-------|-------------|-------------|
| **Loan Amount** | Requested loan in dollars | $10,000 - $5,000,000 |
| **Property Value** | Appraised property value | $10,000 - $10,000,000 |
| **Annual Income** | Borrower's gross income | $0 - $10,000,000 |
| **Interest Rate** | Proposed rate (%) | 0% - 20% |
| **Loan Term** | Term in months | 60 - 480 |

### Optional Fields

| Field | Description | Default |
|-------|-------------|---------|
| **State** | US state code | Auto-detected |
| **FHA Loan** | Federal Housing Administration | No |
| **VA Loan** | Veterans Affairs | No |
| **Has Co-Borrower** | Joint application | No |
| **Applicant Age** | Age in years | 35 |

### Selecting a Model

In the sidebar, choose from:

| Model | Best For |
|-------|----------|
| **XGBoost Fair** | Balance of accuracy & fairness |
| **GLM Fair** | Highest accuracy (98.1%) |
| **Logistic Regression Fair** | Fast, interpretable decisions |
| **FasterRisk Fair** | Transparent scoring rules |
| **GOSDT Fair** | Clear decision tree rules |

---

## Understanding Results

### Prediction Display

After submitting an application, you'll see:

```
┌─────────────────────────────────────┐
│  ✅ APPROVED                        │
│  Probability: 85.2%                 │
│  Confidence: 70.4%                  │
│  Risk Level: Low Risk               │
│  Model: XGBoost Fair                │
│  Processing Time: 125ms             │
└─────────────────────────────────────┘
```

### Risk Levels

| Risk Level | Probability | Recommended Action |
|------------|-------------|-------------------|
| 🟢 **Low Risk** | ≥ 80% | Likely approval candidate |
| 🟡 **Moderate Risk** | 50-79% | Review key factors carefully |
| 🔴 **High Risk** | < 50% | Document denial reasoning |

### Confidence Score

The confidence score indicates how certain the model is:

| Confidence | Interpretation |
|------------|----------------|
| **≥ 80%** | High confidence - model is very certain |
| **60-79%** | Moderate confidence - typical case |
| **< 60%** | Low confidence - manual review recommended |

---

## Interpreting Key Factors

### SHAP Explanation

The system shows which factors influenced the decision:

```
Top Factors Increasing Approval:
  • income: +0.15 (High income supports approval)
  • loan_to_value_ratio: +0.08 (Good LTV ratio)
  • has_coborrower: +0.05 (Co-borrower reduces risk)

Top Factors Decreasing Approval:
  • interest_rate: -0.12 (High rate indicates risk)
  • loan_to_income_ratio: -0.06 (High DTI concern)
```

### Factor Impact Scale

| SHAP Value | Impact |
|------------|--------|
| **> +0.10** | Strong positive influence |
| **+0.05 to +0.10** | Moderate positive influence |
| **-0.05 to +0.05** | Minimal influence |
| **-0.10 to -0.05** | Moderate negative influence |
| **< -0.10** | Strong negative influence |

---

## Underwriting Ratios

The dashboard automatically calculates key ratios:

### Loan-to-Value (LTV)

```
LTV = Loan Amount / Property Value × 100
```

| LTV Range | Assessment |
|-----------|------------|
| < 80% | Excellent - No PMI required |
| 80-90% | Good - Standard terms |
| 90-95% | Acceptable - Higher scrutiny |
| > 95% | High risk - May require MI |

### Loan-to-Income (LTI)

```
LTI = Loan Amount / Annual Income
```

| LTI Range | Assessment |
|-----------|------------|
| < 3.0 | Conservative - Strong position |
| 3.0-4.0 | Moderate - Typical range |
| 4.0-5.0 | Stretched - Careful review |
| > 5.0 | High - Additional documentation |

### Housing Expense Ratio

```
HER = (Monthly Payment / Monthly Income) × 100
```

| HER Range | Assessment |
|-----------|------------|
| < 28% | Ideal - Well within means |
| 28-33% | Acceptable - Standard guideline |
| 33-36% | Elevated - Compensating factors needed |
| > 36% | High - Strong reserves required |

---

## Using the Fairness Dashboard

### Purpose

The Fairness Dashboard ensures decisions are equitable across:
- **Race** (White, Black, Asian, Other)
- **Ethnicity** (Hispanic, Non-Hispanic)
- **Sex** (Male, Female)

### Fairness Metrics

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| **Demographic Parity Difference (DPD)** | < 0.05 | Approval rates are similar across groups |
| **Equalized Odds Difference (EOD)** | < 0.10 | Error rates are similar across groups |
| **Disparate Impact (80% Rule)** | > 0.80 | No group approved at < 80% of highest group |

### Reading the Charts

- **Green bars** = Passing fairness thresholds
- **Red/Orange bars** = Potential fairness concerns

If you see red indicators, consult with compliance before proceeding with borderline cases.

---

## Workflow Recommendations

### Standard Application

1. Enter application details in the form
2. Click "Get Prediction"
3. Review risk level and probability
4. Check key factors for reasonableness
5. Document decision with supporting factors

### Borderline Cases (50-70% probability)

1. Review all SHAP factors carefully
2. Check underwriting ratios manually
3. Consider requesting additional documentation
4. Compare results across multiple models
5. Document reasoning for final decision

### Denial Documentation

When denying an application:

1. Note the model's probability score
2. List top negative factors from SHAP
3. Identify specific adverse action reasons
4. Ensure reasons are legally permissible
5. Generate adverse action notice

---

## Comparison Mode

### Comparing Models

To compare predictions across models:

1. Enter the same application
2. Select different models from sidebar
3. Click "Get Prediction" for each
4. Document consistency across models

### When Models Disagree

If models give different predictions:

| Scenario | Action |
|----------|--------|
| All models agree | Proceed with high confidence |
| 5-6 models agree | Follow majority |
| Models split | Manual review required |
| All disagree | Escalate to senior underwriter |

---

## Best Practices

### Do's ✅

- Review SHAP explanations for every decision
- Document model outputs in loan file
- Cross-check ratios with manual calculations
- Escalate unusual patterns to compliance
- Use multiple models for borderline cases

### Don'ts ❌

- Don't rely solely on model predictions
- Don't use protected attributes in decisions
- Don't ignore low confidence scores
- Don't skip manual review for denials
- Don't override without documentation

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `R` | Refresh dashboard |
| `C` | Clear form |
| `Enter` | Submit application |

---

## Getting Help

### Support Resources

- **Technical Issues:** Contact IT Help Desk
- **Model Questions:** compliance@company.com
- **Training:** See [FAQ](faq.md)

### Reporting Issues

If you encounter unexpected behavior:

1. Screenshot the issue
2. Note the application details (redacted)
3. Record the timestamp
4. Submit via IT ticketing system

---

## Glossary

| Term | Definition |
|------|------------|
| **AUC** | Area Under Curve - model accuracy metric |
| **DPD** | Demographic Parity Difference - fairness metric |
| **EOD** | Equalized Odds Difference - fairness metric |
| **LTV** | Loan-to-Value ratio |
| **DTI** | Debt-to-Income ratio |
| **SHAP** | SHapley Additive exPlanations - explainability method |
| **Fair Model** | ML model trained with fairness constraints |
