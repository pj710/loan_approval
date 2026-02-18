# Model Card: AI-Powered Mortgage Underwriting Assistant

**Author:** Josiah Gordor  
**Version:** 1.0.0  
**Last Updated:** February 2026

---

## Model Overview

### Model Description

The AI-Powered Mortgage Underwriting Assistant is an ensemble of fair machine learning models designed to provide consistent, fair, and accurate loan approval recommendations. The system uses multiple fairness-aware classification models trained on HMDA 2024 data.

### Available Models

| Model | Type | Description |
|-------|------|-------------|
| **XGB_Fair** | XGBoost | Gradient boosted trees with fairness constraints |
| **RF_Fair** | Random Forest | Ensemble of decision trees with balanced sampling |
| **LR_Fair** | Logistic Regression | Linear model with regularization for fairness |
| **NN_Fair** | Neural Network | Deep learning model with fairness-aware training |
| **GLM_Fair** | Generalized Linear Model | Statistical model with calibrated probabilities |
| **FasterRisk_Fair** | FasterRisk | Interpretable risk scoring model |
| **GOSDT_Fair** | GOSDT | Optimal sparse decision trees |
| **Ensemble_Fair** | Ensemble | Weighted combination of all models |

### Primary Use Case

Support human underwriters in making mortgage loan approval decisions by providing:
- Instant approval probability scores
- Risk level assessments (Low/Moderate/High)
- SHAP-based explanations for each prediction
- Fairness monitoring across protected attributes

---

## Training Data

### Data Source

- **Dataset:** Home Mortgage Disclosure Act (HMDA) 2024 Data
- **Source:** Federal Financial Institutions Examination Council (FFIEC)
- **Coverage:** NJ, NY, PA, CT (Northeast US)
- **Size:** ~2.4 million loan applications

### Data Filtering

Applications were filtered to include only:
- Owner-occupied home purchase loans
- Conventional, FHA, and VA loan types
- Complete applications (excludes withdrawn/incomplete)
- Final decisions only (approved or denied)

### Target Variable

- **Column:** `action_taken`
- **Values:** 1 = Approved (Originated), 0 = Denied
- **Class Distribution:** Approximately 95% approved, 5% denied (imbalanced)

### Feature Set

32 features used for prediction:

| Category | Features |
|----------|----------|
| **Loan Characteristics** | loan_amount, interest_rate, rate_spread, origination_charges, loan_term |
| **Property Info** | property_value, tract_population, tract_median_age_of_housing_units |
| **Income/Location** | income, ffiec_msa_md_median_family_income, tract_to_msa_income_percentage |
| **Underwriting Ratios** | loan_to_income_ratio, loan_to_value_ratio, housing_expense_ratio |
| **Risk Indicators** | combined_risk_score, dti_ltv_interaction, income_coborrower_interaction |
| **Categorical** | loan_type, construction_method, has_coborrower, state_approval_rate |

### Data Split

- **Training:** 70%
- **Validation:** 15%
- **Test:** 15%

---

## Model Performance

### Performance Metrics (Test Set)

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 Score |
|-------|----------|---------|-----------|--------|----------|
| GLM_Fair | **98.1%** | **99.5%** | **99.8%** | 98.2% | **99.0%** |
| LR_Fair | 97.7% | 98.7% | 99.7% | 97.9% | 98.8% |
| XGB_Fair | 95.2% | 84.4% | 95.2% | **99.9%** | 97.5% |
| NN_Fair | 95.2% | 60.2% | 95.2% | 100% | 97.5% |
| FasterRisk_Fair | 95.2% | 67.8% | 95.2% | 100% | 97.5% |
| GOSDT_Fair | 95.2% | 50.0% | 95.2% | 100% | 97.5% |
| RF_Fair | 77.0% | 76.3% | 97.2% | 78.0% | 86.6% |
| Ensemble_Fair | 95.6% | 98.1% | 95.6% | 100% | 97.7% |

### Performance Notes

- **Best Overall:** GLM_Fair achieves highest AUC (99.5%) and accuracy (98.1%)
- **Best Recall:** XGB_Fair, NN_Fair, FasterRisk_Fair achieve near-perfect recall
- **Most Interpretable:** GOSDT_Fair and FasterRisk_Fair provide human-readable rules

### Response Time Benchmarks

| Endpoint | Mean | Median | P95 | P99 |
|----------|------|--------|-----|-----|
| `/health` | 12ms | 9ms | 29ms | 36ms |
| `/predict` | 145ms | 125ms | 310ms | 346ms |
| `/explain` | 285ms | 264ms | 574ms | 579ms |
| `/batch/predict` (10) | 1.4s | 934ms | 9.5s | 9.9s |

**Target:** < 500ms for single predictions ✅

---

## Fairness Evaluation

### Protected Attributes Monitored

- **Race:** White, Black/African American, Asian, Other
- **Ethnicity:** Hispanic/Latino, Non-Hispanic/Latino
- **Sex:** Male, Female

### Fairness Metrics

| Metric | Definition | Threshold |
|--------|------------|-----------|
| **Demographic Parity Difference (DPD)** | Difference in approval rates across groups | < 0.05 |
| **Equalized Odds Difference (EOD)** | Difference in TPR/FPR across groups | < 0.10 |
| **Disparate Impact (DI)** | Ratio of approval rates (80% rule) | > 0.80 |

### Fairness Results by Model

#### XGB_Fair - **Best Fairness**
| Protected Attribute | DPD | EOD | Disparate Impact | Passes All |
|---------------------|-----|-----|------------------|------------|
| Race | 0.002 ✅ | 0.011 ✅ | 0.998 ✅ | ✅ |
| Ethnicity | 0.0002 ✅ | 0.0007 ✅ | 1.000 ✅ | ✅ |
| Sex | 0.0006 ✅ | 0.0007 ✅ | 0.999 ✅ | ✅ |

#### FasterRisk_Fair - **Most Interpretable + Fair**
| Protected Attribute | DPD | EOD | Disparate Impact | Passes All |
|---------------------|-----|-----|------------------|------------|
| Race | 0.0004 ✅ | 0.011 ✅ | 1.000 ✅ | ✅ |
| Ethnicity | 0.00002 ✅ | 0.0006 ✅ | 1.000 ✅ | ✅ |
| Sex | 0.0001 ✅ | 0.0015 ✅ | 1.000 ✅ | ✅ |

#### GLM_Fair - **Best Performance**
| Protected Attribute | DPD | EOD | Disparate Impact | Passes All |
|---------------------|-----|-----|------------------|------------|
| Race | 0.055 ⚠️ | 0.047 ✅ | 0.942 ✅ | ⚠️ |
| Ethnicity | 0.032 ✅ | 0.006 ✅ | 0.966 ✅ | ✅ |
| Sex | 0.008 ✅ | 0.015 ✅ | 0.991 ✅ | ✅ |

### Fairness Summary

- **Best Fairness:** XGB_Fair, FasterRisk_Fair (all metrics pass)
- **Notable Trade-off:** GLM_Fair has best performance but slightly elevated DPD for race
- **Perfect Parity:** NN_Fair, GOSDT_Fair show 0 disparity (predicts all approved)

---

## Ethical Considerations

### Intended Use

- **Decision Support:** Provide recommendations to augment human underwriters
- **Consistency:** Ensure consistent evaluation criteria across applications
- **Transparency:** Explain factors influencing each recommendation
- **Fairness Monitoring:** Track demographic disparities in real-time

### Limitations

1. **Not a Replacement for Human Judgment:** Model recommendations should be reviewed by qualified underwriters
2. **Regional Scope:** Trained on Northeast US data (NJ, NY, PA, CT); may not generalize to other regions
3. **Temporal Drift:** Model performance may degrade as economic conditions change
4. **Class Imbalance:** High approval rate in training data may bias toward approval
5. **Proxy Discrimination:** Location features may correlate with protected attributes

### Out-of-Scope Uses

- **Automated Decision-Making:** Should not be used for fully automated approvals/denials
- **Credit Scoring:** Not designed as a credit score replacement
- **Non-Mortgage Lending:** Not validated for other loan types
- **International Applications:** US-specific regulatory framework

### Bias Mitigation

1. **Pre-processing:** Removed explicit protected attributes from model features
2. **In-processing:** Applied fairness constraints during training (reweighting, adversarial debiasing)
3. **Post-processing:** Threshold calibration to balance demographic parity
4. **Monitoring:** Real-time fairness dashboard for ongoing monitoring

---

## Model Architecture

### XGBoost Fair

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=19,  # Handles class imbalance
    eval_metric='auc'
)
```

### Random Forest Fair

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1
)
```

### Neural Network Fair

```python
Sequential([
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Feature Preprocessing

- **Numerical Features:** StandardScaler normalization
- **Categorical Features:** One-hot encoding via OrdinalEncoder
- **Missing Values:** Imputation with median/mode

---

## Maintenance & Updates

### Monitoring Requirements

- **Performance Drift:** Monitor AUC weekly; retrain if drops > 5%
- **Fairness Drift:** Monitor DPD/EOD daily; alert if threshold exceeded
- **Data Quality:** Validate input distributions against training data

### Retraining Schedule

- **Quarterly:** Update with latest HMDA data
- **Event-Triggered:** Retrain if significant market changes (e.g., interest rate shifts)

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Feb 2026 | Initial release with 7 fair models |

---

## References

1. FFIEC HMDA Data Browser: https://ffiec.cfpb.gov/data-browser/
2. Fairlearn Documentation: https://fairlearn.org/
3. SHAP Documentation: https://shap.readthedocs.io/
4. Equal Credit Opportunity Act (ECOA): 15 U.S.C. §§ 1691-1691f
5. Fair Housing Act: 42 U.S.C. §§ 3601-3619

---

## Contact

**Author:** Josiah Gordor  
**Repository:** https://github.com/josiahgordor/loan_approval  
**License:** MIT
