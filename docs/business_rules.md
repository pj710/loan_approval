# Loan Approval Business Rules

**Project**: AI-Powered Mortgage Underwriting Assistant  
**Date**: 2026-02-11  
**Dataset**: HMDA 2024 (NJ, NY, PA, CT)  

---

## Target Variable Definition

The target variable for our binary classification model is derived from the HMDA `action_taken` field.

### Target = 1 (Approved)
Applications that received approval from the lender:

1. **Loan originated**: The application was approved and the loan was made by the institution
2. **Application approved but not accepted**: The lender approved the application, but the applicant chose not to accept the offer
3. **Purchased loan**: The loan was purchased by the institution (indicates it was approved elsewhere and deemed creditworthy)

**Business Rationale**: These three categories represent positive underwriting decisions where the lender determined the applicant met credit standards.

### Target = 0 (Denied)
Applications that were rejected by the lender:

1. **Application denied**: The lender denied the application based on underwriting criteria

**Business Rationale**: This represents a negative underwriting decision where the applicant did not meet the lender's credit standards.

### Excluded (Not Used for Modeling)
Applications that did not result in a final underwriting decision:

1. **Application withdrawn by applicant**: The applicant withdrew the application before a decision was made
2. **File closed for incompleteness**: The application was incomplete and no decision could be made
3. **Preapproval request denied**: Preapproval stage decision (not final approval)
4. **Preapproval request approved but not accepted**: Preapproval stage decision (not final approval)

**Business Rationale**: These categories are excluded because:
- They don't represent actual underwriting decisions
- They reflect applicant behavior (withdrawal) rather than lender assessment
- Incomplete applications lack the data needed for proper evaluation
- Preapproval is a preliminary step, not a final approval decision

---

## Dataset Statistics

### Overall Distribution
- **Total Records**: 2,000,000
- **Approved (Target=1)**: 1,223,894 (61.19%)
- **Denied (Target=0)**: 379,971 (19.00%)
- **Excluded**: 396,135 (19.81%)

### Modeling Dataset
- **Total Records for Modeling**: 1,603,865
- **Approval Rate**: 76.31%
- **Denial Rate**: 23.69%
- **Class Imbalance Ratio**: 3.22:1 (Approved:Denied)

---

## Underwriting Criteria (To Be Implemented)

The model will learn to predict approval/denial based on these key underwriting factors:

### Financial Metrics
1. **Debt-to-Income Ratio (DTI)**: Total monthly debt / Gross monthly income
   - Industry Standard: DTI ≤ 43% for qualified mortgages
   - Higher DTI = Higher risk of default

2. **Loan-to-Value Ratio (LTV)**: Loan amount / Property appraised value
   - Industry Standard: LTV ≤ 80% for conventional loans without PMI
   - Higher LTV = Higher risk (less borrower equity)

3. **Income**: Gross annual income of applicant(s)
   - Must be sufficient to support mortgage payment + other debts
   - Verified through pay stubs, tax returns, bank statements

### Loan Characteristics
4. **Loan Amount**: Total mortgage amount requested
5. **Interest Rate**: Annual percentage rate (APR)
6. **Loan Term**: Typically 15 or 30 years
7. **Loan Type**: Conventional, FHA, VA, USDA
8. **Loan Purpose**: Purchase, Refinance, Home Improvement

### Property Information
9. **Property Value**: Appraised value of the property
10. **Property Type**: Single-family, Condo, Multi-family, etc.
11. **Occupancy Type**: Owner-occupied, Investment, Second home
12. **Location**: State, County, Census Tract (affects local market conditions)

### Credit Profile
13. **Credit Score**: FICO score (if available in dataset)
14. **Rate Spread**: Difference between APR and benchmark rate (proxy for risk tier)

---

## Fairness Considerations

### Protected Attributes (Monitored but NOT Used as Features)
Per ECOA (Equal Credit Opportunity Act) and Fair Housing Act, the following cannot be used for credit decisions:
- Race/Ethnicity
- Sex/Gender
- Age (except to verify legal capacity)
- Marital Status
- Religion
- National Origin

**Our Approach**:
1. These attributes will **NOT** be included as model features
2. They will be used for **post-hoc fairness auditing** to detect disparate impact
3. We will monitor for proxy discrimination through correlated features

### Fairness Metrics to Monitor
1. **Demographic Parity Difference (DPD)**: Difference in approval rates between groups
   - Target: |DPD| < 0.05 (5 percentage points)

2. **Equalized Odds Difference (EOD)**: Difference in true positive rates and false positive rates
   - Target: |EOD| < 0.05

3. **Disparate Impact Ratio**: (Approval rate for protected group) / (Approval rate for reference group)
   - Legal Standard: Ratio ≥ 0.80 (80% rule)

---

## Model Governance

### Explainability Requirements
- All approval/denial decisions must be explainable using SHAP values
- Top 5 factors influencing each decision will be displayed
- Human underwriters can review and override model recommendations

### Model Monitoring
- Monthly fairness audits across all protected classes
- Quarterly model performance reviews (AUC, Precision, Recall)
- Annual model retraining with updated data
- Immediate alert system for fairness metric violations

### Documentation
- Model card documenting training data, features, performance, limitations
- Audit trail for all model predictions and explanations
- Regular reports to compliance and risk management teams

---

## Next Steps

1. **Phase 2**: Data cleaning and preprocessing
2. **Phase 3**: Feature engineering (calculate DTI, LTV, etc.)
3. **Phase 4**: Exploratory data analysis
4. **Phase 5**: Model training and hyperparameter tuning
5. **Phase 6**: Fairness analysis and bias mitigation
6. **Phase 7**: Model evaluation and selection

---

*This document will be updated as the project progresses and new insights are discovered.*
