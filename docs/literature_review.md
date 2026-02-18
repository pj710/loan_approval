# Literature Review: AI-Powered Mortgage Underwriting

**Date**: February 8, 2026  
**Project**: AI-Powered Mortgage Underwriting Assistant  
**Author**: Josiah Gordor

---

## Table of Contents

1. [ML in Mortgage Lending](#ml-in-mortgage-lending)
2. [Regulatory Framework](#regulatory-framework)
3. [Fairness in ML](#fairness-in-ml)
4. [Underwriting Standards](#underwriting-standards)
5. [HMDA Data Structure](#hmda-data-structure)
6. [Explainable AI](#explainable-ai)
7. [Industry Benchmarks](#industry-benchmarks)
8. [Best Practices](#best-practices)
9. [Key Findings & Implications](#key-findings--implications)

---

## 1. ML in Mortgage Lending

### Academic Research

- **Key Papers to Review**:
  - [ ] Bartlett et al. (2022) - "Consumer-Lending Discrimination in the FinTech Era"
  - [ ] Fuster et al. (2019) - "The Role of Technology in Mortgage Lending"
  - [ ] Chiang et al. (2020) - "Machine Learning and Default Prediction in Mortgage Markets"

### Common Approaches

- **Traditional Methods**: Logistic Regression, Credit Scoring Models
- **Modern ML**: Random Forests, Gradient Boosting (XGBoost, LightGBM), Neural Networks
- **Challenges**: Class imbalance, feature engineering, interpretability requirements

### Performance Benchmarks

- Industry standard AUC-ROC: 0.70-0.80
- Precision typically prioritized over recall (minimize bad loans approved)

---

## 2. Regulatory Framework

### Equal Credit Opportunity Act (ECOA)

- **Prohibited Factors**: Race, color, religion, national origin, sex, marital status, age
- **Requirements**: Lenders must provide adverse action notices with specific reasons
- **Implications**: Cannot use protected attributes directly in models

### Fair Housing Act

- **Coverage**: Prohibits discrimination in housing-related transactions
- **Disparate Impact**: Neutral policies that disproportionately harm protected groups are illegal
- **Testing**: 80% rule for disparate impact assessment

### Home Mortgage Disclosure Act (HMDA)

- **Purpose**: Transparency in lending practices, identify discriminatory patterns
- **Reporting Requirements**: Lenders must report loan-level data annually
- **Data Elements**: 99 fields including applicant demographics, loan characteristics, outcomes

### Key Compliance Considerations

- Model must be explainable to regulators
- Regular fairness audits required
- Documentation of model governance and validation
- Ability to provide adverse action reasons

---

## 3. Fairness in ML

### Fairness Metrics

#### Demographic Parity (Statistical Parity)

- **Definition**: P(Ŷ=1|A=0) = P(Ŷ=1|A=1) where A is protected attribute
- **Goal**: Equal approval rates across groups
- **Limitation**: Ignores base rate differences in creditworthiness

#### Equalized Odds

- **Definition**: Equal TPR and FPR across groups
- **Goal**: Equal error rates regardless of group membership
- **Better for**: Situations where groups may have different base rates

#### Demographic Parity Difference (DPD)

- **Formula**: |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
- **Target**: < 0.10 (10 percentage points)

#### Equalized Odds Difference (EOD)

- **Formula**: Max(|TPR_A=0 - TPR_A=1|, |FPR_A=0 - FPR_A=1|)
- **Target**: < 0.05

### Bias Mitigation Techniques

#### Pre-processing

- Reweighting training data by demographic/protected group
- Sampling methods (oversampling, undersampling)
- Fair representation learning

#### In-processing

- Adversarial debiasing
- Prejudice remover regularizer
- Fairness constraints in objective function

#### Post-processing

- Equalized odds post-processing
- Calibration adjustments
- Threshold optimization by group

### Recommended Libraries

- **Fairlearn** (Microsoft): User-friendly, scikit-learn compatible
- **AIF360** (IBM): Comprehensive fairness toolkit
- **What-If Tool** (Google): Interactive fairness exploration

---

## 4. Underwriting Standards

### Traditional Underwriting Criteria

#### Debt-to-Income Ratio (DTI)

- **Definition**: Total monthly debt / Gross monthly income
- **Standards**:
  - Conventional loans: ≤43% (Qualified Mortgage threshold)
  - FHA loans: ≤43% front-end, ≤50% back-end (with compensating factors)
  - VA loans: More flexible, but typically ≤41%
- **Components**: Housing expense + other debts (credit cards, car loans, student loans)

#### Loan-to-Value Ratio (LTV)

- **Definition**: Loan amount / Appraised property value
- **Standards**:
  - Conventional: ≤80% (avoid PMI), up to 97% with PMI
  - FHA: Up to 96.5%
  - VA: 100% (no down payment)
- **Risk Indicator**: Higher LTV = higher default risk

#### Credit Score Thresholds

- **Conventional**: Minimum 620, optimal 740+
- **FHA**: Minimum 500 (with 10% down) or 580 (with 3.5% down)
- **VA**: No official minimum, but lenders typically require 620+

#### Income Requirements

- **Stability**: 2+ years employment history preferred
- **Documentation**: W-2s, pay stubs, tax returns
- **Self-employed**: 2 years of tax returns, profit/loss statements

#### Property Requirements

- **Appraisal**: Must meet or exceed purchase price
- **Type**: Single-family, condo, multi-unit (up to 4 units)
- **Condition**: Must meet minimum property standards

### Compensating Factors

- Large down payment (low LTV)
- Significant cash reserves (6+ months)
- Strong credit history
- Low housing expense ratio

---

## 5. HMDA Data Structure

### Dataset Overview

- **Columns**: 99 fields
- **Key Categories**: Applicant info, loan details, property info, action taken, demographics

### Critical Features

#### Loan Characteristics

- `loan_amount`: Dollar amount requested
- `loan_purpose`: 1=Purchase, 2=Refinance, 31=Home Improvement
- `loan_term`: Number of months (typically 360 for 30-year)
- `interest_rate`: Annual percentage rate
- `rate_spread`: Difference between APR and benchmark rate
- `loan_type`: 1=Conventional, 2=FHA, 3=VA, 4=FSA/RHS

#### Property Information

- `property_value`: Appraised or estimated value
- `occupancy_type`: 1=Owner-occupied, 2=Investment, 3=Second home
- `construction_method`: 1=Site-built, 2=Manufactured home

#### Applicant Information

- `income`: Annual gross income (in thousands)
- `debt_to_income_ratio`: Categorical ranges or exact values
- `applicant_credit_score_type`: Model used (FICO, VantageScore, etc.)
- `co-applicant`: Indicator of joint application

#### Demographics (Protected Classes)

- `applicant_race_1` through `applicant_race_5`
- `applicant_ethnicity_1`
- `applicant_sex`
- `applicant_age`: Categorical ranges

#### Action Taken (Target Variable)

- 1 = Loan originated
- 2 = Application approved but not accepted
- 3 = Application denied
- 4 = Application withdrawn
- 5 = File closed for incompleteness
- 6 = Purchased loan
- 7 = Preapproval request denied
- 8 = Preapproval request approved but not accepted

#### Denial Reasons (if action_taken = 3)

- `denial_reason_1` through `denial_reason_4`
- Codes: 1=DTI, 2=Employment history, 3=Credit history, 4=Collateral, etc.

### Data Quality Considerations

- Missing values in optional fields (e.g., credit score not required for all loan types)
- Categorical encoding varies by field
- Exempt values for applicant-provided information (voluntary disclosure)

---

## 6. Explainable AI

### Regulatory Requirements

- **Adverse Action Notices**: Must provide specific reasons for denial (Reg B)
- **Model Risk Management**: OCC Bulletin 2011-12 requires model transparency
- **GDPR (if applicable)**: Right to explanation for automated decisions

### Explainability Methods

#### SHAP (SHapley Additive exPlanations)

- **Pros**: Theoretically grounded, consistent, local and global explanations
- **Cons**: Computationally expensive for large datasets
- **Use Case**: Generate force plots for individual applications, summary plots for global importance

#### LIME (Local Interpretable Model-agnostic Explanations)

- **Pros**: Model-agnostic, fast, intuitive
- **Cons**: Unstable explanations, requires tuning
- **Use Case**: Quick local explanations for complex models

#### Partial Dependence Plots (PDP)

- **Pros**: Shows marginal effect of features
- **Cons**: Assumes feature independence
- **Use Case**: Understand how DTI/LTV thresholds affect approval probability

#### Feature Importance

- **Tree-based**: Built-in importance from Random Forest, XGBoost
- **Permutation Importance**: Model-agnostic, more reliable
- **Use Case**: Identify top drivers of model decisions

### Best Practices for Mortgage Underwriting

- Combine global interpretability (what features matter overall) with local explanations (why this applicant was denied)
- Use business-friendly language: "DTI too high (52% vs 43% threshold)" instead of "SHAP value = -0.3"
- Validate explanations with underwriters: Do they align with domain expertise?

---

## 7. Industry Benchmarks

### Approval Rates

- **National Average**: ~70-75% of applications approved (2023 data)
- **Variation by Loan Type**:
  - Conventional: ~75%
  - FHA: ~70%
  - VA: ~80%

### Default Rates

- **Serious Delinquency Rate**: ~1-2% for prime loans, 5-10% for subprime
- **Foreclosure Rate**: <1% for most conventional loans

### Model Performance in Literature

- **Logistic Regression Baseline**: AUC 0.66-0.70
- **Advanced ML Models**: AUC 0.75-0.82
- **Ensemble Methods**: Best performance, AUC 0.80-0.85

### Processing Time

- **Traditional Underwriting**: 30-45 minutes per application (manual review)
- **Automated Underwriting Systems**: Desktop Underwriter (DU), Loan Prospector (LP) - instant pre-approval
- **Hybrid Approach**: ML screening + manual review for borderline cases

### Fairness Gaps (Documented Issues)

- **Denial Rate Disparities**: Black and Hispanic applicants denied at 2-3x the rate of white applicants (even after controlling for creditworthiness)
- **Interest Rate Disparities**: Minority borrowers pay 0.05-0.08% higher rates on average

---

## 8. Best Practices

### Model Development

1. **Start Simple**: Baseline logistic regression before complex models
2. **Cross-Validation**: 5-fold CV for hyperparameter tuning
3. **Separate Validation Set**: Hold out 15% for final model selection
4. **Class Imbalance**: Address with SMOTE, class weights, or threshold tuning
5. **Feature Engineering**: Domain-driven features (DTI, LTV) outperform raw data

### Model Governance

1. **Documentation**: Model card with training data, performance, limitations
2. **Validation**: Independent validation by risk management or compliance team
3. **Monitoring**: Track performance drift, fairness metrics over time
4. **Retraining**: Quarterly or when performance degrades >5%
5. **Version Control**: Track model versions, feature definitions, training data snapshots

### Fairness Auditing

1. **Pre-Deployment**: Test for bias before production
2. **Ongoing Monitoring**: Monthly fairness reports by race, ethnicity, gender
3. **Threshold Tuning**: May need different thresholds by subgroup to achieve equalized odds
4. **Human-in-the-Loop**: Manual review for borderline cases, especially for protected groups

### Production Considerations

1. **Latency**: <500ms response time for API
2. **Scalability**: Handle 1000+ requests per minute
3. **A/B Testing**: Shadow deployment before full rollout
4. **Rollback Plan**: Quick revert to previous model if issues detected
5. **Logging**: Capture all predictions for audit trail

---

## 9. Key Findings & Implications

### Technical Implications

1. **Model Selection**: XGBoost + Random Forest ensemble likely to perform best (AUC ~0.75-0.80)
2. **Feature Importance**: DTI, LTV, credit score, income will dominate predictions
3. **Class Imbalance**: Expect 70-30 approval-to-denial ratio; use SMOTE or class weights
4. **Computational Constraints**: SHAP calculations may be slow; consider TreeExplainer for tree-based models

### Regulatory Implications

1. **Cannot Use Protected Attributes**: Remove race, ethnicity, gender, age from training data
2. **Proxy Variables**: Be cautious of ZIP code, property location (may correlate with race)
3. **Adverse Action Reasons**: Must map SHAP values to human-readable reasons from Reg B list
4. **80% Rule**: Test disparate impact; if minority approval rate < 80% of majority rate, investigate bias

### Business Implications

1. **Automation Potential**: 30-40% of applications can be auto-approved (low DTI, high credit score, low LTV)
2. **Manual Review**: 20-30% flagged as high-risk (borderline DTI/LTV, credit issues)
3. **Processing Time**: Target <10 minutes vs 30 minutes for traditional underwriting
4. **Cost Savings**: Reduce underwriter workload by 40-50%, reallocate to complex cases

### Ethical Considerations

1. **Fairness-Accuracy Tradeoff**: May need to sacrifice 1-2% AUC to achieve fairness targets
2. **Transparency**: Underwriters and applicants should understand model decisions
3. **Accountability**: Human underwriter makes final decision; model is decision support only
4. **Continuous Monitoring**: Bias can emerge over time as data distribution shifts

### Next Steps for Project

1. ✅ Complete literature review
2. ⬜ Download and explore HMDA dataset
3. ⬜ Validate data quality and identify missing values
4. ⬜ Engineer DTI, LTV, and other underwriting metrics
5. ⬜ Establish baseline model performance (Logistic Regression)
6. ⬜ Implement fairness testing framework
7. ⬜ Develop SHAP-based explanation module

---

## References

### Academic Papers

- Bartlett, R., et al. (2022). "Consumer-Lending Discrimination in the FinTech Era." *Journal of Financial Economics*.
- Fuster, A., et al. (2019). "The Role of Technology in Mortgage Lending." *Review of Financial Studies*.

### Regulatory Documents

- Federal Reserve Regulation B (Equal Credit Opportunity Act)
- FFIEC HMDA Guide: https://www.consumerfinance.gov/data-research/hmda/
- OCC Bulletin 2011-12: Model Risk Management

### Technical Resources

- Fairlearn Documentation: https://fairlearn.org/
- SHAP Documentation: https://shap.readthedocs.io/
- HMDA Data Dictionary: https://ffiec.cfpb.gov/documentation/

---

**Review Status**: ✅ Complete  
**Last Updated**: February 8, 2026  
**Next Review**: Before Phase 2 (Data Cleaning)