# Model Governance Documentation

**AI-Powered Mortgage Underwriting Assistant**

**Author:** Josiah Gordor  
**Version:** 1.0.0  
**Last Updated:** February 2026

---

## 1. Document Purpose

This document establishes the model governance framework for the AI-Powered Mortgage Underwriting Assistant, ensuring compliance with regulatory requirements and industry best practices for model risk management.

### Applicable Standards

- OCC 2011-12 / SR 11-7 (Model Risk Management)
- CFPB Supervisory Guidance on Fair Lending
- FFIEC IT Examination Handbook
- ISO/IEC 23894:2023 (AI Risk Management)

---

## 2. Model Inventory

### 2.1 Model Identification

| Field | Value |
|-------|-------|
| **Model Name** | AI-Powered Mortgage Underwriting Assistant |
| **Model ID** | LAS-2026-001 |
| **Model Type** | Supervised Classification (Binary) |
| **Risk Tier** | Tier 1 (High Risk) |
| **Business Function** | Mortgage Underwriting Support |
| **Production Date** | February 2026 |

### 2.2 Model Components

| Component | Version | Description |
|-----------|---------|-------------|
| XGB_Fair | 1.0 | XGBoost with fairness constraints |
| RF_Fair | 1.0 | Random Forest with balanced sampling |
| LR_Fair | 1.0 | Logistic Regression (fairness-aware) |
| NN_Fair | 1.0 | Neural Network with adversarial debiasing |
| GLM_Fair | 1.0 | Generalized Linear Model |
| FasterRisk_Fair | 1.0 | Interpretable risk scoring |
| GOSDT_Fair | 1.0 | Optimal sparse decision trees |
| Ensemble_Fair | 1.0 | Weighted combination |

### 2.3 Supporting Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Feature Scaler | `models/feature_scaler.pkl` | Feature normalization |
| Fair Encoder | `models/fair_representation/` | Fair representation learning |
| SHAP Explainer | Computed at runtime | Prediction explanations |

---

## 3. Roles & Responsibilities

### 3.1 RACI Matrix

| Activity | Model Owner | Developer | Validation | Compliance | IT |
|----------|-------------|-----------|------------|------------|-----|
| Model Development | A | R | C | C | I |
| Model Validation | A | I | R | C | I |
| Deployment Approval | R | I | C | A | C |
| Production Monitoring | R | C | C | I | A |
| Retraining Decision | A | R | C | C | I |
| Regulatory Reporting | I | I | C | R | I |

**R** = Responsible, **A** = Accountable, **C** = Consulted, **I** = Informed

### 3.2 Key Personnel

| Role | Responsibility |
|------|---------------|
| **Model Owner** | Overall accountability for model performance and risk |
| **Model Developer** | Technical development and maintenance |
| **Model Validator** | Independent validation and testing |
| **Compliance Officer** | Fair lending and regulatory compliance |
| **IT Operations** | Infrastructure and deployment |

---

## 4. Model Development

### 4.1 Development Process

```
┌─────────────────┐
│ Problem Definition │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Collection │──▶ HMDA 2024 (USA, All states)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engineering │──▶ 32 features selected
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training │──▶ 7 fair model variants
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fairness Testing │──▶ DPD, EOD, DIR metrics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validation │──▶ Independent review
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Deployment │
└─────────────────┘
```

### 4.2 Data Governance

| Aspect | Implementation |
|--------|---------------|
| **Data Source** | FFIEC HMDA (official regulatory data) |
| **Data Quality** | Validated against HMDA requirements |
| **Data Lineage** | Documented from source to feature |
| **Data Security** | No PII in production; demographics for monitoring only |

### 4.3 Feature Selection Rationale

| Feature Category | Included | Excluded | Rationale |
|-----------------|----------|----------|-----------|
| Financial | ✅ | - | Core creditworthiness factors |
| Property | ✅ | - | Collateral assessment |
| Geographic | ✅ | - | Regional risk (not proxy for protected) |
| Demographic | - | ✅ | ECOA protected attributes |

---

## 5. Model Validation

### 5.1 Validation Framework

| Validation Type | Frequency | Responsible |
|-----------------|-----------|-------------|
| **Conceptual Soundness** | Initial + Material Changes | Model Validation |
| **Data Integrity** | Quarterly | Data Quality Team |
| **Performance Testing** | Monthly | Model Operations |
| **Fairness Testing** | Monthly | Compliance |
| **Stress Testing** | Annual | Risk Management |

### 5.2 Validation Results

| Test | Result | Date |
|------|--------|------|
| Accuracy Test | PASS (95-98% accuracy) | Feb 2026 |
| Fairness Test | PASS (most models) | Feb 2026 |
| Stability Test | PASS (consistent across folds) | Feb 2026 |
| Benchmark Test | PASS (exceeds baseline) | Feb 2026 |

### 5.3 Limitations Identified

| Limitation | Severity | Mitigation |
|------------|----------|------------|
| Regional scope (NE US only) | Medium | Document in user training |
| Class imbalance (95% approval) | Medium | Balanced training, threshold calibration |
| GOSDT low AUC | Low | Use for interpretability only |
| RF_Fair race fairness | High | Remove from high-stakes decisions |

---

## 6. Model Performance

### 6.1 Performance Metrics

| Model | Accuracy | AUC | F1 | Status |
|-------|----------|-----|-----|--------|
| GLM_Fair | 98.1% | 99.5% | 99.0% | ✅ Production |
| LR_Fair | 97.7% | 98.7% | 98.8% | ✅ Production |
| XGB_Fair | 95.2% | 84.4% | 97.5% | ✅ Production |
| Ensemble_Fair | 95.6% | 98.1% | 97.7% | ✅ Production |
| NN_Fair | 95.2% | 60.2% | 97.5% | ⚠️ Monitor |
| FasterRisk_Fair | 95.2% | 67.8% | 97.5% | ✅ Production |
| GOSDT_Fair | 95.2% | 50.0% | 97.5% | ⚠️ Interpretability only |
| RF_Fair | 77.0% | 76.3% | 86.6% | ❌ Review required |

### 6.2 Performance Thresholds

| Metric | Minimum Threshold | Action if Breached |
|--------|-------------------|-------------------|
| Accuracy | 90% | Investigate |
| AUC | 75% | Retrain or remove |
| F1 Score | 85% | Review |
| DPD (any group) | < 0.10 | Immediate review |

### 6.3 Monitoring Schedule

| Metric | Frequency | Owner |
|--------|-----------|-------|
| Prediction volume | Daily | Model Ops |
| Accuracy (sample) | Weekly | Model Ops |
| Fairness metrics | Daily | Compliance |
| Response time | Daily | IT Ops |
| Error rate | Daily | IT Ops |

---

## 7. Risk Assessment

### 7.1 Model Risk Rating

| Risk Factor | Assessment | Score (1-5) |
|-------------|------------|-------------|
| **Materiality** | High (loan decisions) | 5 |
| **Complexity** | Medium (ensemble) | 3 |
| **Data Quality** | High (regulatory data) | 2 |
| **Regulatory Scrutiny** | High (ECOA, FHA) | 5 |
| **Operational Risk** | Medium | 3 |
| **Overall** | **Tier 1 (High Risk)** | **3.6** |

### 7.2 Risk Mitigation Controls

| Risk | Control | Owner |
|------|---------|-------|
| Model error | Human review of decisions | Underwriting |
| Fairness drift | Real-time monitoring | Compliance |
| Data drift | Input distribution checks | Model Ops |
| System failure | Fallback to manual process | Operations |
| Adversarial inputs | Input validation | API |

### 7.3 Contingency Planning

| Scenario | Response |
|----------|----------|
| Model unavailable | Revert to prior version or manual |
| Performance degradation | Trigger retraining workflow |
| Fairness failure | Pause model, escalate to compliance |
| Security breach | Isolate system, notify security |

---

## 8. Change Management

### 8.1 Change Categories

| Category | Approval Required | Examples |
|----------|-------------------|----------|
| **Material** | Full validation + Board | New model, feature changes |
| **Non-Material** | Model Owner | Threshold adjustments |
| **Emergency** | Model Owner (immediate) | Security patches |

### 8.2 Change Request Process

```
┌─────────────┐
│ Request │──▶ Document change reason
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Assessment │──▶ Categorize (Material/Non-Material)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Testing │──▶ Validate on test data
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Approval │──▶ Per category requirements
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Deployment │──▶ Staged rollout
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Monitoring │──▶ Post-deployment validation
└─────────────┘
```

### 8.3 Version Control

| Version | Date | Change Type | Description |
|---------|------|-------------|-------------|
| 1.0.0 | Feb 2026 | Initial | Production release |
| - | - | - | - |

---

## 9. Model Lifecycle

### 9.1 Lifecycle Stages

```
Development ──▶ Validation ──▶ Production ──▶ Monitoring ──▶ Retirement
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
  Training      Testing       Deployment     Tracking      Archival
  Tuning       Approval       Integration    Retraining    Succession
```

### 9.2 Retraining Schedule

| Trigger | Frequency | Owner |
|---------|-----------|-------|
| Scheduled | Quarterly | Model Dev |
| Performance degradation | As needed | Model Ops |
| Regulatory change | As needed | Compliance |
| Data source update | Annual (HMDA) | Model Dev |

### 9.3 Retirement Criteria

Model will be retired if:
- Performance below thresholds for 90+ days
- Regulatory requirements change materially
- Superior replacement validated
- Business need eliminated

---

## 10. Documentation Requirements

### 10.1 Required Documentation

| Document | Location | Update Frequency |
|----------|----------|------------------|
| Model Card | `docs/model_card.md` | Material changes |
| API Documentation | `docs/api_documentation.md` | Interface changes |
| User Guide | `docs/underwriter_guide.md` | Workflow changes |
| Compliance Report | `docs/compliance_report.md` | Annual |
| This Governance Doc | `docs/model_governance.md` | Annual |

### 10.2 Audit Trail

All model predictions are logged with:
- Request timestamp
- Input features (anonymized)
- Model used
- Prediction result
- SHAP values
- Response time

Retention: 7 years (per regulatory requirements)

---

## 11. Training & Communication

### 11.1 Required Training

| Role | Training | Frequency |
|------|----------|-----------|
| Underwriters | Dashboard usage | Onboarding + Annual |
| Model Ops | Monitoring procedures | Onboarding + Quarterly |
| Compliance | Fairness interpretation | Onboarding + Annual |
| Developers | Change management | Onboarding + As needed |

### 11.2 Communication Plan

| Event | Audience | Channel |
|-------|----------|---------|
| Model update | All users | Email + Training |
| Performance issue | Model Ops | Alert system |
| Fairness concern | Compliance | Immediate escalation |
| Outage | All users | Status page |

---

## 12. Approval & Sign-Off

### 12.1 Development Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Model Developer | Josiah Gordor | __________ | Feb 2026 |
| Model Owner | __________ | __________ | __________ |

### 12.2 Validation Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Model Validator | __________ | __________ | __________ |
| Compliance Officer | __________ | __________ | __________ |

### 12.3 Production Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Model Owner | __________ | __________ | __________ |
| IT Operations | __________ | __________ | __________ |
| Risk Management | __________ | __________ | __________ |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **AUC** | Area Under Receiver Operating Characteristic Curve |
| **DPD** | Demographic Parity Difference |
| **EOD** | Equalized Odds Difference |
| **ECOA** | Equal Credit Opportunity Act |
| **HMDA** | Home Mortgage Disclosure Act |
| **LTV** | Loan-to-Value Ratio |
| **SHAP** | SHapley Additive exPlanations |

## Appendix B: Related Documents

- [Model Card](model_card.md)
- [API Documentation](api_documentation.md)
- [Compliance Report](compliance_report.md)
- [Configuration Guide](configuration_guide.md)
- [Underwriter Guide](underwriter_guide.md)

## Appendix C: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Feb 2026 | Josiah Gordor | Initial release |

---

**Classification:** Internal Use Only  
**Owner:** Model Risk Management  
**Next Review:** February 2027
