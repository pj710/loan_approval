# Frequently Asked Questions (FAQ)

**Author:** Josiah Gordor  
**Last Updated:** February 2026

---

## General Questions

### What is the AI-Powered Mortgage Underwriting Assistant?

An ML-powered decision support tool that helps mortgage underwriters make consistent, fair, and accurate loan approval recommendations. It uses multiple fairness-aware models trained on 2024 HMDA data to provide instant predictions, risk assessments, and explanations.

### Does this system make final loan decisions?

**No.** The system provides recommendations to support human underwriters. All final decisions must be made by qualified personnel following your organization's policies and regulatory requirements.

### What data was used to train the models?

The models were trained on 2024 HMDA (Home Mortgage Disclosure Act) data from the FFIEC, covering approximately 2.4 million loan applications from NJ, NY, PA, and CT. Only owner-occupied home purchase loans with complete decisions (approved or denied) were included.

---

## Model Questions

### Which model should I use?

| Your Priority | Recommended Model |
|--------------|-------------------|
| Best overall accuracy | GLM Fair (98.1% accuracy) |
| Best fairness | XGBoost Fair (lowest DPD) |
| Most interpretable | FasterRisk Fair or GOSDT Fair |
| Balance of both | Ensemble Fair |

### Why are there multiple models?

Different models have different strengths:
- Some prioritize accuracy over interpretability
- Some achieve better fairness metrics
- Having multiple models allows comparison and consensus

### What does "Fair" in the model name mean?

"Fair" indicates the model was trained with fairness constraints to reduce bias across protected groups (race, ethnicity, sex). These models minimize discrimination while maintaining predictive accuracy.

### How accurate are the predictions?

| Model | Accuracy | AUC |
|-------|----------|-----|
| GLM Fair | 98.1% | 99.5% |
| LR Fair | 97.7% | 98.7% |
| XGB Fair | 95.2% | 84.4% |
| Ensemble Fair | 95.6% | 98.1% |

---

## Prediction Questions

### What does the probability score mean?

The probability (0-100%) indicates how likely the model believes the loan should be approved. A score of 85% means the model predicts approval with 85% confidence based on similar historical applications.

### What's the difference between probability and confidence?

- **Probability**: Likelihood of approval (0-100%)
- **Confidence**: How certain the model is about its prediction (0-100%)

A high probability with low confidence means: "This looks like an approval, but I'm not very sure."

### How are risk levels determined?

| Probability | Risk Level |
|-------------|------------|
| ≥ 80% | Low Risk |
| 50-79% | Moderate Risk |
| < 50% | High Risk |

### Why did the model recommend denial for a good application?

Possible reasons:
1. **Unusual combination** - Individual factors may look fine, but the combination is rare in training data
2. **Missing context** - Model doesn't have full credit bureau data
3. **Regional differences** - Local market conditions not captured
4. **Model limitations** - Always review with human judgment

### Can I override the model's recommendation?

Yes, and you should when you have information the model doesn't. Document your reasoning and the additional factors you considered.

---

## Fairness Questions

### How is fairness measured?

Three key metrics:

1. **Demographic Parity Difference (DPD)** - Are approval rates similar across groups?
2. **Equalized Odds Difference (EOD)** - Are error rates similar across groups?
3. **Disparate Impact (80% Rule)** - Is any group approved at less than 80% the rate of the highest group?

### What do the fairness thresholds mean?

| Metric | Threshold | Passing Means |
|--------|-----------|---------------|
| DPD | < 0.05 | Groups have ≤5% difference in approval rates |
| EOD | < 0.10 | Groups have ≤10% difference in error rates |
| Disparate Impact | > 0.80 | All groups approved at ≥80% of highest group |

### Does the model use race or ethnicity to make predictions?

**No.** Race, ethnicity, and sex are explicitly excluded from prediction features. These attributes are only used for fairness monitoring to ensure the model doesn't discriminate.

### What if the fairness dashboard shows red indicators?

Red indicators mean a fairness metric exceeds the threshold. Actions:
1. Review the specific metric and protected group
2. Consult with compliance team
3. Consider using a different model with better fairness for that attribute
4. Document any borderline decisions carefully

---

## Technical Questions

### What does "API Offline" mean?

The backend prediction service isn't running. Contact IT support or restart the API:

```bash
uvicorn src.api.main:app --port 8000
```

### Why is the dashboard slow?

Possible causes:
- First prediction after startup (model loading)
- Network latency to API server
- SHAP computation for explanations

Typical response times:
- Prediction: ~125ms
- Explanation: ~265ms

### What does "Models Loading" mean?

The API is starting up and loading model files into memory. Wait 10-30 seconds and refresh.

### Why am I getting validation errors?

Input values may be outside valid ranges:
- Loan amount must be > 0
- Interest rate must be 0-100%
- Loan term must be 1-600 months

Check the form for highlighted fields.

---

## SHAP Explanation Questions

### What are SHAP values?

SHAP (SHapley Additive exPlanations) values show how much each feature contributed to the prediction. Positive values push toward approval; negative values push toward denial.

### How do I interpret the factor list?

```
income: +0.15 → High income increased approval probability by 15%
interest_rate: -0.08 → High rate decreased approval probability by 8%
```

### Why don't the SHAP values add up to the probability?

SHAP values are relative contributions that sum to the difference between the base rate (average approval rate) and the final prediction. The base rate is around 95% for this dataset.

### What's a "base value"?

The base value is the average prediction across all training data (~0.95 or 95% for this dataset). SHAP values show deviations from this baseline.

---

## Workflow Questions

### How should I document model recommendations?

Include in your file:
1. Model name and version
2. Probability score
3. Risk level
4. Top 3-5 key factors
5. Any manual overrides and reasoning

### When should I request manual review?

- Confidence < 60%
- Models disagree
- Borderline probability (45-55%)
- Unusual factor combinations
- High-value applications

### How often is the model updated?

Models are retrained quarterly with the latest HMDA data. The dashboard shows the model version in the sidebar.

---

## Compliance Questions

### Is this system compliant with ECOA and Fair Housing Act?

The system is designed with compliance in mind:
- Protected attributes excluded from predictions
- Fairness metrics monitored in real-time
- All models trained with bias mitigation
- Full audit trail in prediction logs

However, final compliance responsibility rests with your organization.

### Can I use model predictions as adverse action reasons?

You can use the SHAP factors to support adverse action reasons, but ensure:
1. Reasons are specific and factual
2. They relate to creditworthiness factors
3. They don't reference protected characteristics
4. They comply with your institution's policies

### Where are prediction logs stored?

API logs are stored in the server logs. For audit purposes, ensure your organization has logging and retention policies in place.

---

## Troubleshooting

### The prediction seems wrong. What should I do?

1. Verify input data is correct
2. Check if the application is unusual (very high/low values)
3. Try a different model for comparison
4. Use your professional judgment
5. Document your reasoning if overriding

### I'm getting different results than a colleague.

Possible causes:
- Different models selected
- Slightly different input values
- Model update between predictions

Ensure you're using the same model and inputting identical data.

### The dashboard crashed. What do I do?

1. Refresh the browser (F5)
2. Clear browser cache
3. Check API status in sidebar
4. Contact IT if issue persists

---

## Contact

**Technical Support:** IT Help Desk  
**Model Questions:** compliance@company.com  
**Author:** Josiah Gordor
