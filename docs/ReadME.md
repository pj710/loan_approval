# AI-Powered Mortgage Underwriting Assistant

> An intelligent ML-based decision support system that helps mortgage lenders make consistent, fair, and accurate loan approval decisions using HMDA data.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

### Context & Motivation

Homeownership represents a cornerstone of the American dream—the aspiration to achieve success and build wealth through years of hard work. Mortgages enable millions of Americans to realize this dream by providing financing to creditworthy applicants for purchasing homes as primary residences or as investment vehicles for wealth creation.

### The Challenge: Complexity & Inefficiency in Traditional Underwriting

The mortgage industry operates as a highly regulated, data-intensive enterprise requiring rigorous underwriting processes to determine loan approval decisions. Current challenges include:

- **Process Duration**: The approval process spans multiple stages—from initial application through underwriting to closing—often taking days to months to complete
- **Regulatory Burden**: Lenders must ensure strict compliance with requirements from federal regulators and investors at every stage, facing substantial penalties for violations
- **Product Complexity**: The diverse array of mortgage products with varying requirements makes manual review time-consuming and cognitively demanding
- **Human Bias Risk**: Manual underwriting processes are susceptible to unconscious biases that can lead to discriminatory lending practices

### The Opportunity: Machine Learning for Automated Decision Support

Machine learning offers transformative capabilities for modernizing the mortgage approval process through intelligent automation:

- **Speed**: Automated analytics accelerate application review, providing real-time recommendations for each submission
- **Consistency**: ML models apply uniform evaluation criteria across all applications, reducing variability in decisions
- **Risk Detection**: Algorithms can flag potential issues or red flags for manual review by experienced underwriters
- **Predictive Power**: Supervised classifiers generate probability scores based on comprehensive analysis of applicant information, loan terms, and property details
- **Transparency**: Feature importance analysis reveals which factors drive approval decisions, supporting underwriter judgment and regulatory compliance

### The Critical Issue: Algorithmic Fairness

Despite their operational benefits, ML models pose significant risks if deployed without rigorous fairness safeguards:

- **Bias Amplification**: Models trained on historical data may learn and perpetuate discriminatory patterns embedded in past lending decisions
- **Compliance Risk**: Biased algorithmic decisions expose lenders to violations of fair lending laws (Equal Credit Opportunity Act, Fair Housing Act)
- **Reputational Harm**: Discriminatory outcomes can damage institutional reputation and erode community trust

### Project Focus: Fair & Explainable Mortgage Underwriting

This project addresses the algorithmic fairness challenge by implementing **fair learning techniques** to develop equitable ML models for mortgage approval. The system is designed to:

1. **Detect Bias**: Evaluate model performance across protected classes (race, ethnicity, age, gender)
2. **Mitigate Disparate Impact**: Apply fairness constraints during training to reduce discriminatory outcomes
3. **Monitor Continuously**: Track fairness metrics (Demographic Parity Difference, Equalized Odds) in production
4. **Explain Decisions**: Provide transparent, auditable explanations for each approval recommendation using SHAP values

By combining predictive accuracy with fairness guarantees, this system enables lenders to accelerate processing while ensuring equitable access to homeownership opportunities.

## Solution Architecture

### Performance Targets

| Objective | Target Metric | Goal |
|-----------|---------------|------|
| **Predictive Accuracy** | AUC-ROC | ≥ 0.75 |
| **Balanced Performance** | F1 Score | ≥ 0.80 |
| **Processing Speed** | Review Time | < 10 minutes per application |
| **Fairness (Race)** | Demographic Parity Difference | < 0.05 |
| **Fairness (Outcomes)** | Equalized Odds Difference | < 0.05 |
| **API Response** | Latency | < 500ms |

### Key Capabilities

- **Multi-Model Ensemble**: Combines Random Forest, XGBoost, and TabNet classifiers with calibrated probability outputs
- **Automated Underwriting Metrics**: Calculates debt-to-income ratio (DTI), loan-to-value ratio (LTV), and income verification ratios
- **Continuous Fairness Monitoring**: Real-time tracking of Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD) across protected groups
- **Model Explainability**: SHAP force plots reveal feature contributions for each prediction, supporting human review
- **Interactive Dashboard**: Streamlit web interface displays approval confidence, risk scores, and flagged applications
- **Production-Ready API**: FastAPI backend with Pydantic validation for seamless integration into existing workflows

## Technical Approach

### Machine Learning Task

**Problem Type**: Supervised binary classification

**Target Variable**: Loan approval decision (Binary: 1 = Approved, 0 = Denied)

**Model Output**: 
- Approval probability score (0.0 to 1.0)
- Binary classification (Approved/Denied)
- Feature attribution (SHAP values)

### Data

**Source**: 2024 Home Mortgage Disclosure Act (HMDA) dataset  
**Scale**: ~2.4M applications across all U.S. states  
**Scope**: Owner-occupied home purchase loans (excludes refinances and investment properties)

**Input Features** (15-20 variables):

- **Financial**: Income, debt-to-income ratio (DTI), loan-to-value ratio (LTV)
- **Loan Characteristics**: Amount, term, interest rate, product type
- **Property Details**: Value, type, geographic location
- **Demographics**: Race, ethnicity, age, gender (used only for fairness auditing, not as model inputs)

### Methodology

**Preprocessing**:

- Feature hashing for high-cardinality categorical variables
- One-hot encoding for nominal features
- Normalization/standardization for numerical features
- Fair representation learning to generate unbiased feature embeddings

**Models**:

- Random Forest (ensemble of decision trees)
- XGBoost (gradient boosting)
- TabNet (deep learning for tabular data)

**Training**:

- Loss function: Binary Cross-Entropy (BCE)
- Fairness constraints: Demographic parity regularization
- Validation: Stratified k-fold cross-validation

**Evaluation**:

- Performance: AUC-ROC, Precision, Recall, F1 Score, Confusion Matrix
- Fairness: Demographic Parity Difference, Equalized Odds Difference, 80% Rule
- Explainability: SHAP feature importance, force plots

## Technology Stack

**Core**: Python 3.8+

**Data & ML**: pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, TabNet

**Fairness & Explainability**: Fairlearn, AIF360, SHAP, LIME

**API & Backend**: FastAPI, Pydantic, Uvicorn

**Frontend**: Streamlit, Plotly

**DevOps**: Docker, GitHub Actions

**Development**: Jupyter, pytest, black, Git

##  Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/loan_approval.git
cd loan_approval
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download HMDA data**

- Visit [FFIEC HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/)
- Download 2024 data
- Place in `data/raw/hdma_loan_data_2024.csv`

5. **Run preprocessing and training**

```bash
jupyter notebook notebooks/00_data_exploration.py
python src/models/trainer.py --config config.yaml
```

6. **Launch API and Dashboard**

```bash
uvicorn src.api.main:app --reload
streamlit run src/dashboard/app.py
```

---

## Usage Example

### API Prediction

```python
import requests

application = {
    "income": 95000,
    "loan_amount": 350000,
    "property_value": 450000,
    "dti": 28,
    "credit_score": 740,
    "property_type": "Single Family",
    "loan_term": 30
}

response = requests.post("http://localhost:8000/predict", json=application)
print(response.json())
# Output: {"prediction": "approved", "probability": 0.87, "explanation": "..."}
```

### Dashboard

1. Navigate to `http://localhost:8501`
2. Enter application details
3. Review approval recommendation and SHAP explanation
4. Explore "What-if" scenarios

---

## Project Status

| Phase | Status |
|-------|--------|
| Data Collection & Cleaning | ✅ Complete |
| Exploratory Analysis | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Model Training & Evaluation | ✅ Complete |
| Fairness Assessment | ✅ Complete |
| Explainability | ✅ Complete |
| API Development | ✅ Complete |
| Dashboard | ✅ Complete |
| Documentation | ✅ Complete |

---

## Contact

**Josiah Gordor**  
Email: gordorjoe@gmail.com  
GitHub: [@pj710](https://github.com/pj710)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Data**: Federal Financial Institutions Examination Council (FFIEC) HMDA dataset
- **Fairness Tools**: Fairlearn and AI Fairness 360 teams
- **Development**: Built with Claude Sonnet 4.5 coding agent

---

**Disclaimer**: This project is for educational and portfolio demonstration purposes. Production deployment requires comprehensive legal review, compliance validation, and regulatory approval.
