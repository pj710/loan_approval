# AI-Powered Mortgage Underwriting Assistant

> An intelligent ML-based decision support system that helps mortgage lenders make consistent, fair, and accurate loan approval decisions using HMDA data.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

The American dream is the hope of achieving success and fulfillment from years of hardwork, symbolised by homeownership. Mortgages aid millions to realize this dream of homeownership by providing financing to credit-worthy applicants to acquire homes as their primary residences or as investment instruments for building wealth.

The mortgage business is a highly regulated and data intensive enterprise, requiring meticulous underwriting processes for determining which applications get approved or denied. The approval process involves several stages from initial application, through underwriting to closing, and can last from several days to months. Complexities in loan requirements for the many mortgage products, make the approval process slow and prone to human bias.

Machine learning provides tools and techniques for improving the approval process through automated analytics. Automated analytics can speed up review process and minimize bias by providing live recommendations for each application and flagging possible issues/red flags for further review by underwriters. A trained supervised classifier can generate probability scores for each application based on applicant information, loan terms and property details. Feature importance information from the model can also provide additional insights into factors impacting the systems decision and support underwriting activities.

## System Objectives

The **AI-powered underwriting assistant** analyzes mortgage applications and provides real-time approval recommendations, risk assessments, and fairness audits. The system is designed to:

- **Accelerate Processing**: Reduce underwriting review time to under 10 minutes per application
- **Ensure Fairness**: Monitor for disparate impact across protected demographics (race, ethnicity, gender, age)
- **Improve Accuracy**: Achieve ≥75% AUC-ROC with f1 Score > 80%
- **Support Decisions**: Provide explainable AI insights (SHAP values) to augment human underwriters
- **Minimize Risk**: Balance precision to reduce bad loan approvals (false positives) with Recall to reduce wrong denials (False negatives)
- **Maximize Expected Value** 
  
### System Features

- **Multi-Model Ensemble**: Top perfoming models are ensembled into one with calibrated probabilities
- **Underwriting Metrics**: Automated calculation of DTI (debt-to-income), LTV (loan-to-value), income ratios
- **Fairness Auditing**: Real-time monitoring of Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD)
- **Explainability**: SHAP force plots showing which factors drive each approval recommendation
- **Decision Dashboard**: Web interface displaying approval confidence, risk scores, and flagged applications
- **API Integration**: FastAPI backend for seamless workflow integration (<500ms response time)

## Machine Learning Task

The system utilizes a **supervised binary classifier** that predicts whether a mortgage loan application will be **approved** or **denied** based on applicant information, loan characteristics, and property details.

**Target Variable**: Loan approval decision (Binary: 1 = Approved, 0 = Denied)

### Model Output

- Approval probability (0.0 to 1.0)
- Binary prediction (Approved/Denied)
- Explainability features (SHAP values showing which factors influenced the decision)

### Data & Scope

**Data Source**: 2024 HMDA dataset (Home Mortgage Disclosure Act); ~2.4M applications across all states in US
**Key input features**: Dataset consists of 15-20 underwriting factors including:

- Financial metrics (income, debt-to-income* ratio, loan-to-value ratio)
- Loan characteristics (amount, term, interest rate, loan type)
- Property information (value, type, location)
- Applicant demographics (for fairness monitoring only, not model features)

**Scope**: Owner-occupied home purchase loans only (excludes refinances and investment properties)

### Modeling Approach

- Data: Hashed features, One-hot encoding, normalization, fair learning (generate fair feature embeddings for training)
- Model : Random Forest, Xgboost, Neural Net(Tabnet)
- Loss Function: Binary Cross Entropy (BCE)
- Evaluation: Cross Validation, Confusion Matrix (priority: f1 Score),  AUC

## Project Phases & Tasks

| Phase | Task Description                | Key Deliverables                          | Status     |
|-------|---------------------------------|-------------------------------------------|------------|
| 1     | Data Collection                 | Raw loan data CSV                         | Completed  |
| 2     | Data Cleaning                   | Cleaned dataset, notebook                 | Completed  |
| 3     | Exploratory Data Analysis (EDA) | EDA notebook, summary insights            | Completed  |
| 4     | Feature Engineering             | Feature set, transformation scripts        | Completed  |
| 5     | Model Training                  | ML models (XGBoost, TensorFlow)           | Completed  |
| 6     | Model Evaluation                | Performance metrics CSV                   | Completed  |
| 7     | Fairness Assessment             | Fairness metrics CSV                      | Completed  |
| 8     | Explainability                  | SHAP plots, explainability notebook        | Completed  |
| 9     | API Development                 | FastAPI backend, API docs                  | Completed  |
| 10    | Dashboard Development           | Streamlit dashboard, app.py                | Completed  |
| 11    | Documentation                   | Model card, guides, compliance docs        | Completed  |

##  Technical Stack

### Data & ML

- **Python 3.8+**
- **Data Processing**: pandas, NumPy, scikit-learn
- **ML Models**: XGBoost, TensorFlow/Keras
- **Fairness**: Fairlearn, AIF360
- **Explainability**: SHAP, LIME

### API & Backend

- **FastAPI**: REST API framework
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### Frontend

- **Streamlit**: Interactive dashboard
- **Plotly**: Visualizations

### DevOps

- **Docker**: Containerization
- **GitHub Actions**: CI/CD
- **AWS/GCP/Azure**: Cloud hosting

### Development Tools

- **Jupyter**: Notebooks for exploration
- **pytest**: Unit testing
- **black**: Code formatting
- **Git/GitHub**: Version control

---

## Success Metrics

### Model Performance (Technical)
| Metric | Target | Current |
|--------|--------|---------|
| AUC-ROC | ≥ 0.75 | TBD |
| Precision | ≥ 0.80 | TBD |
| Recall | ≥ 0.70 | TBD |
| F1 Score | ≥ 0.70 | TBD |

### Fairness & Compliance

| Metric | Target | Current |
|--------|--------|---------|
| Demographic Parity Difference (DPD) | < 0.05 | TBD|
| Equalized Odds Difference (EOD) | < 0.05 | TBD |
| 80% Rule Test | Pass | TBD |

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

1. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

1. **Download HMDA data**

- Visit [FFIEC HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/)
- Download 2024 data for NJ, NY, PA, CT
- Place in `data/raw/hdma_loan_data_2024.csv`

1. **Run data preprocessing**

```bash
jupyter notebook notebooks/data_cleaning.ipynb
```

1. **Train models**

```bash
python src/models/trainer.py --config config.yaml
```

1. **Launch API**

```bash
uvicorn src.api.main:app --reload
```

1. **Launch Dashboard**

```bash
streamlit run src/dashboard/app.py
```

---

## Usage

### Making Predictions via API

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

### Using the Dashboard

1. Navigate to `http://localhost:8501`
2. Fill in application details in the input form
3. Click "Submit" to get instant approval recommendation
4. Review SHAP explanation and risk factors
5. Adjust parameters in "What-if Analysis" to explore scenarios

---

## Documentation

### Technical Documentation

| Document | Description |
|----------|-------------|
| [Model Card](model_card.md) | Architecture, performance, fairness, limitations |
| [API Documentation](api_documentation.md) | Endpoint specifications, request/response examples |
| [Configuration Guide](configuration_guide.md) | Setup instructions, environment variables |

### User Documentation

| Document | Description |
|----------|-------------|
| [Underwriter Guide](underwriter_guide.md) | How to use the dashboard, interpret recommendations |
| [FAQ](faq.md) | Common questions about model predictions |
| [Troubleshooting](troubleshooting.md) | Error messages and solutions |

### Compliance Documentation

| Document | Description |
|----------|-------------|
| [Compliance Report](compliance_report.md) | Fair lending compliance (ECOA, Fair Housing Act) |
| [Model Governance](model_governance.md) | Model development, validation, monitoring framework |
| [Business Rules](business_rules.md) | Underwriting business logic |

---

## Contact

**Josiah Gordor**  
Email: [gordorjoe@gmail.com]  
GitHub: [@pj710](https://github.com/pj710)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **HMDA Data**: Federal Financial Institutions Examination Council (FFIEC)
- **Fairness Libraries**: Fairlearn and AI Fairness 360 teams
- **Inspiration**: Fair lending practices and responsible AI principles

---

**Note**: This project is for educational and portfolio purposes. Any production deployment requires comprehensive legal review, compliance validation, and regulatory approval. The project relied on heavy use of claude sonnet 4.5 coding agent.
