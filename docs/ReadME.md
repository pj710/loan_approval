# AI-Powered Mortgage Underwriting Assistant

> An intelligent ML-based decision support system that helps mortgage lenders make consistent, fair, and accurate loan approval decisions using HMDA data.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [End-to-End Project Tasks](#end-to-end-project-tasks)
- [Technical Stack](#technical-stack)
- [Success Metrics](#success-metrics)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Contact](#contact)

---

## 🎯 Project Overview

### Machine Learning Task
This is a **supervised binary classification problem** that predicts whether a mortgage loan application will be **approved** or **denied** based on applicant information, loan characteristics, and property details.

**Input Features**: 15-20 underwriting factors including:

- Financial metrics (income, debt-to-income* ratio, loan-to-value ratio)
- Loan characteristics (amount, term, interest rate, loan type)
- Property information (value, type, location)
- Applicant demographics (for fairness monitoring only, not model features)

**Target Variable**: Loan approval decision (Binary: 1 = Approved, 0 = Denied)

**Model Output**: 
- Approval probability (0.0 to 1.0)
- Binary prediction (Approved/Denied)
- Explainability features (SHAP values showing which factors influenced the decision)

### Business Objectives
The **AI-powered underwriting assistant** analyzes mortgage applications and provides real-time approval recommendations, risk assessments, and fairness audits. The system is designed to:

- **Accelerate Processing**: Reduce underwriting time from 30 minutes to under 10 minutes per application
- **Ensure Fairness**: Monitor for disparate impact across protected demographics (race, ethnicity, gender)
- **Improve Accuracy**: Achieve ≥75% AUC-ROC with balanced precision (≥0.80) and recall (≥0.70)
- **Support Decisions**: Provide explainable AI insights (SHAP values) to augment human underwriters
- **Minimize Risk**: Prioritize precision to reduce bad loan approvals (false positives)

### Data & Scope
**Data Source**: 2024 HMDA dataset (Home Mortgage Disclosure Act) covering NJ, NY, PA, and CT (~2.4M applications)

**Scope**: Owner-occupied home purchase loans only (excludes refinances and investment properties)

**Modeling Approach**: Ensemble of XGBoost, Random Forest, and Neural Networks with calibrated probabilities

---

## ✨ Key Features

- **Multi-Model Ensemble**: XGBoost, Random Forest, and Neural Networks with calibrated probabilities
- **Underwriting Metrics**: Automated calculation of DTI (debt-to-income), LTV (loan-to-value), income ratios
- **Fairness Auditing**: Real-time monitoring of Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD)
- **Explainability**: SHAP force plots showing which factors drive each approval recommendation
- **Decision Dashboard**: Web interface displaying approval confidence, risk scores, and flagged applications
- **API Integration**: FastAPI backend for seamless workflow integration (<500ms response time)

---

## 📁 Project Structure

```
loan_approval/
│
├── README.md                          # Project documentation and task breakdown
├── requirements.txt                   # Python package dependencies
│
├── data/
│   ├── raw/                          # Original HMDA dataset
│   │   └── hmda_loan_data_2024.csv  # Raw 2024 HMDA data (NJ, NY, PA, CT)
│   └── processed/                    # Cleaned and feature-engineered data
│       ├── train.csv                # Training set
│       ├── validation.csv           # Validation set
│       └── test.csv                 # Test set
│
├── docs/                             # Project documentation
│   ├── ReadME.md                    # Main project documentation
│   ├── api_documentation.md         # API endpoint specifications
│   ├── business_rules.md            # Underwriting business logic
│   ├── compliance_report.md         # Fair lending compliance report
│   ├── configuration_guide.md       # Setup and configuration
│   ├── faq.md                       # Frequently asked questions
│   ├── literature_review.md         # Research findings and references
│   ├── model_card.md                # Model documentation
│   ├── model_governance.md          # Model governance framework
│   ├── troubleshooting.md           # Error solutions guide
│   └── underwriter_guide.md         # Dashboard usage guide
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   ├── 02_data_cleaning.ipynb       # Data preprocessing
│   ├── 03_eda.ipynb                 # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb # Feature creation and selection
│   ├── 05_model_training.ipynb      # Model development and tuning
│   ├── 06_fairness_analysis.ipynb   # Bias detection and mitigation
│   └── 07_model_evaluation.ipynb    # Final model assessment
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Data loading utilities
│   │   ├── preprocessor.py          # Data cleaning and preprocessing
│   │   └── feature_engineering.py   # Feature creation pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Model training scripts
│   │   ├── evaluator.py             # Performance evaluation
│   │   ├── explainer.py             # SHAP and interpretability
│   │   └── fairness.py              # Fairness metrics and mitigation
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── routes.py                # API endpoints
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                   # Streamlit dashboard interface
│
├── models/                           # Saved model artifacts
│   ├── baseline_logistic.pkl        # Baseline model
│   ├── xgboost_model.pkl           # XGBoost model
│   ├── random_forest_model.pkl     # Random Forest model
│   ├── ensemble_model.pkl          # Final ensemble model
│   └── feature_scaler.pkl          # Feature scaling transformer
│
├── reports/                          # Generated reports and visualizations
│   ├── loan_data_profile_report.html # Data profiling report
│   ├── eda_visualizations.pdf       # EDA charts and plots
│   ├── fairness_audit_report.pdf    # Fairness analysis results
│   └── model_performance_report.pdf # Final evaluation report
│
├── results/                          # Model metrics and outputs
│   ├── fairness_metrics_*.csv       # Fairness audit results by date
│   ├── model_performance_*.csv      # Performance metrics by date
│   └── summary_*.json               # Experiment summary files
│
├── tests/                            # Unit and integration tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_model_trainer.py
│   └── test_api.py
│
├── config/                           # Configuration files
│   ├── config.yaml                  # Main configuration
│   ├── model_params.yaml            # Model hyperparameters
│   └── feature_config.yaml          # Feature engineering settings
│
├── docker/                           # Docker configurations
│   ├── Dockerfile                   # API container
│   └── docker-compose.yml           # Multi-container orchestration
│
└── .gitignore                       # Git ignore file
```

---

## 🔨 End-to-End Project Tasks

### **Phase 1: Data Acquisition & Understanding** (Week 1)

#### Task 1.0: Literature Review & Research

- [ ] **ML in Mortgage Lending**: Review academic papers on loan approval prediction models
- [ ] **Regulatory Framework**: Study ECOA (Equal Credit Opportunity Act), Fair Housing Act, HMDA requirements
- [ ] **Fairness in ML**: Review fairness metrics (demographic parity, equalized odds), bias mitigation techniques
- [ ] **Underwriting Standards**: Research traditional underwriting criteria (DTI, LTV thresholds, credit standards)
- [ ] **HMDA Data**: Review HMDA data dictionary, understand all 99 features and their business meaning
- [ ] **Explainable AI**: Review SHAP, LIME, and other interpretability methods for regulated industries
- [ ] **Industry Benchmarks**: Research typical approval rates, default rates, and model performance in mortgage lending
- [ ] **Best Practices**: Review model governance, validation, and monitoring frameworks for financial services
- [ ] **Deliverable**: `docs/literature_review.md` summarizing key findings and implications for project

#### Task 1.1: Data Collection

- [ ] Download 2024 HMDA dataset for NJ, NY, PA, CT from [FFIEC website](https://ffiec.cfpb.gov/data-browser/)
- [ ] Store raw data in `data/raw/hdma_loan_data_2024.csv`
- [ ] Document data source, collection date, and any licensing requirements

#### Task 1.2: Initial Data Exploration

- [ ] Load dataset and inspect dimensions (~2.4M rows expected)
- [ ] Review HMDA data dictionary to understand all 99 columns
- [ ] Check data types, missing value patterns, and basic statistics
- [ ] Create data profile report using `pandas-profiling` or `ydata-profiling`
- [ ] **Deliverable**: `reports/loan_data_profile_report.html`

#### Task 1.3: Define Target Variable

- [ ] Identify target column: `action_taken` (1=Approved, 0=Denied)
- [ ] Filter for relevant actions (exclude withdrawn, incomplete applications)
- [ ] Check class distribution for overall and demographic classes (expect imbalance: more approvals than denials)
- [ ] Document business rules for loan approval in `docs/business_rules.md`

---

### **Phase 2: Data Cleaning & Preprocessing** (Week 1-2)

#### Task 2.1: Data Quality Assessment

- [ ] Identify columns with >40% missing values (consider dropping)
- [ ] Analyze missing data patterns (MCAR, MAR, MNAR)
- [ ] Check for duplicate records (based on applicant ID, loan amount, property address)
- [ ] Validate data ranges (e.g., income > 0, LTV between 0-100)

#### Task 2.2: Handle Missing Values

- [ ] **Numeric features**: Impute with median or create "missing" indicator
- [ ] **Categorical features**: Create "Unknown" category or mode imputation
- [ ] **Critical features**: Drop rows if missing (e.g., income, loan amount)
- [ ] Document imputation strategy in `notebooks/data_cleaning.ipynb`

#### Task 2.3: Filter and Subset Data

- [ ] Filter for owner-occupied principal residences only (`occupancy_type == 1`)
- [ ] Filter for home purchase loans only (`loan_purpose == 1`)
- [ ] Remove outliers: loans >$3M, income >$1M (discuss thresholds)
- [ ] Filter for conventional and government-backed loans (FHA, VA)

---

### **Phase 3: Feature Engineering** (Week 2)

#### Task 3.1: Create Underwriting Metrics if not available 

- [ ] **DTI (Debt-to-Income Ratio)**: `debt / income`
- [ ] **LTV (Loan-to-Value Ratio)**: `loan_amount / property_value`
- [ ] **Housing Expense Ratio**: `(loan_amount * interest_rate) / income`

#### Task 3.2: Derive Additional Features

- [ ] **Loan characteristics**: interest rate spread, loan term (30-year vs 15-year)
- [ ] **Property features**: property type (single-family, condo, etc.), location (urban vs rural)
- [ ] **Applicant demographics**: age (derived from application date), co-applicant presence
- [ ] **Geographic features**: county-level median income, home prices, unemployment rate

#### Task 3.3: Create Interaction Features

- [ ] DTI × LTV (joint risk indicator)
- [ ] Income × Property Location (affordability in expensive areas)
- [ ] Loan amount × Interest rate (payment burden)

#### Task 3.4: Feature Selection

- [ ] Remove highly correlated features (>0.95 correlation)
- [ ] Use Random Forest feature importance to rank features
- [ ] Select top features based on business relevance + statistical importance
- [ ] **Deliverable**: `data/processed/feature_selected_data.csv`

#### Task 3.5: Encode Categorical Variables

- [ ] One-hot encode key categorical Variables
- [ ] Ordinal encode: credit score ranges (if available)
- [ ] Target encode high-cardinality features (county code, census tract)
- [ ] Document encoding mappings in `src/data/encodings.json`

---

### **Phase 4: Exploratory Data Analysis** (Week 2)

#### Task 4.1: Univariate Analysis

- [ ] Plot distributions of key continuous features (income, loan amount, DTI, LTV)
- [ ] Analyze categorical feature frequencies (property type, loan type)
- [ ] Check for skewness and apply transformations if needed (log, Box-Cox)

#### Task 4.2: Bivariate Analysis

- [ ] Approval rate by income bracket
- [ ] Approval rate by DTI and LTV thresholds (industry standards: DTI <43%, LTV <80%)
- [ ] Approval rate by property type and location
- [ ] Correlation heatmap of numeric features

#### Task 4.3: Fairness Preliminary Analysis

- [ ] Approval rates by race, ethnicity, and gender
- [ ] Visual comparison of DTI/LTV distributions across demographics
- [ ] Identify potential disparities requiring mitigation
- [ ] **Deliverable**: `reports/eda_visualizations.pdf`

---

### **Phase 5: Model Development** (Week 2-3)

- [ ] remove irrelevant features

#### Task 5.1: Train-Test Split

- [ ] Split data: 70% train, 15% validation, 15% test
- [ ] Ensure stratification by target variable (preserve class distribution)
- [ ] Set random seed for reproducibility

#### Task 5.2: Baseline Model

- [ ] Train logistic regression as baseline
- [ ] Evaluate with AUC-ROC, precision, recall, F1 score
- [ ] Document baseline performance: Expected AUC ~0.66-0.70

#### Task 5.3: Advanced Models

- [ ] **XGBoost**: Tune hyperparameters (learning rate, max depth, n_estimators)
- [ ] **Random Forest**: Tune n_estimators, max_features, min_samples_split
- [ ] **Neural Network**: 2-3 hidden layers, batch normalization, dropout
- [ ] Use 5-fold cross-validation for hyperparameter tuning


#### Task 5.5: Ensemble Model

- [ ] Create soft voting ensemble (average probabilities from top 3 models)
- [ ] Apply probability calibration (Platt scaling or isotonic regression)
- [ ] Validate calibration with reliability diagrams

#### Task 5.6: Model Evaluation

- [ ] Evaluate all models on validation set
- [ ] Generate classification reports (precision, recall, F1 for each class)
- [ ] Plot ROC curves and precision-recall curves
- [ ] Select best model based on AUC-ROC and business objectives
- [ ] **Deliverable**: `models/best_model.pkl`, `results/model_performance_report.csv`

---

### **Phase 6: Fairness Analysis & Mitigation** (Week 3)

#### Task 6.1: Calculate Fairness Metrics

- [ ] **Demographic Parity Difference (DPD)**: Compare approval rates across groups
- [ ] **Equalized Odds Difference (EOD)**: Compare TPR and FPR across groups
- [ ] **Disparate Impact**: 80% rule test (CFR §1607.4D)
- [ ] Calculate for race, ethnicity, and gender subgroups

#### Task 6.2: Bias Detection

- [ ] Identify features contributing to bias using SHAP dependence plots
- [ ] Test for indirect discrimination (proxy variables)
- [ ] Document findings in `reports/fairness_audit_report.pdf`

#### Task 6.3: Bias Mitigation

- [ ] **Pre-processing**: Reweighting or resampling training data
- [ ] **In-processing**: Adversarial debiasing or fairness constraints during training
- [ ] **Post-processing**: Equalized odds adjustment to decision thresholds
- [ ] Re-evaluate model performance after mitigation

#### Task 6.4: Validate Fairness

- [ ] Ensure DPD and EOD < 0.05 on test set
- [ ] Verify 80% rule compliance
- [ ] Trade-off analysis: fairness vs accuracy
- [ ] **Deliverable**: `results/fairness_metrics.csv`

---

### **Phase 7: Model Explainability** (Week 3)

#### Task 7.1: Global Interpretability

- [ ] SHAP summary plot: Top 10 most important features
- [ ] Feature importance rankings from each model
- [ ] Partial dependence plots for DTI, LTV, income

#### Task 7.2: Local Interpretability

- [ ] Generate SHAP force plots for individual predictions
- [ ] Create example explanations for approved, denied, and borderline cases
- [ ] Build explanation templates for underwriters

#### Task 7.3: Create Explanation Module

- [ ] Build function to generate human-readable explanations
- [ ] Format: "Approved (85% confidence) because: low DTI (28%), high income ($95K), excellent LTV (75%)"
- [ ] Integrate into API and dashboard
- [ ] **Deliverable**: `src/models/explainer.py`

---

### **Phase 8: API Development** (Week 3-4)

#### Task 8.1: FastAPI Backend

- [ ] Create `/predict` endpoint: accepts application data, returns approval recommendation
- [ ] Create `/explain` endpoint: returns SHAP values for a given prediction
- [ ] Create `/health` endpoint: checks model availability and system status
- [ ] Implement input validation using Pydantic schemas

#### Task 8.2: Model Serving

- [ ] Load trained model on API startup
- [ ] Implement request preprocessing pipeline
- [ ] Add error handling and logging
- [ ] Optimize for <500ms response time

#### Task 8.3: API Testing

- [ ] Write unit tests for each endpoint
- [ ] Load testing with 100 concurrent requests
- [ ] Document API with auto-generated Swagger docs
- [ ] **Deliverable**: `src/api/main.py`, API documentation at `/docs`

---

### **Phase 9: Dashboard Development** (Week 4)

#### Task 9.1: Streamlit Interface

- [ ] **Input Form**: Collect loan application details (income, loan amount, property value, etc.)
- [ ] **Prediction Display**: Show approval/denial with confidence score
- [ ] **Explanation Panel**: Display SHAP force plot and key factors
- [ ] **Fairness Dashboard**: Show demographic parity metrics in real-time

#### Task 9.2: Interactive Features

- [ ] "What-if" analysis: Adjust DTI/LTV sliders to see impact on approval
- [ ] Batch upload: Process CSV of multiple applications
- [ ] Risk segmentation: Flag high-risk applications for manual review

#### Task 9.3: Visualization

- [ ] Approval probability gauge
- [ ] Feature importance bar chart
- [ ] Fairness metrics table
- [ ] **Deliverable**: `src/dashboard/app.py`

---

### **Phase 10: Testing & Validation** (Week 4)

#### Task 10.1: Model Validation

- [ ] Final evaluation on held-out test set
- [ ] Compare against success criteria (AUC ≥0.75, Precision ≥0.80, Recall ≥0.70)
- [ ] Business validation: Test on real-world edge cases
- [ ] Shadow deployment: Run predictions on historical data and compare to actual decisions

#### Task 10.2: System Testing

- [ ] End-to-end integration testing (API → Dashboard)
- [ ] Performance testing (latency, throughput)
- [ ] Security testing (input sanitization, authentication)
- [ ] User acceptance testing with mock underwriters

#### Task 10.3: Error Analysis

- [ ] Analyze false positives (bad loans approved)
- [ ] Analyze false negatives (good borrowers denied)
- [ ] Identify systematic errors and root causes
- [ ] Document model limitations

---

### **Phase 11: Documentation** (Week 4) ✅ COMPLETE

#### Task 11.1: Technical Documentation

- [x] **Model Card**: Architecture, training data, performance, fairness, limitations
- [x] **API Documentation**: Endpoint specifications, request/response examples
- [x] **Code Documentation**: Docstrings for all functions and classes
- [x] **Configuration Guide**: Setup instructions, environment variables

#### Task 11.2: User Documentation

- [x] **Underwriter Guide**: How to use the dashboard, interpret recommendations
- [x] **FAQ**: Common questions about model predictions
- [x] **Troubleshooting Guide**: Error messages and solutions

#### Task 11.3: Compliance Documentation

- [x] Fair lending compliance report (ECOA, Fair Housing Act)
- [x] Model governance documentation (development, validation, monitoring)
- [x] Audit trail for regulatory review
- [x] **Deliverable**: `docs/` folder with all documentation

---

### **Phase 12: Deployment** (Week 4)

#### Task 12.1: Containerization

- [ ] Create Dockerfile for API
- [ ] Create docker-compose.yml for full stack (API + Dashboard)
- [ ] Test containers locally

#### Task 12.2: Cloud Deployment

- [ ] Deploy API to AWS/GCP/Azure (or Heroku for MVP)
- [ ] Deploy dashboard as web app
- [ ] Configure environment variables and secrets management
- [ ] Set up HTTPS and authentication

#### Task 12.3: Monitoring & Maintenance

- [ ] Set up logging (application logs, prediction logs)
- [ ] Configure alerting for system errors and performance degradation
- [ ] Create retraining pipeline for quarterly model updates
- [ ] Document rollback procedures

---

## 🛠️ Technical Stack

### Data & ML

- **Python 3.8+**
- **Data Processing**: pandas, NumPy, scikit-learn
- **ML Models**: XGBoost, LightGBM, TensorFlow/Keras
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

## 📊 Success Metrics

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
| Demographic Parity Difference (DPD) | < 0.05 | TBD |
| Equalized Odds Difference (EOD) | < 0.05 | TBD |
| 80% Rule Test | Pass | TBD |

### Business Impact

| Metric | Target |
|--------|--------|
| Processing Time Reduction | 67% (30 min → <10 min) |
| Auto-Approve Rate | 30-40% of low-risk applications |
| Decision Consistency | >90% agreement with senior underwriters |
| API Response Time | <500ms |

---

## 🚀 Setup & Installation

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

## 💡 Usage

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

## � Documentation

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

## �📧 Contact

**Josiah Gordor**  
Email: [gordorjoe@gmail.com]  
GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HMDA Data**: Federal Financial Institutions Examination Council (FFIEC)
- **Fairness Libraries**: Fairlearn and AI Fairness 360 teams
- **Inspiration**: Fair lending practices and responsible AI principles

---

**Note**: This project is for educational and portfolio purposes. Any production deployment requires comprehensive legal review, compliance validation, and regulatory approval.
