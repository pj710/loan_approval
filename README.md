# AI-Powered Mortgage Underwriting Assistant

> An intelligent ML-based decision support system that helps mortgage lenders make consistent, fair, and accurate loan approval decisions using HMDA data.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

### Context & Motivation

Homeownership represents a cornerstone of the American dream;the aspiration to achieve success and build wealth through years of hard work. Mortgages enable millions of Americans to realize this dream by providing financing to creditworthy applicants for purchasing homes as primary residences or as investment vehicles for wealth creation.

### The Challenge: Complexity & Inefficiency in Traditional Underwriting

The mortgage business is a fast paced, highly regulated, data-intensive enterprise requiring rigorous underwriting processes to determine loan approval decisions. Current challenges include:

- **Process Duration**: The approval process spans multiple stages from initial application through underwriting to closing;often taking 45 to 60 days.
- **Regulatory Burden**: Lenders must ensure strict compliance with requirements from federal regulators and investors at every stage, facing substantial penalties for violations
- **Product Complexity**: The diverse array of mortgage products with varying requirements from regulators, secondary markets, and investors make manual review time-consuming and cognitively demanding.
- **Human Bias Risk**: Manual underwriting processes are susceptible to unconscious biases that can lead to discriminatory lending practices

### The Opportunity: Machine Learning for Automated Decision Support

Machine learning offers transformative capabilities for modernizing the mortgage approval process through intelligent automation:

- **Speed**: Automated analytics accelerate application review, providing real-time recommendations for each submission
- **Consistency**: ML models apply uniform evaluation criteria across all applications, reducing variability in decisions
- **Risk Detection**: Algorithms can flag potential issues or red flags for manual review by experienced underwriters
- **Predictive Power**: Supervised classifiers generate probability scores based on comprehensive analysis of applicant information, loan terms, and property details
- **Transparency**: Feature importance analysis reveals which factors drive approval decisions, supporting underwriters judgment and regulatory compliance, through advanced analytics.

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
| **Cost-Sensitive Decision Quality** | Expected Value | Higher than baseline approval/denial policy under asymmetric costs |
| **Processing Speed** | Review Time | < 10 minutes per application |
| **Fairness (Race)** | Demographic Parity Difference | < 0.05 |
| **Fairness (Outcomes)** | Equalized Odds Difference | < 0.05 |
| **API Response** | Latency | < 500ms |

### Key Capabilities

Current implementation (available now):

- **Data Ingestion and Validation**: Loads pipe-delimited HMDA records and validates required columns.
- **Preprocessing and Labeling Scaffold**: Standardizes missing values, converts configured numeric fields, and creates a binary target from HMDA action codes.
- **Train/Validation Split**: Produces stratified train/validation partitions with imputation-ready preprocessing.
- **Pipeline Health API**: FastAPI service exposes `/health` and `/pipeline` endpoints for operational checks.
- **Dashboard Scaffold**: Streamlit shell is available for future risk, explainability, and fairness visualizations.

Planned capabilities (roadmap):

- Multi-model training (Random Forest, XGBoost, TabNet)
- Fairness metrics and mitigation workflows
- SHAP-driven explanation artifacts and what-if analysis
- Production-grade prediction endpoint (`/predict`)

## Technical Approach

### Machine Learning Task

**Problem Type**: Supervised binary classification

**Target Variable**: Loan approval decision (Binary: 1 = Approved, 0 = Denied)

**Model Output**: 
- Approval probability score (0.0 to 1.0)
- Binary classification (Approved/Denied)
- Feature attribution (SHAP values)

### Data

**Current Source in Repo**: HMDA lender-level extract (`5493001WHVQBGRSWEU75_header.txt`)  
**Current Scale**: ~30K records in the local sample used for scaffolding and pipeline validation  
**Target Scale**: Full annual HMDA dataset in downstream training/evaluation runs

**Input Features** (15-20 variables):

- **Financial**: Income, debt-to-income ratio (DTI), loan-to-value ratio (LTV)
- **Loan Characteristics**: Amount, term, interest rate, product type
- **Property Details**: Value, type, geographic location
- **Demographics**: Race, ethnicity, age, gender (used only for fairness auditing, not as model inputs)

### Methodology

The items below define the target methodology for upcoming phases. The current codebase implements ingestion, preprocessing scaffolding, target labeling, and train/validation splitting.

**Preprocessing**:

- Feature hashing for high-cardinality categorical variables
- One-hot encoding for nominal features
- Normalization/standardization for numerical features
- Adversarial debiasing using autoencoders

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
- Expected Value Analysis: Compare approval policies using a cost-sensitive expected-value framework that accounts for asymmetric misclassification costs, where false positives (approving a risky loan) and false negatives (denying a creditworthy applicant) can carry different financial and regulatory consequences
- Fairness: Demographic Parity Difference, Equalized Odds Difference, 80% Rule
- Explainability: SHAP feature importance, force plots

**Expected Value Comparison**:

In addition to standard classification metrics, the model will be evaluated by estimating expected value under a cost matrix such as:

- True Positive (TP): value of approving a creditworthy applicant
- True Negative (TN): value of correctly denying an uncreditworthy applicant
- False Positive (FP): cost of approving a risky applicant
- False Negative (FN): cost of denying a creditworthy applicant

The expected value can be expressed as:

$EV = TP \times V_{TP} + TN \times V_{TN} + FP \times V_{FP} + FN \times V_{FN}$

This makes it possible to compare competing models or decision thresholds in a way that reflects real lending economics rather than relying on accuracy alone.

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
- Place it in the path configured under `paths.data_raw` in `config.yaml`

5. **Run preprocessing scaffold and pipeline check**

```bash
python -c "from loan_approval.pipeline import run_training_pipeline; print(run_training_pipeline())"
```

6. **Launch API and Dashboard**

```bash
uvicorn loan_approval.app.main:app --reload
streamlit run loan_approval/app/dashboard.py
```

---

## Usage Example

### API Status Check

```python
import requests

response = requests.get("http://localhost:8000/pipeline")
print(response.json())
# Output includes data profile, missing-column checks, target summary, and split summary
```

### Dashboard

1. Navigate to `http://localhost:8501`
2. Validate pipeline readiness and ingestion status
3. (Planned) Review approval recommendation and SHAP explanation
4. (Planned) Explore "What-if" scenarios

---

## Project Status

| Phase | Status |
|-------|--------|
| Data Collection & Ingestion Scaffold | ✅ Complete |
| Preprocessing & Target Mapping Scaffold | ✅ Complete |
| Train/Validation Split Scaffold | ✅ Complete |
| Model Training & Evaluation | 🚧 In Progress |
| Fairness Assessment | 🚧 In Progress |
| Explainability | 🚧 In Progress |
| API Development | 🚧 In Progress |
| Dashboard | 🚧 In Progress |
| Documentation | 🚧 In Progress |

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
