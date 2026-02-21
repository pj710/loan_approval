# Loan Approval Prediction API Documentation

**Author:** Josiah Gordor  
**Version:** 1.0.0  
**Last Updated:** February 2026

## Overview

The Loan Approval Prediction API provides fair and explainable loan approval recommendations powered by machine learning. Built with FastAPI, it offers instant predictions using multiple fairness-aware models trained on HMDA 2024 data.

### Key Features

- **Instant Predictions**: Get loan approval recommendations in <500ms
- **Explainable AI**: SHAP-based explanations for every decision
- **Fairness-Aware**: Model trained to reduce demographic bias
- **Batch Processing**: Process up to 100 applications at once
- **Interactive Docs**: Built-in Swagger UI and ReDoc

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api.main:app --reload --port 8000
```

### First Request

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "loan_amount": 250000,
        "property_value": 300000,
        "income": 85000,
        "interest_rate": 6.5,
        "loan_term": 360
    }
)

print(response.json())
# {
#     "prediction": "Approved",
#     "probability": 0.8542,
#     "confidence": 70.8,
#     "risk_level": "Low Risk",
#     "key_factors": [...],
#     "model_name": "XGBoost Fair",
#     "processing_time_ms": 12.5
# }
```

---

## API Endpoints

### Health Check

**GET** `/health`

Check API status and model availability.

#### Response

```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_name": "XGBoost Fair",
    "encoder_loaded": true,
    "scaler_loaded": true,
    "version": "1.0.0",
    "timestamp": "2024-12-28T15:30:00Z"
}
```

---

### Predict Loan Approval

**POST** `/predict`

Get an approval recommendation for a single loan application.

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `loan_amount` | float | Yes | Loan amount in dollars (> 0) |
| `property_value` | float | Yes | Property value in dollars (> 0) |
| `income` | float | Yes | Annual income in dollars (> 0) |
| `interest_rate` | float | Yes | Interest rate percentage (0-100) |
| `loan_term` | int | Yes | Loan term in months (1-600) |
| `state_code` | string | No | Two-letter state code (e.g., "CA") |
| `county_code` | string | No | FIPS county code |
| `is_fha_loan` | bool | No | FHA-insured loan (default: false) |
| `is_va_loan` | bool | No | VA-guaranteed loan (default: false) |
| `has_coborrower` | bool | No | Has co-borrower (default: false) |
| `applicant_age` | int | No | Applicant age in years |
| `rate_spread` | float | No | Rate spread over APOR |
| `origination_charges` | float | No | Origination charges in dollars |
| `model_name` | string | No | Model to use for prediction (see Model Selection) |

---

### Model Selection

The API supports multiple fair ML models. Specify the desired model using the `model_name` parameter:

| Model Name | Description | Best For |
|------------|-------------|----------|
| `xgb_fair` | XGBoost with fairness constraints | Balance of performance & fairness |
| `rf_fair` | Random Forest with balanced sampling | Robust predictions |
| `lr_fair` | Logistic Regression (fairness-aware) | Fast, interpretable |
| `nn_fair` | Neural Network (fairness-aware) | Complex patterns |
| `glm_fair` | Generalized Linear Model | Best overall accuracy |
| `fasterrisk_fair` | FasterRisk scoring model | Highly interpretable |
| `gosdt_fair` | Optimal sparse decision trees | Rule-based decisions |

**Default:** `xgb_fair`

#### Example with Model Selection

```json
{
    "loan_amount": 250000,
    "property_value": 300000,
    "income": 85000,
    "interest_rate": 6.5,
    "loan_term": 360,
    "model_name": "glm_fair"
}
```

#### Example Request

```json
{
    "loan_amount": 250000,
    "property_value": 300000,
    "income": 85000,
    "interest_rate": 6.5,
    "loan_term": 360,
    "state_code": "CA",
    "is_fha_loan": false,
    "has_coborrower": true,
    "applicant_age": 35
}
```

#### Response

```json
{
    "prediction": "Approved",
    "probability": 0.8542,
    "confidence": 70.8,
    "risk_level": "Low Risk",
    "key_factors": [
        {
            "feature": "loan_to_value_ratio",
            "impact": 0.1,
            "direction": "positive"
        },
        {
            "feature": "income",
            "impact": 0.08,
            "direction": "positive"
        }
    ],
    "model_name": "XGBoost Fair",
    "processing_time_ms": 12.5
}
```

#### Risk Levels

| Probability | Risk Level |
|-------------|------------|
| ≥ 0.80 | Low Risk |
| 0.50 - 0.79 | Moderate Risk |
| < 0.50 | High Risk |

---

### Explain Prediction

**POST** `/explain`

Get a SHAP-based explanation for a loan approval prediction.

#### Request Body

Same as `/predict` endpoint.

#### Response

```json
{
    "prediction": "Approved",
    "probability": 0.8542,
    "base_value": 0.5,
    "feature_contributions": [
        {
            "feature": "income",
            "value": 85000,
            "shap_value": 0.0823
        },
        {
            "feature": "loan_to_value_ratio",
            "value": 0.833,
            "shap_value": 0.0654
        }
    ],
    "top_positive_factors": [
        {
            "feature": "income",
            "value": 85000,
            "shap_value": 0.0823
        }
    ],
    "top_negative_factors": [
        {
            "feature": "interest_rate",
            "value": 6.5,
            "shap_value": -0.0321
        }
    ],
    "explanation_text": "APPROVED (85% probability, 71% confidence) based on: income (+0.08), loan_to_value_ratio (+0.07), interest_rate (-0.03)"
}
```

---

### Batch Predictions

**POST** `/batch/predict`

Process multiple loan applications at once (max 100).

#### Request Body

```json
{
    "applications": [
        {
            "loan_amount": 250000,
            "property_value": 300000,
            "income": 85000,
            "interest_rate": 6.5,
            "loan_term": 360
        },
        {
            "loan_amount": 400000,
            "property_value": 500000,
            "income": 120000,
            "interest_rate": 7.0,
            "loan_term": 360
        }
    ]
}
```

#### Response

```json
{
    "predictions": [
        {
            "prediction": "Approved",
            "probability": 0.8542,
            "confidence": 70.8,
            "risk_level": "Low Risk",
            "key_factors": [...],
            "model_name": "XGBoost Fair",
            "processing_time_ms": 12.5
        },
        {
            "prediction": "Approved",
            "probability": 0.7621,
            "confidence": 52.4,
            "risk_level": "Moderate Risk",
            "key_factors": [...],
            "model_name": "XGBoost Fair",
            "processing_time_ms": 11.2
        }
    ],
    "total_processed": 2,
    "processing_time_ms": 45.3
}
```

---

## Error Handling

### Validation Errors (422)

```json
{
    "detail": [
        {
            "loc": ["body", "loan_amount"],
            "msg": "ensure this value is greater than 0",
            "type": "value_error.number.not_gt"
        }
    ]
}
```

### Service Unavailable (503)

```json
{
    "detail": "Models not loaded. Check /health endpoint."
}
```

### Internal Server Error (500)

```json
{
    "error": "Prediction failed",
    "message": "Error details here",
    "status_code": 500
}
```

---

## Interactive Documentation

### Swagger UI

Access interactive API documentation at:
```
http://localhost:8000/docs
```

### ReDoc

Access alternative documentation at:
```
http://localhost:8000/redoc
```

### OpenAPI Schema

Download the OpenAPI specification:
```
http://localhost:8000/openapi.json
```

---

## Model Information

### Available Fair Models

The API supports seven fairness-aware ML models:

| Model | Accuracy | AUC | DPD (Race) | Best Use Case |
|-------|----------|-----|------------|---------------|
| GLM_Fair | 98.1% | 99.5% | 0.055 | Best accuracy |
| LR_Fair | 97.7% | 98.7% | 0.051 | Fast, interpretable |
| XGB_Fair | 95.2% | 84.4% | 0.002 | Best fairness |
| Ensemble_Fair | 95.6% | 98.1% | 0.013 | Balanced |
| FasterRisk_Fair | 95.2% | 67.8% | 0.0004 | Most interpretable |
| NN_Fair | 95.2% | 60.2% | 0.0 | Complex patterns |
| GOSDT_Fair | 95.2% | 50.0% | 0.0 | Rule-based |
| RF_Fair | 77.0% | 76.3% | 0.131 | Robust predictions |

### Fair Representation Learning

Before classification, features are transformed through a fair representation encoder that:

1. **Scales features** using StandardScaler (32 features)
2. **Encodes to latent space** (32 → 64 dimensions)
3. **Reduces demographic information** while preserving predictive power

---

## Usage Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
health = requests.get(f"{BASE_URL}/health").json()
print(f"API Status: {health['status']}")

# Single prediction
application = {
    "loan_amount": 250000,
    "property_value": 300000,
    "income": 85000,
    "interest_rate": 6.5,
    "loan_term": 360,
    "has_coborrower": True
}

result = requests.post(f"{BASE_URL}/predict", json=application).json()
print(f"Prediction: {result['prediction']} ({result['probability']:.1%})")

# Get explanation
explanation = requests.post(f"{BASE_URL}/explain", json=application).json()
print(f"Explanation: {explanation['explanation_text']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 250000,
    "property_value": 300000,
    "income": 85000,
    "interest_rate": 6.5,
    "loan_term": 360
  }'

# Explanation
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 250000,
    "property_value": 300000,
    "income": 85000,
    "interest_rate": 6.5,
    "loan_term": 360
  }'
```

### JavaScript (Fetch)

```javascript
const BASE_URL = 'http://localhost:8000';

// Prediction
const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        loan_amount: 250000,
        property_value: 300000,
        income: 85000,
        interest_rate: 6.5,
        loan_term: 360
    })
});

const result = await response.json();
console.log(`Prediction: ${result.prediction} (${result.probability})`);
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Path to models directory | `./models` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Running in Production

```bash
# With Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# With Docker
docker build -t loan-api .
docker run -p 8000:8000 loan-api
```

---

## Testing

### Run Unit Tests

```bash
# All tests
pytest tests/test_api.py -v

# With coverage
pytest tests/test_api.py -v --cov=src.api --cov-report=html
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/locustfile.py --host=http://localhost:8000
```

---

## Rate Limits & Performance

| Endpoint | Mean | Median | P95 |
|----------|------|--------|-----|
| `/health` | 12ms | 9ms | 29ms |
| `/predict` | 145ms | 125ms | 310ms |
| `/explain` | 285ms | 264ms | 574ms |
| `/batch/predict` (10) | 1.4s | 934ms | 9.5s |

- **Target SLA**: < 500ms for single predictions 
- **Recommended throughput**: 50-100 requests/second

---

## Support

**Author:** Josiah Gordor

For issues or questions, please open an issue in the repository.
