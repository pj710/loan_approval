# Configuration Guide

**Author:** Josiah Gordor  
**Last Updated:** February 2026

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4+ cores |
| Disk Space | 2 GB | 5 GB |

### Operating System

- macOS 10.15+
- Ubuntu 18.04+
- Windows 10+ (with WSL2 recommended)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/josiahgordor/loan_approval.git
cd loan_approval
```

### 2. Create Virtual Environment

**Using Conda (Recommended)**

```bash
conda create -n loan_approval python=3.10
conda activate loan_approval
```

**Using venv**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, shap; print('All dependencies installed!')"
```

---

## Environment Variables

### API Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MODEL_DIR` | Path to models directory | `./models` | `/app/models` |
| `LOG_LEVEL` | Logging verbosity | `INFO` | `DEBUG`, `WARNING`, `ERROR` |
| `API_HOST` | API bind address | `0.0.0.0` | `127.0.0.1` |
| `API_PORT` | API port number | `8000` | `8080` |
| `WORKERS` | Number of worker processes | `1` | `4` |

### Dashboard Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `API_URL` | Backend API URL | `http://localhost:8000` | `https://api.example.com` |
| `STREAMLIT_PORT` | Dashboard port | `8501` | `8080` |

### Setting Environment Variables

**macOS/Linux**

```bash
export MODEL_DIR=/path/to/models
export LOG_LEVEL=INFO
export API_PORT=8000
```

**Windows (PowerShell)**

```powershell
$env:MODEL_DIR = "C:\path\to\models"
$env:LOG_LEVEL = "INFO"
$env:API_PORT = "8000"
```

**Using .env file**

Create a `.env` file in the project root:

```env
MODEL_DIR=./models
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Starting the Application

### Development Mode

**Start API Server**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**Start Dashboard**

```bash
streamlit run src/dashboard/app.py --server.port 8501
```

### Production Mode

**API with Gunicorn**

```bash
gunicorn src.api.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

**Dashboard**

```bash
streamlit run src/dashboard/app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0
```

---

## Project Structure

```
loan_approval/
├── config/                    # Configuration files
│   ├── config.yaml           # Main configuration
│   ├── model_params.yaml     # Model hyperparameters
│   └── feature_config.yaml   # Feature engineering settings
├── data/
│   ├── raw/                  # Original HMDA data
│   └── processed/            # Cleaned datasets
├── docs/                     # Documentation
├── models/                   # Trained model artifacts
│   ├── fair_models/         # Fairness-aware models
│   ├── xgboost_model.pkl
│   ├── feature_scaler.pkl
│   └── ...
├── notebooks/               # Jupyter notebooks
├── reports/                 # Generated reports
├── results/                 # Metrics & benchmarks
├── src/
│   ├── api/                # FastAPI backend
│   ├── dashboard/          # Streamlit frontend
│   ├── data/               # Data processing
│   └── models/             # ML training & evaluation
├── tests/                  # Test suite
├── requirements.txt
└── README.md
```

---

## Model Files

### Required Model Artifacts

The API requires these files in `models/`:

| File | Description | Size |
|------|-------------|------|
| `feature_scaler.pkl` | StandardScaler for features | ~10 KB |
| `fair_models/xgb_fair.pkl` | XGBoost fair model | ~5 MB |
| `fair_models/rf_fair.pkl` | Random Forest fair model | ~50 MB |
| `fair_models/lr_fair.pkl` | Logistic Regression fair model | ~50 KB |
| `fair_models/nn_fair.pkl` | Neural Network fair model | ~5 MB |
| `fair_models/glm_fair.pkl` | GLM fair model | ~50 KB |
| `fair_models/fasterrisk_fair.pkl` | FasterRisk fair model | ~10 KB |
| `fair_models/gosdt_fair.pkl` | GOSDT fair model | ~10 KB |

### Generating Models

If model files are missing, train them using:

```bash
python -m src.models.trainer
```

Or run the training notebook:

```bash
jupyter notebook notebooks/05_model_training.ipynb
```

---

## Configuration Files

### config.yaml

```yaml
# Main configuration
data:
  raw_path: "data/raw/hdma_loan_data_2024.csv"
  processed_path: "data/processed/"
  test_size: 0.15
  validation_size: 0.15
  random_state: 42

model:
  default: "xgb_fair"
  threshold: 0.5
  
api:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

dashboard:
  api_url: "http://localhost:8000"
  theme: "light"
```

### model_params.yaml

```yaml
# Model hyperparameters
xgboost:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
  min_child_weight: 3
  subsample: 0.8
  colsample_bytree: 0.8
  scale_pos_weight: 19

random_forest:
  n_estimators: 200
  max_depth: 15
  min_samples_split: 5
  min_samples_leaf: 2
  class_weight: "balanced"

neural_network:
  hidden_layers: [128, 64, 32]
  dropout: 0.3
  learning_rate: 0.001
  epochs: 50
  batch_size: 256
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_DIR=/app/models
      - LOG_LEVEL=INFO

  dashboard:
    build: .
    command: streamlit run src/dashboard/app.py --server.port 8501 --server.headless true
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

### Running with Docker

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Troubleshooting Configuration

### Common Issues

**Models not loading**

```bash
# Check model files exist
ls -la models/fair_models/

# Check file permissions
chmod 644 models/fair_models/*.pkl
```

**Port already in use**

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn src.api.main:app --port 8001
```

**Import errors**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Next Steps

- [API Documentation](api_documentation.md) - Endpoint details
- [Underwriter Guide](underwriter_guide.md) - Using the dashboard
- [Troubleshooting](troubleshooting.md) - Error solutions
