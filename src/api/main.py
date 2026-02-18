"""
FastAPI Application for Loan Approval Prediction Service.

This API provides:
- Loan approval predictions using a fairness-aware ML model
- SHAP-based explanations for decisions
- Health monitoring endpoint

Usage:
    uvicorn src.api.main:app --reload --port 8000
    
API Docs:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import (
    health_router,
    predict_router,
    explain_router,
    load_models
)
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Load models on startup, cleanup on shutdown.
    """
    logger.info("🚀 Starting Loan Approval API...")
    
    # Determine model directory
    # Support both local development and production paths
    if os.environ.get("MODEL_DIR"):
        model_dir = Path(os.environ["MODEL_DIR"])
    else:
        # Default: relative to project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        model_dir = project_root / "models"
    
    logger.info(f"📁 Loading models from: {model_dir}")
    
    # Load all model components
    success = load_models(model_dir)
    
    if success:
        logger.info("✅ All models loaded successfully")
    else:
        logger.warning("⚠️ Some models failed to load - API may have limited functionality")
    
    yield  # Application runs
    
    # Cleanup
    logger.info("👋 Shutting down Loan Approval API...")


# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Loan Approval Prediction API",
    description="""
## AI-Powered Mortgage Underwriting Assistant

This API provides fair and explainable loan approval predictions using
a machine learning model trained with fairness constraints.

### Features

- **Instant Predictions**: Get loan approval recommendations in <500ms
- **Explainable AI**: SHAP-based explanations for every decision
- **Fairness-Aware**: Model trained to reduce demographic bias
- **Batch Processing**: Process up to 100 applications at once

### Model Details

The API uses an XGBoost classifier trained on HMDA 2024 data with:
- Fair representation learning to reduce bias
- 95%+ accuracy on validation data
- Compliance with fair lending regulations

### Quick Start

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
```
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8080",  # Alternative dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

app.include_router(health_router, prefix="", tags=["Health"])
app.include_router(predict_router, prefix="", tags=["Predictions"])
app.include_router(explain_router, prefix="", tags=["Explainability"])


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Loan Approval Prediction API",
        "version": __version__,
        "description": "AI-Powered Mortgage Underwriting Assistant",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
