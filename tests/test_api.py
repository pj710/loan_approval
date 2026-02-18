"""
API Unit Tests for Loan Approval Prediction Service.

Run tests:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --cov=src.api
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def sample_application():
    """Sample valid loan application."""
    return {
        "loan_amount": 250000,
        "property_value": 300000,
        "income": 85000,
        "interest_rate": 6.5,
        "loan_term": 360,
        "state_code": "CA",
        "is_fha_loan": False,
        "is_va_loan": False,
        "has_coborrower": True,
        "applicant_age": 35
    }


@pytest.fixture(scope="module")
def minimal_application():
    """Minimal valid application (required fields only)."""
    return {
        "loan_amount": 200000,
        "property_value": 250000,
        "income": 60000,
        "interest_rate": 7.0,
        "loan_term": 360
    }


@pytest.fixture(scope="module")
def mock_models():
    """Mock model components for testing without loading actual models."""
    import numpy as np
    
    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.15, 0.85]]))
    
    mock_scaler = MagicMock()
    mock_scaler.transform = MagicMock(return_value=np.zeros((1, 32)))
    
    mock_encoder = MagicMock()
    mock_encoder.predict = MagicMock(return_value=np.zeros((1, 64)))
    
    return mock_model, mock_encoder, mock_scaler


@pytest.fixture(scope="module")
def test_client(mock_models):
    """
    Create test client with mocked models.
    """
    from src.api import routes
    from src.api.main import app
    
    mock_model, mock_encoder, mock_scaler = mock_models
    
    # Inject mocked models (using new models dict structure)
    routes.models = {
        "xgb_fair": mock_model,
        "rf_fair": mock_model,
        "lr_fair": mock_model
    }
    routes.encoder = mock_encoder
    routes.scaler = mock_scaler
    routes.feature_names = [f"feature_{i}" for i in range(32)]
    routes.current_model_name = "MockXGBoost"
    
    client = TestClient(app)
    return client


# ============================================================================
# HEALTH ENDPOINT TESTS
# ============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_returns_200(self, test_client):
        """Health endpoint should return 200 when models loaded."""
        response = test_client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_structure(self, test_client):
        """Health response should have required fields."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "encoder_loaded" in data
        assert "scaler_loaded" in data
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_check_models_loaded(self, test_client):
        """Health check should show models as loaded."""
        response = test_client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["encoder_loaded"] is True
        assert data["scaler_loaded"] is True


# ============================================================================
# PREDICT ENDPOINT TESTS
# ============================================================================

class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_predict_returns_200(self, test_client, sample_application):
        """Predict should return 200 for valid input."""
        response = test_client.post("/predict", json=sample_application)
        assert response.status_code == 200
    
    def test_predict_response_structure(self, test_client, sample_application):
        """Predict response should have required fields."""
        response = test_client.post("/predict", json=sample_application)
        data = response.json()
        
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "key_factors" in data
        assert "model_name" in data
        assert "processing_time_ms" in data
    
    def test_predict_probability_range(self, test_client, sample_application):
        """Probability should be between 0 and 1."""
        response = test_client.post("/predict", json=sample_application)
        data = response.json()
        
        assert 0.0 <= data["probability"] <= 1.0
    
    def test_predict_minimal_input(self, test_client, minimal_application):
        """Predict should work with minimal required fields."""
        response = test_client.post("/predict", json=minimal_application)
        assert response.status_code == 200
    
    def test_predict_invalid_loan_amount(self, test_client):
        """Should reject negative loan amount."""
        invalid_app = {
            "loan_amount": -50000,  # Invalid
            "property_value": 300000,
            "income": 85000,
            "interest_rate": 6.5,
            "loan_term": 360
        }
        response = test_client.post("/predict", json=invalid_app)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_required_field(self, test_client):
        """Should reject request missing required fields."""
        incomplete_app = {
            "loan_amount": 250000,
            "property_value": 300000,
            # Missing: income, interest_rate, loan_term
        }
        response = test_client.post("/predict", json=incomplete_app)
        assert response.status_code == 422
    
    def test_predict_risk_level_valid(self, test_client, sample_application):
        """Risk level should be valid category."""
        response = test_client.post("/predict", json=sample_application)
        data = response.json()
        
        valid_risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
        assert data["risk_level"] in valid_risk_levels
    
    def test_predict_processing_time(self, test_client, sample_application):
        """Processing time should be under 500ms (requirement)."""
        response = test_client.post("/predict", json=sample_application)
        data = response.json()
        
        # Should be fast with mocked models
        assert data["processing_time_ms"] < 500


# ============================================================================
# EXPLAIN ENDPOINT TESTS
# ============================================================================

class TestExplainEndpoint:
    """Tests for /explain endpoint."""
    
    def test_explain_returns_200(self, test_client, sample_application):
        """Explain should return 200 for valid input."""
        response = test_client.post("/explain", json=sample_application)
        assert response.status_code == 200
    
    def test_explain_response_structure(self, test_client, sample_application):
        """Explain response should have required fields."""
        response = test_client.post("/explain", json=sample_application)
        data = response.json()
        
        assert "prediction" in data
        assert "probability" in data
        assert "base_value" in data
        assert "feature_contributions" in data
        assert "top_positive_factors" in data
        assert "top_negative_factors" in data
        assert "explanation_text" in data
    
    def test_explain_feature_contributions_format(self, test_client, sample_application):
        """Feature contributions should have correct structure."""
        response = test_client.post("/explain", json=sample_application)
        data = response.json()
        
        if data["feature_contributions"]:
            contribution = data["feature_contributions"][0]
            assert "feature" in contribution
            assert "shap_value" in contribution
    
    def test_explain_text_not_empty(self, test_client, sample_application):
        """Explanation text should not be empty."""
        response = test_client.post("/explain", json=sample_application)
        data = response.json()
        
        assert len(data["explanation_text"]) > 0


# ============================================================================
# BATCH PREDICT ENDPOINT TESTS
# ============================================================================

class TestBatchPredictEndpoint:
    """Tests for /batch/predict endpoint."""
    
    def test_batch_predict_returns_200(self, test_client, sample_application):
        """Batch predict should return 200."""
        request = {"applications": [sample_application, sample_application]}
        response = test_client.post("/batch/predict", json=request)
        assert response.status_code == 200
    
    def test_batch_predict_response_structure(self, test_client, sample_application):
        """Batch response should have required fields."""
        request = {"applications": [sample_application]}
        response = test_client.post("/batch/predict", json=request)
        data = response.json()
        
        assert "predictions" in data
        assert "total_processed" in data
        assert "processing_time_ms" in data
    
    def test_batch_predict_correct_count(self, test_client, sample_application):
        """Should process correct number of applications."""
        apps = [sample_application for _ in range(5)]
        request = {"applications": apps}
        response = test_client.post("/batch/predict", json=request)
        data = response.json()
        
        assert data["total_processed"] == 5
        assert len(data["predictions"]) == 5
    
    def test_batch_predict_empty_list(self, test_client):
        """Should reject empty application list."""
        request = {"applications": []}
        response = test_client.post("/batch/predict", json=request)
        
        # Empty list should be rejected by Pydantic validation (min_items=1)
        assert response.status_code == 422


# ============================================================================
# ROOT ENDPOINT TESTS
# ============================================================================

class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_200(self, test_client):
        """Root endpoint should return 200."""
        response = test_client.get("/")
        assert response.status_code == 200
    
    def test_root_response_structure(self, test_client):
        """Root should return API info."""
        response = test_client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Tests for Pydantic input validation."""
    
    def test_invalid_interest_rate_high(self, test_client):
        """Should reject interest rate > 100."""
        app = {
            "loan_amount": 250000,
            "property_value": 300000,
            "income": 85000,
            "interest_rate": 150,  # Invalid
            "loan_term": 360
        }
        response = test_client.post("/predict", json=app)
        assert response.status_code == 422
    
    def test_invalid_interest_rate_negative(self, test_client):
        """Should reject negative interest rate."""
        app = {
            "loan_amount": 250000,
            "property_value": 300000,
            "income": 85000,
            "interest_rate": -5,  # Invalid
            "loan_term": 360
        }
        response = test_client.post("/predict", json=app)
        assert response.status_code == 422
    
    def test_invalid_loan_term(self, test_client):
        """Should reject invalid loan term."""
        app = {
            "loan_amount": 250000,
            "property_value": 300000,
            "income": 85000,
            "interest_rate": 6.5,
            "loan_term": 0  # Invalid
        }
        response = test_client.post("/predict", json=app)
        assert response.status_code == 422
    
    def test_auto_computed_ratios(self, test_client):
        """LTI and LTV ratios should be auto-computed."""
        app = {
            "loan_amount": 300000,
            "property_value": 400000,  # LTV = 0.75
            "income": 100000,  # LTI = 3.0
            "interest_rate": 6.5,
            "loan_term": 360
        }
        response = test_client.post("/predict", json=app)
        assert response.status_code == 200


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_large_loan(self, test_client):
        """Should handle jumbo loans."""
        app = {
            "loan_amount": 5000000,
            "property_value": 7000000,
            "income": 500000,
            "interest_rate": 7.5,
            "loan_term": 360
        }
        response = test_client.post("/predict", json=app)
        assert response.status_code == 200
    
    def test_minimum_values(self, test_client):
        """Should handle minimum valid values."""
        app = {
            "loan_amount": 1,
            "property_value": 1,
            "income": 1,
            "interest_rate": 0.1,
            "loan_term": 1
        }
        response = test_client.post("/predict", json=app)
        assert response.status_code == 200
    
    def test_with_all_optional_fields(self, test_client):
        """Should handle request with all optional fields."""
        app = {
            "loan_amount": 250000,
            "property_value": 300000,
            "income": 85000,
            "interest_rate": 6.5,
            "loan_term": 360,
            "state_code": "TX",
            "county_code": "201",
            "is_fha_loan": True,
            "is_va_loan": False,
            "has_coborrower": True,
            "applicant_age": 45,
            "rate_spread": 2.5,
            "origination_charges": 3500.00
        }
        response = test_client.post("/predict", json=app)
        assert response.status_code == 200


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
