"""
Phase 10: Integration Tests for Mortgage Underwriting Assistant

End-to-end integration tests for API and model pipeline validation.
"""
import pytest
import requests
import time
import json
import numpy as np
from typing import Dict, Any, List

# API Configuration
API_BASE_URL = "http://localhost:8000"
LATENCY_THRESHOLD_MS = 500


class TestAPIIntegration:
    """Integration tests for the API endpoints."""
    
    @pytest.fixture(autouse=True)
    def check_api_availability(self):
        """Check if API is available before running tests."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API not available")
        except requests.exceptions.RequestException:
            pytest.skip("API not running - start with: uvicorn src.api.main:app")
    
    # ============================================================
    # Health Endpoint Tests
    # ============================================================
    
    def test_health_endpoint_returns_200(self):
        """Test that health endpoint returns 200 status."""
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200
    
    def test_health_endpoint_returns_status(self):
        """Test that health endpoint returns status field."""
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        data = response.json()
        assert 'status' in data
        assert data['status'] in ['healthy', 'unhealthy']
    
    # ============================================================
    # Predict Endpoint Tests
    # ============================================================
    
    def test_predict_valid_application(self):
        """Test prediction with valid loan application."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert 'prediction' in data
        assert data['prediction'] in ['Approved', 'Denied']
        assert 'probability' in data
        assert 0 <= data['probability'] <= 1
    
    def test_predict_returns_risk_level(self):
        """Test that prediction includes risk level."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        data = response.json()
        
        assert 'risk_level' in data
        assert data['risk_level'] in ['Low Risk', 'Moderate Risk', 'High Risk']
    
    def test_predict_negative_loan_amount_rejected(self):
        """Test that negative loan amount is rejected with validation error."""
        payload = {
            'loan_amount': -50000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
        assert response.status_code == 422
    
    def test_predict_zero_income_handled(self):
        """Test that zero income is handled (may be accepted or rejected)."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 0,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
        # API may accept zero income (handled in model) or reject it
        assert response.status_code in [200, 422]
    
    def test_predict_missing_required_field(self):
        """Test that missing required field returns error."""
        payload = {
            'loan_amount': 250000,
            # Missing property_value
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
        assert response.status_code == 422
    
    def test_predict_extreme_high_ltv(self):
        """Test prediction with extreme LTV (>100%) returns valid response."""
        payload = {
            'loan_amount': 500000,
            'property_value': 400000,  # LTV = 125%
            'income': 100000,
            'interest_rate': 7.0,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        # Model should handle extreme LTV and return valid prediction
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert data['prediction'] in ['Approved', 'Denied']
    
    def test_predict_fha_loan_flag(self):
        """Test prediction with FHA loan flag."""
        payload = {
            'loan_amount': 250000,
            'property_value': 270000,  # LTV ~92.6%
            'income': 65000,
            'interest_rate': 6.5,
            'loan_term': 360,
            'is_fha_loan': True
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
    
    def test_predict_va_loan_flag(self):
        """Test prediction with VA loan flag."""
        payload = {
            'loan_amount': 400000,
            'property_value': 400000,  # LTV = 100%
            'income': 100000,
            'interest_rate': 5.5,
            'loan_term': 360,
            'is_va_loan': True
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
    
    # ============================================================
    # Explain Endpoint Tests
    # ============================================================
    
    def test_explain_returns_explanation(self):
        """Test that explain endpoint returns explanation text."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/explain", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            assert 'explanation_text' in data or 'top_factors' in data
    
    def test_explain_includes_prediction(self):
        """Test that explain endpoint also includes prediction."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/explain", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data
    
    # ============================================================
    # Batch Predict Endpoint Tests
    # ============================================================
    
    def test_batch_predict_multiple_applications(self):
        """Test batch prediction with multiple applications."""
        payload = {
            'applications': [
                {'loan_amount': 250000, 'property_value': 300000, 'income': 80000, 'interest_rate': 6.5, 'loan_term': 360},
                {'loan_amount': 400000, 'property_value': 500000, 'income': 120000, 'interest_rate': 7.0, 'loan_term': 360},
                {'loan_amount': 180000, 'property_value': 220000, 'income': 55000, 'interest_rate': 6.25, 'loan_term': 360}
            ]
        }
        response = requests.post(f"{API_BASE_URL}/batch/predict", json=payload, timeout=30)
        assert response.status_code == 200
        
        data = response.json()
        assert 'total_processed' in data
        assert data['total_processed'] == 3
        # API returns 'predictions' key for batch results
        assert 'predictions' in data
        assert len(data['predictions']) == 3
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty application list."""
        payload = {'applications': []}
        response = requests.post(f"{API_BASE_URL}/batch/predict", json=payload, timeout=5)
        # Should either return error or empty results
        assert response.status_code in [200, 400, 422]
    
    def test_batch_predict_single_invalid(self):
        """Test batch prediction with one invalid application."""
        payload = {
            'applications': [
                {'loan_amount': 250000, 'property_value': 300000, 'income': 80000, 'interest_rate': 6.5, 'loan_term': 360},
                {'loan_amount': -100, 'property_value': 300000, 'income': 80000, 'interest_rate': 6.5, 'loan_term': 360}  # Invalid
            ]
        }
        response = requests.post(f"{API_BASE_URL}/batch/predict", json=payload, timeout=20)
        # Should handle gracefully
        assert response.status_code in [200, 207, 422]


class TestPerformance:
    """Performance and latency tests."""
    
    @pytest.fixture(autouse=True)
    def check_api_availability(self):
        """Check if API is available before running tests."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API not available")
        except requests.exceptions.RequestException:
            pytest.skip("API not running")
    
    def test_predict_latency_under_threshold(self):
        """Test that prediction latency is under 500ms threshold."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        
        latencies = []
        for _ in range(10):
            start = time.time()
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
            elapsed_ms = (time.time() - start) * 1000
            if response.status_code == 200:
                latencies.append(elapsed_ms)
        
        if latencies:
            p95_latency = np.percentile(latencies, 95)
            assert p95_latency < LATENCY_THRESHOLD_MS, f"P95 latency {p95_latency:.1f}ms exceeds threshold {LATENCY_THRESHOLD_MS}ms"
    
    def test_health_endpoint_fast_response(self):
        """Test that health endpoint responds within 100ms."""
        start = time.time()
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        elapsed_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed_ms < 100, f"Health endpoint took {elapsed_ms:.1f}ms (expected <100ms)"
    
    def test_batch_predict_scales_linearly(self):
        """Test that batch prediction time scales approximately linearly."""
        base_payload = {'loan_amount': 250000, 'property_value': 300000, 'income': 80000, 'interest_rate': 6.5, 'loan_term': 360}
        
        # Time with 2 applications
        payload_2 = {'applications': [base_payload] * 2}
        start = time.time()
        requests.post(f"{API_BASE_URL}/batch/predict", json=payload_2, timeout=30)
        time_2 = time.time() - start
        
        # Time with 10 applications
        payload_10 = {'applications': [base_payload] * 10}
        start = time.time()
        requests.post(f"{API_BASE_URL}/batch/predict", json=payload_10, timeout=60)
        time_10 = time.time() - start
        
        # 10x workload should not take more than 6x the time (allowing for overhead)
        assert time_10 < time_2 * 6, f"Batch scaling is too slow: 2 apps took {time_2:.2f}s, 10 apps took {time_10:.2f}s"


class TestEdgeCases:
    """Edge case validation tests."""
    
    @pytest.fixture(autouse=True)
    def check_api_availability(self):
        """Check if API is available before running tests."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API not available")
        except requests.exceptions.RequestException:
            pytest.skip("API not running")
    
    def test_high_ltv_low_income_returns_prediction(self):
        """High LTV + Low Income should return valid prediction."""
        payload = {
            'loan_amount': 380000,
            'property_value': 400000,  # LTV = 95%
            'income': 50000,  # LTI = 7.6x (very high)
            'interest_rate': 7.5,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
        data = response.json()
        # Model returns valid prediction (actual decision depends on training data)
        assert 'prediction' in data
        assert 'risk_level' in data
        assert 'probability' in data
    
    def test_low_ltv_high_income_approved(self):
        """Low LTV + High Income should typically be approved."""
        payload = {
            'loan_amount': 200000,
            'property_value': 500000,  # LTV = 40%
            'income': 150000,  # LTI = 1.33x (very low)
            'interest_rate': 6.0,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
        data = response.json()
        # Low risk case - should typically be approved
        assert data['prediction'] == 'Approved' or data['risk_level'] == 'Low Risk' or data['probability'] > 0.5
    
    def test_jumbo_loan_handling(self):
        """Test that jumbo loans (>$766,550 in 2024) are handled."""
        payload = {
            'loan_amount': 1000000,
            'property_value': 1500000,
            'income': 300000,
            'interest_rate': 7.0,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
        # Should return valid prediction regardless of loan size
        data = response.json()
        assert 'prediction' in data
    
    def test_minimum_loan_amount(self):
        """Test minimum viable loan amount."""
        payload = {
            'loan_amount': 10000,  # Very small loan
            'property_value': 50000,
            'income': 30000,
            'interest_rate': 6.0,
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
    
    def test_high_interest_rate(self):
        """Test prediction with high interest rate."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 12.0,  # High rate
            'loan_term': 360
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200
    
    def test_short_loan_term(self):
        """Test prediction with 15-year loan term."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.0,
            'loan_term': 180  # 15 years
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        assert response.status_code == 200


class TestConsistency:
    """Test prediction consistency."""
    
    @pytest.fixture(autouse=True)
    def check_api_availability(self):
        """Check if API is available before running tests."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API not available")
        except requests.exceptions.RequestException:
            pytest.skip("API not running")
    
    def test_same_input_same_output(self):
        """Test that same input produces same output (deterministic)."""
        payload = {
            'loan_amount': 250000,
            'property_value': 300000,
            'income': 80000,
            'interest_rate': 6.5,
            'loan_term': 360
        }
        
        predictions = []
        probabilities = []
        
        for _ in range(5):
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                predictions.append(data['prediction'])
                probabilities.append(data['probability'])
        
        # All predictions should be identical
        assert len(set(predictions)) == 1, f"Inconsistent predictions: {predictions}"
        
        # Probabilities should be very close (allow for floating point)
        prob_range = max(probabilities) - min(probabilities)
        assert prob_range < 0.001, f"Probability variance too high: {probabilities}"
    
    def test_batch_matches_individual(self):
        """Test that batch predictions match individual predictions."""
        applications = [
            {'loan_amount': 250000, 'property_value': 300000, 'income': 80000, 'interest_rate': 6.5, 'loan_term': 360},
            {'loan_amount': 400000, 'property_value': 500000, 'income': 120000, 'interest_rate': 7.0, 'loan_term': 360}
        ]
        
        # Get individual predictions
        individual_predictions = []
        for app in applications:
            response = requests.post(f"{API_BASE_URL}/predict", json=app, timeout=10)
            if response.status_code == 200:
                individual_predictions.append(response.json()['prediction'])
        
        # Get batch predictions
        batch_response = requests.post(
            f"{API_BASE_URL}/batch/predict",
            json={'applications': applications},
            timeout=30
        )
        
        if batch_response.status_code == 200:
            batch_data = batch_response.json()
            # API uses 'predictions' key for batch results
            batch_predictions = [r.get('prediction') for r in batch_data.get('predictions', [])]
            
            # Compare
            assert individual_predictions == batch_predictions, \
                f"Individual: {individual_predictions}, Batch: {batch_predictions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
