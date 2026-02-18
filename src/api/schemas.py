"""
Pydantic schemas for API request/response validation.

Defines data models for:
- LoanApplication: Input features for prediction
- PredictionResponse: Model prediction output
- ExplanationResponse: SHAP-based explanations
- HealthResponse: System health status
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    UNKNOWN = "Unknown"


class IncomeBracket(str, Enum):
    """Income bracket categories."""
    UNDER_50K = "<$50K"
    BRACKET_50_75K = "$50-75K"
    BRACKET_75_100K = "$75-100K"
    BRACKET_100_150K = "$100-150K"
    BRACKET_150_250K = "$150-250K"
    OVER_250K = ">$250K"


class LoanSizeCategory(str, Enum):
    """Loan size categories."""
    UNDER_200K = "<$200K"
    BRACKET_200_300K = "$200-300K"
    BRACKET_300_400K = "$300-400K"
    BRACKET_400_600K = "$400-600K"
    OVER_600K = ">$600K"


class AgeGroup(str, Enum):
    """Applicant age groups."""
    UNDER_25 = "<25"
    AGE_25_34 = "25-34"
    AGE_35_44 = "35-44"
    AGE_45_54 = "45-54"
    AGE_55_64 = "55-64"
    AGE_65_74 = "65-74"
    OVER_74 = ">74"
    NOT_APPLICABLE = "Not applicable"


class FairModel(str, Enum):
    """Available fair ML models for prediction."""
    XGB_FAIR = "xgb_fair"
    RF_FAIR = "rf_fair"
    LR_FAIR = "lr_fair"
    NN_FAIR = "nn_fair"
    GLM_FAIR = "glm_fair"
    FASTERRISK_FAIR = "fasterrisk_fair"
    GOSDT_FAIR = "gosdt_fair"


class LoanApplication(BaseModel):
    """
    Input schema for loan application prediction.
    
    Contains all features required for the fair ML model prediction.
    """
    
    # Core loan characteristics
    loan_amount: float = Field(..., gt=0, description="Loan amount in dollars")
    property_value: float = Field(..., gt=0, description="Property value in dollars")
    income: float = Field(..., ge=0, description="Annual applicant income in dollars")
    interest_rate: float = Field(..., ge=0, le=100, description="Interest rate percentage")
    loan_term: float = Field(default=360, gt=0, le=600, description="Loan term in months")
    
    # Underwriting ratios (computed if not provided)
    loan_to_income_ratio: Optional[float] = Field(None, description="Loan amount / Income ratio")
    loan_to_value_ratio: Optional[float] = Field(None, description="Loan amount / Property value ratio (LTV)")
    
    # Location features
    state_code: Optional[str] = Field(None, min_length=2, max_length=2, description="Two-letter US state code (e.g., NY, CA, TX)")
    county_code: Optional[str] = Field(None, description="County FIPS code")
    
    # Loan type flags
    is_fha_loan: bool = Field(default=False, description="FHA insured loan")
    is_va_loan: bool = Field(default=False, description="VA guaranteed loan")
    
    # Additional characteristics
    has_coborrower: bool = Field(default=False, description="Has co-borrower")
    applicant_age: Optional[int] = Field(None, ge=18, le=100, description="Applicant age in years")
    
    # Optional advanced features (computed internally if not provided)
    origination_charges: Optional[float] = Field(None, ge=0, description="Origination charges in dollars")
    rate_spread: Optional[float] = Field(None, description="Rate spread over prime")
    
    # Model selection
    model_name: Optional[FairModel] = Field(
        default=FairModel.XGB_FAIR,
        description="Fair model to use for prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "loan_amount": 350000,
                "property_value": 450000,
                "income": 120000,
                "interest_rate": 6.5,
                "loan_term": 360,
                "state_code": "CA",
                "is_fha_loan": False,
                "is_va_loan": False,
                "has_coborrower": True,
                "applicant_age": 42
            }
        }
    
    @validator('loan_to_income_ratio', always=True)
    def compute_lti_ratio(cls, v, values):
        """Compute loan-to-income ratio if not provided."""
        if v is None and 'loan_amount' in values and 'income' in values:
            if values['income'] > 0:
                return values['loan_amount'] / values['income']
        return v
    
    @validator('loan_to_value_ratio', always=True)
    def compute_ltv_ratio(cls, v, values):
        """Compute loan-to-value ratio if not provided."""
        if v is None and 'loan_amount' in values and 'property_value' in values:
            if values['property_value'] > 0:
                return values['loan_amount'] / values['property_value']
        return v


class PredictionResponse(BaseModel):
    """
    Response schema for loan approval prediction.
    """
    
    prediction: str = Field(..., description="Prediction: 'Approved' or 'Denied'")
    probability: float = Field(..., ge=0, le=1, description="Approval probability (0-1)")
    confidence: float = Field(..., ge=0, le=100, description="Prediction confidence percentage")
    risk_level: str = Field(..., description="Risk assessment: Low, Moderate, or High")
    
    # Key factors (top 3 features)
    key_factors: List[Dict[str, Union[str, float]]] = Field(
        ..., 
        description="Top factors influencing the decision"
    )
    
    # Metadata
    model_name: str = Field(..., description="Model used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Approved",
                "probability": 0.87,
                "confidence": 74.0,
                "risk_level": "Low Risk",
                "key_factors": [
                    {"feature": "loan_to_value_ratio", "impact": 0.15, "direction": "positive"},
                    {"feature": "income", "impact": 0.12, "direction": "positive"},
                    {"feature": "loan_to_income_ratio", "impact": -0.08, "direction": "negative"}
                ],
                "model_name": "XGBoost Fair",
                "processing_time_ms": 45.3
            }
        }


class ExplanationResponse(BaseModel):
    """
    Response schema for SHAP-based model explanation.
    """
    
    prediction: str = Field(..., description="Prediction: 'Approved' or 'Denied'")
    probability: float = Field(..., ge=0, le=1, description="Approval probability (0-1)")
    base_value: float = Field(..., description="Model baseline (expected value)")
    
    # Feature contributions
    feature_contributions: List[Dict[str, Union[str, float]]] = Field(
        ...,
        description="SHAP values for each feature"
    )
    
    # Top factors summary
    top_positive_factors: List[Dict[str, Union[str, float]]] = Field(
        ...,
        description="Top factors increasing approval likelihood"
    )
    top_negative_factors: List[Dict[str, Union[str, float]]] = Field(
        ...,
        description="Top factors decreasing approval likelihood"
    )
    
    # Human-readable explanation
    explanation_text: str = Field(
        ...,
        description="Human-readable explanation of the decision"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Approved",
                "probability": 0.87,
                "base_value": 0.5,
                "feature_contributions": [
                    {"feature": "loan_to_value_ratio", "value": 0.78, "shap_value": 0.15},
                    {"feature": "income", "value": 120000, "shap_value": 0.12}
                ],
                "top_positive_factors": [
                    {"feature": "loan_to_value_ratio", "shap_value": 0.15}
                ],
                "top_negative_factors": [
                    {"feature": "loan_to_income_ratio", "shap_value": -0.08}
                ],
                "explanation_text": "APPROVED (87% confidence) because: excellent LTV ratio (78%), strong income ($120K)."
            }
        }


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """
    
    status: str = Field(..., description="System status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: str = Field(..., description="Name of loaded model")
    encoder_loaded: bool = Field(..., description="Whether the fair encoder is loaded")
    scaler_loaded: bool = Field(..., description="Whether the feature scaler is loaded")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current server timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "XGBoost Fair",
                "encoder_loaded": True,
                "scaler_loaded": True,
                "version": "1.0.0",
                "timestamp": "2026-02-17T12:00:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch predictions.
    """
    
    applications: List[LoanApplication] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of loan applications (max 100)"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch predictions.
    """
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )
    total_processed: int = Field(..., description="Total applications processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class ErrorResponse(BaseModel):
    """
    Response schema for error responses.
    """
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, str]] = Field(None, description="Additional error details")
