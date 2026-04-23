"""
Pydantic models for request/response validation.
All field names match the Kaggle creditcard.csv schema.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """
    Single transaction input for fraud prediction.
    V1-V28 are PCA-transformed features from the original dataset.
    """

    # PCA features (anonymized)
    V1: float = Field(..., description="PCA feature 1")
    V2: float = Field(..., description="PCA feature 2")
    V3: float = Field(..., description="PCA feature 3")
    V4: float = Field(..., description="PCA feature 4")
    V5: float = Field(..., description="PCA feature 5")
    V6: float = Field(..., description="PCA feature 6")
    V7: float = Field(..., description="PCA feature 7")
    V8: float = Field(..., description="PCA feature 8")
    V9: float = Field(..., description="PCA feature 9")
    V10: float = Field(..., description="PCA feature 10")
    V11: float = Field(..., description="PCA feature 11")
    V12: float = Field(..., description="PCA feature 12")
    V13: float = Field(..., description="PCA feature 13")
    V14: float = Field(..., description="PCA feature 14")
    V15: float = Field(..., description="PCA feature 15")
    V16: float = Field(..., description="PCA feature 16")
    V17: float = Field(..., description="PCA feature 17")
    V18: float = Field(..., description="PCA feature 18")
    V19: float = Field(..., description="PCA feature 19")
    V20: float = Field(..., description="PCA feature 20")
    V21: float = Field(..., description="PCA feature 21")
    V22: float = Field(..., description="PCA feature 22")
    V23: float = Field(..., description="PCA feature 23")
    V24: float = Field(..., description="PCA feature 24")
    V25: float = Field(..., description="PCA feature 25")
    V26: float = Field(..., description="PCA feature 26")
    V27: float = Field(..., description="PCA feature 27")
    V28: float = Field(..., description="PCA feature 28")

    # Non-anonymized features
    Amount: float = Field(..., ge=0.0, description="Transaction amount in USD")
    Time: float = Field(..., ge=0.0, description="Seconds elapsed since first transaction")

    model_config = {"json_schema_extra": {
        "example": {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
            "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
            "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
            "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
            "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
            "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
            "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
            "V28": -0.021053, "Amount": 149.62, "Time": 0.0,
        }
    }}


class PredictionResponse(BaseModel):
    """Fraud prediction result with confidence and metadata."""

    transaction_id: str = Field(..., description="Unique prediction ID")
    is_fraud: bool = Field(..., description="True if transaction is predicted fraudulent")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud [0–1]")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    model_version: str = Field(..., description="MLflow model version used")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    amount: float = Field(..., description="Transaction amount echoed back")

    @field_validator("risk_level")
    @classmethod
    def validate_risk(cls, v: str) -> str:
        if v not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("risk_level must be LOW, MEDIUM, or HIGH")
        return v


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: Optional[str]
    mlflow_uri: str


class BatchTransactionRequest(BaseModel):
    """Batch prediction for multiple transactions."""

    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse]
    total: int
    fraud_count: int
    processing_time_ms: float
