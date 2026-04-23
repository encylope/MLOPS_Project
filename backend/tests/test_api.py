"""
Unit tests for the Fraud Detection API.
Run: pytest backend/tests/ -v

Test plan:
  TC-001: Health endpoint returns 200
  TC-002: Ready endpoint returns 503 when model not loaded
  TC-003: Prediction schema validation — valid input accepted
  TC-004: Prediction schema validation — negative amount rejected
  TC-005: Risk level computation — low probability → LOW
  TC-006: Risk level computation — high probability → HIGH
  TC-007: Batch endpoint rejects empty list
  TC-008: Batch endpoint rejects lists > 100
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Patch MLflow before importing app to avoid real connection
with patch("mlflow.set_tracking_uri"), patch("app.services.model_service.ModelService.load_model"):
    from app.main import app
    from app.models.schemas import TransactionRequest, PredictionResponse

client = TestClient(app)

VALID_TRANSACTION = {
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


# TC-001
def test_health_endpoint_returns_200():
    """Health check should always return 200 regardless of model state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


# TC-002
def test_ready_endpoint_returns_503_when_model_not_loaded():
    """Readiness check fails when model is not loaded."""
    with patch("app.services.model_service.model_service.is_loaded", False):
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"


# TC-003
def test_valid_transaction_schema_accepted():
    """Valid transaction payload should pass Pydantic validation."""
    txn = TransactionRequest(**VALID_TRANSACTION)
    assert txn.Amount == 149.62
    assert txn.V1 == -1.359807


# TC-004
def test_negative_amount_rejected():
    """Negative transaction amounts should be rejected by Pydantic."""
    invalid = {**VALID_TRANSACTION, "Amount": -10.0}
    with pytest.raises(Exception):
        TransactionRequest(**invalid)


# TC-005
def test_risk_level_low():
    """Probability < 0.3 should produce LOW risk."""
    from app.services.model_service import ModelService
    svc = ModelService()
    assert svc._compute_risk_level(0.1) == "LOW"
    assert svc._compute_risk_level(0.29) == "LOW"


# TC-006
def test_risk_level_high():
    """Probability >= 0.7 should produce HIGH risk."""
    from app.services.model_service import ModelService
    svc = ModelService()
    assert svc._compute_risk_level(0.7) == "HIGH"
    assert svc._compute_risk_level(0.99) == "HIGH"


# TC-007
def test_batch_rejects_empty_list():
    """Batch endpoint should reject empty transaction list."""
    response = client.post("/api/v1/predict/batch", json={"transactions": []})
    assert response.status_code == 422  # Pydantic validation error


# TC-008
def test_batch_rejects_oversized_list():
    """Batch endpoint should reject lists with more than 100 items."""
    transactions = [VALID_TRANSACTION] * 101
    response = client.post("/api/v1/predict/batch", json={"transactions": transactions})
    assert response.status_code == 422


# TC-009: Prediction response schema
def test_prediction_response_schema():
    """PredictionResponse should accept valid risk levels."""
    resp = PredictionResponse(
        transaction_id="test-123",
        is_fraud=True,
        fraud_probability=0.95,
        risk_level="HIGH",
        model_version="1",
        inference_time_ms=5.2,
        amount=149.62,
    )
    assert resp.is_fraud is True
    assert resp.risk_level == "HIGH"
