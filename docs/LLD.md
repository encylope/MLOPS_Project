# Low-Level Design — API Endpoint Specifications
## Credit Card Fraud Detection

---

## Base URL

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:8000` |
| Production (Docker) | `http://backend:8000` (internal) |

---

## Endpoints

### GET /health
**Purpose:** Liveness check — confirms the API process is running.

**Response 200:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3",
  "mlflow_uri": "http://mlflow:5000"
}
```

---

### GET /ready
**Purpose:** Readiness check — returns 503 until model is loaded.

**Response 200:**
```json
{ "status": "ready", "model_version": "3" }
```

**Response 503:**
```json
{ "status": "not_ready", "reason": "model_not_loaded" }
```

---

### POST /api/v1/predict
**Purpose:** Single-transaction fraud classification.

**Request body:**
```json
{
  "V1": -1.359807, "V2": -0.072781, ..., "V28": -0.021053,
  "Amount": 149.62,
  "Time": 0.0
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| V1–V28 | float | required | PCA-transformed features |
| Amount | float | ≥ 0.0 | Transaction amount (USD) |
| Time | float | ≥ 0.0 | Seconds since first transaction |

**Response 200:**
```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_fraud": false,
  "fraud_probability": 0.023456,
  "risk_level": "LOW",
  "model_version": "3",
  "inference_time_ms": 4.7,
  "amount": 149.62
}
```

| Field | Type | Values |
|-------|------|--------|
| transaction_id | string (UUID4) | unique per call |
| is_fraud | boolean | true / false |
| fraud_probability | float [0, 1] | probability of fraud |
| risk_level | string | LOW / MEDIUM / HIGH |
| model_version | string | MLflow model version |
| inference_time_ms | float | latency in ms |
| amount | float | echoed from request |

**Response 422:** Pydantic validation error (missing/invalid fields)  
**Response 503:** Model not loaded / MLflow unreachable

---

### POST /api/v1/predict/batch
**Purpose:** Batch classification for up to 100 transactions.

**Request body:**
```json
{
  "transactions": [
    { "V1": ..., "Amount": 149.62, "Time": 0.0 },
    { "V1": ..., "Amount": 21.50, "Time": 500.0 }
  ]
}
```

Constraints: `1 ≤ len(transactions) ≤ 100`

**Response 200:**
```json
{
  "predictions": [ { ...PredictionResponse... }, ... ],
  "total": 2,
  "fraud_count": 1,
  "processing_time_ms": 12.4
}
```

---

### POST /api/v1/reload-model
**Purpose:** Hot-reload the model from MLflow registry (call after promoting new version).

**Response 200:**
```json
{ "status": "reloaded", "model_version": "4" }
```

**Response 503:** Reload failed (check MLflow connectivity)

---

### GET /api/v1/pipeline/status
**Purpose:** Return DVC pipeline stage statuses.

**Response 200:**
```json
{ "dvc_status": "...", "error": null }
```

---

### GET /api/v1/pipeline/dag
**Purpose:** Return DVC DAG as text.

**Response 200:**
```json
{ "dag": "validate -> preprocess -> feature_engineering -> split -> train -> evaluate" }
```

---

### GET /metrics
**Purpose:** Prometheus metrics endpoint.  
**Format:** Prometheus text exposition format (not JSON)

**Key metrics exposed:**
```
fraud_api_requests_total{method, endpoint, status}
fraud_api_request_duration_seconds{endpoint}
fraud_predictions_total{prediction}     # "fraud" or "legitimate"
fraud_model_confidence_last
fraud_active_model_version
```

---

## Error Response Format (all error codes)
```json
{
  "detail": "Human-readable error description"
}
```

---

## Data Flow Diagram

```
Client (React)
    │  POST /api/v1/predict
    ▼
FastAPI route handler (predict.py)
    │  validates input via Pydantic
    ▼
ModelService.predict()
    │  builds DataFrame from TransactionRequest
    │  calls model.predict_proba()
    │  computes risk tier
    ▼
PredictionResponse (Pydantic)
    │  serialized to JSON
    ▼
Client receives result
    │
    ├── Prometheus counter incremented
    └── Latency histogram observed
```
