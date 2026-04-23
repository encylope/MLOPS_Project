# Test Plan — Credit Card Fraud Detection MLOps

## Acceptance Criteria

| ID | Criterion | Target |
|----|-----------|--------|
| AC-01 | Fraud class F1 score on test set | ≥ 0.85 |
| AC-02 | AUC-ROC | ≥ 0.95 |
| AC-03 | Single inference latency | < 200ms |
| AC-04 | API availability | > 99% uptime |
| AC-05 | All unit tests pass | 100% |
| AC-06 | Docker services start cleanly | All 6 services healthy |

---

## Test Cases

### Unit Tests (backend/tests/test_api.py)

| TC-ID | Description | Input | Expected | Status |
|-------|-------------|-------|----------|--------|
| TC-001 | Health endpoint liveness | GET /health | 200, status=healthy | — |
| TC-002 | Ready returns 503 when model not loaded | GET /ready (no model) | 503 | — |
| TC-003 | Valid transaction schema accepted | Valid 30-field payload | No ValidationError | — |
| TC-004 | Negative amount rejected | Amount=-10 | ValidationError | — |
| TC-005 | Risk LOW for p < 0.3 | probability=0.1 | "LOW" | — |
| TC-006 | Risk HIGH for p ≥ 0.7 | probability=0.9 | "HIGH" | — |
| TC-007 | Batch rejects empty list | transactions=[] | 422 | — |
| TC-008 | Batch rejects > 100 items | 101 transactions | 422 | — |
| TC-009 | PredictionResponse schema valid | Valid response object | No error | — |

### Integration Tests

| TC-ID | Description | Expected |
|-------|-------------|----------|
| TC-101 | Full predict flow end-to-end | 200 with valid PredictionResponse |
| TC-102 | Model reload after new version registered | New version reflected in response |
| TC-103 | Prometheus metrics increment after prediction | Counter > 0 at /metrics |
| TC-104 | Docker Compose all services healthy | All healthchecks pass |
| TC-105 | DVC repro produces deterministic outputs | Same hash on re-run |

### Model Evaluation Tests

| TC-ID | Description | Expected |
|-------|-------------|----------|
| TC-201 | F1 score on test set | ≥ 0.85 |
| TC-202 | AUC-ROC | ≥ 0.95 |
| TC-203 | Precision on fraud class | ≥ 0.80 |
| TC-204 | Recall on fraud class | ≥ 0.80 |
| TC-205 | No data leakage (train/test split check) | No overlapping indices |

### UI Tests (manual)

| TC-ID | Description | Expected |
|-------|-------------|----------|
| TC-301 | Load legitimate sample, submit | Result shows LEGITIMATE, LOW risk |
| TC-302 | Load fraud sample, submit | Result shows FRAUD, HIGH risk |
| TC-303 | Submit with empty fields | Button disabled / validation shown |
| TC-304 | Pipeline page loads tool links | 4 tool cards visible |
| TC-305 | Dashboard charts render | 2 charts visible with data |

---

## Running Tests

```bash
# Unit tests
cd fraud-detection
pytest backend/tests/ -v --cov=backend/app

# Model evaluation
python scripts/evaluate.py --run-id <mlflow-run-id>

# Integration (requires Docker)
docker-compose up -d
pytest backend/tests/integration/ -v
```

## Test Report Template

```
Test Report — {date}
Total test cases: 23
Passed: __
Failed: __
Skipped: __

Failed cases:
  TC-XXX: <description of failure>

Acceptance criteria met: Yes / No
  AC-01 F1 ≥ 0.85:   actual = ___
  AC-02 AUC ≥ 0.95:  actual = ___
  AC-03 Latency < 200ms: actual = ___ms
```
