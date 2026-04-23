# High-Level Design Document
## Credit Card Fraud Detection — MLOps Project

---

## 1. Problem Statement

Financial fraud costs billions annually. This system classifies incoming credit card
transactions as **fraudulent or legitimate** in real time using machine learning,
surfacing a probability score and risk tier to a web-based operator portal.

**ML metric targets:** F1 ≥ 0.85 on fraud class, AUC-PR ≥ 0.80  
**Business metric targets:** Inference latency < 200ms, availability > 99%

---

## 2. Architecture Overview

```
[Raw CSV / Kaggle]
        │
        ▼
[Airflow DAG] ──► validate ──► preprocess ──► feature_eng ──► split
        │
        ▼
[DVC Pipeline] tracks all stage outputs as versioned artifacts
        │
        ▼
[Training Script] ──► MLflow Tracking (params, metrics, artifacts)
                  ──► MLflow Model Registry (versioned model)
        │
        ▼
[FastAPI Backend]  ◄──► MLflow Model Server
        │  /predict
        │  /health, /ready
        │  /metrics  (Prometheus)
        ▼
[React Frontend]  (port 3000)
        │
        ▼
[Prometheus] ──► [Grafana] (NRT dashboards)
```

---

## 3. Design Principles

- **Loose coupling:** Frontend and backend communicate exclusively via REST API.
  They are independent Docker services with no shared code or filesystem.
- **Reproducibility:** Every experiment is uniquely identified by (git commit hash,
  MLflow run ID, DVC pipeline hash).
- **Automation:** All pipeline stages (ingest → train → register) run unattended
  via DVC and Airflow.
- **Environment parity:** `MLproject` + Docker Compose ensure identical dev/test
  environments.
- **Observability:** Prometheus scrapes `/metrics` every 10s; Grafana displays
  request rate, fraud rate, latency, and model version in NRT.

---

## 4. Component Descriptions

### 4.1 Data Layer
- **Source:** Kaggle creditcardfraud dataset (284,807 transactions, 492 fraud)
- **DVC:** Tracks `data/raw/creditcard.csv` and all derived artifacts
- **Airflow DAG (`fraud_ingestion_dag`):** Daily schedule — validate → preprocess
  → feature engineer → split

### 4.2 Training Layer
- **Model:** XGBoost (primary) / RandomForest (baseline comparison)
- **Class imbalance:** SMOTE oversampling at 10% ratio on training set
- **MLflow tracking:** Logs params, metrics, confusion matrix, ROC curve,
  feature importance, and model artifact
- **Model Registry:** Model promoted to `Production` stage after review

### 4.3 Serving Layer
- **FastAPI** exposes `/api/v1/predict` (single) and `/api/v1/predict/batch`
- **Model loading:** Lazy-loaded from MLflow registry on first request; hot-reload
  via `/api/v1/reload-model`
- **Pydantic** validates all input/output schemas
- **Docker service:** Separate container from frontend; connected via `fraud_net`

### 4.4 Frontend Layer
- **React + Vite** SPA served by nginx
- **Pages:** Predict | Pipeline | Dashboard
- **No direct ML logic** — all inference delegated to backend via REST

### 4.5 Monitoring Layer
- **Prometheus:** Scrapes `backend:8000/metrics` every 10s
  - `fraud_api_requests_total` (counter, by method/endpoint/status)
  - `fraud_api_request_duration_seconds` (histogram)
  - `fraud_predictions_total` (counter, by fraud/legitimate)
  - `fraud_model_confidence_last` (gauge)
- **Grafana:** Provisioned datasource (Prometheus). Dashboards at port 3001.
- **Alerting:** Error rate > 5% or latency > 200ms triggers alert

---

## 5. Technology Stack

| Concern | Technology |
|---------|-----------|
| API framework | FastAPI 0.111 |
| ML training | XGBoost 2.0, scikit-learn 1.5 |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Experiment tracking | MLflow 2.13 |
| Pipeline orchestration | Apache Airflow 2.9 |
| Data versioning | DVC 3.51 |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| Frontend | React 18, Vite 5, Recharts |
| CI/CD | GitHub Actions |

---

## 6. Security Considerations

- `.env` file excluded from Git (`.gitignore`)
- Backend runs as non-root user in Docker
- CORS restricted to frontend origin
- Sensitive data not logged (no raw transaction values in logs)
- All secrets via environment variables, never hardcoded

---

## 7. Scalability Notes

- FastAPI runs with 2 Uvicorn workers (horizontal scaling via Docker Swarm possible)
- Batch endpoint supports up to 100 transactions per call
- MLflow model can be swapped without restarting the API (hot-reload endpoint)
