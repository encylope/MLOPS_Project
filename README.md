# Credit Card Fraud Detection — MLOps Project

Real-time fraud detection system built with end-to-end MLOps practices.  
**Repository:** github.com/encylope/MLOPS_Project

---

## Architecture Overview

| Layer | Tools |
|-------|-------|
| Data Engineering | Apache Airflow 2.9, DVC 3.67 |
| Experiment Tracking | MLflow 2.13 |
| Model Serving | FastAPI 0.111 + MLflow Model Registry |
| Frontend | React 18 + Vite 5 (served by nginx) |
| Containerisation | Docker + Docker Compose |
| Monitoring | Prometheus 2.51 + Grafana 10.4 |
| CI/CD | GitHub Actions (4 jobs) + DVC |

---

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9598 |
| AUC-PR | 0.8134 |
| F1 Score (fraud class) | 0.7947 |
| Recall (fraud class) | 0.8108 |
| Avg inference latency | ~60ms |
| Unit tests | 9 / 9 passing |

---

## Quick Start

### Prerequisites
- Docker Desktop (running)
- Python 3.11+
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/encylope/MLOPS_Project
cd MLOPS_Project

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r backend/requirements.txt
pip install dvc

# 4. Copy environment file
copy .env.example .env

# 5. Download dataset from Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in data/raw/

# 6. Initialise DVC and track dataset
dvc init
dvc add data\raw\creditcard.csv
git add .dvc data\.gitignore
git commit -m "feat: add raw dataset under DVC"

# 7. Run data pipeline locally
python scripts\validate_data.py
python scripts\preprocess.py
python scripts\feature_engineering.py
python scripts\split_data.py

# 8. Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db \
              --default-artifact-root ./mlflow/artifacts \
              --host 0.0.0.0 --port 5050

# 9. Train the model (in a new terminal)
$env:MLFLOW_TRACKING_URI="http://localhost:5050"
python scripts\train.py --model-type xgboost

# 10. Promote model to production in MLflow UI
# http://localhost:5050 → Models → fraud-detector → Version N → Add alias: production

# 11. Start full Docker stack
docker-compose up -d
```

### Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Fraud Detection Portal | http://localhost:3030 | — |
| Backend API Docs | http://localhost:8800/docs | — |
| MLflow UI | http://localhost:5050 | — |
| Airflow UI | http://localhost:8082 | admin / admin |
| Grafana | http://localhost:3035 | admin / admin |
| Prometheus | http://localhost:9095 | — |

---

## Project Structure

```
MLOPS_Project/
├── backend/                    # FastAPI inference engine
│   ├── app/
│   │   ├── api/routes/         # predict.py, health.py, pipeline.py
│   │   ├── models/schemas.py   # Pydantic request/response models
│   │   ├── services/           # model_service.py (MLflow loader)
│   │   └── utils/              # logging_config.py
│   ├── tests/test_api.py       # 9 unit tests (pytest)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # React web portal
│   ├── src/
│   │   ├── pages/              # PredictPage, PipelinePage, DashboardPage
│   │   └── services/api.js     # Axios API client
│   ├── Dockerfile
│   └── nginx.conf
├── airflow/dags/               # fraud_ingestion_dag.py (4 tasks)
├── scripts/                    # train.py, evaluate.py, preprocess.py,
│                               # validate_data.py, feature_engineering.py,
│                               # split_data.py
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/datasources.yml
├── mlflow/                     # MLflow DB + artifacts (DVC-ignored)
├── data/                       # DVC-tracked datasets
│   ├── raw/creditcard.csv      # Kaggle dataset (144 MB, tracked by DVC)
│   ├── processed/              # Scaled CSV, scaler params, validation report
│   └── features/               # Train/val/test splits, feature baseline
├── metrics/                    # DVC-tracked metrics (JSON, CSV)
├── notebooks/01_eda.ipynb      # Exploratory data analysis
├── docs/                       # HLD, LLD, test plan, user manual
├── .github/workflows/ci.yml    # GitHub Actions CI/CD (4 jobs)
├── dvc.yaml                    # DVC pipeline (6 stages)
├── dvc.lock                    # Locked stage hashes for reproducibility
├── MLproject                   # MLflow project entry points
├── params.yaml                 # Pipeline hyperparameters
├── python_env.yaml             # MLflow environment spec
├── docker-compose.yml          # 6-service stack
└── .env.example                # Environment variable template
```

---

## Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**File:** `creditcard.csv` → place in `data/raw/`

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.1727%) |
| Features | V1–V28 (PCA-transformed), Amount, Time |
| Target | Class (0 = legitimate, 1 = fraud) |
| Train split | 199,166 rows |
| Validation split | 42,678 rows |
| Test split | 42,679 rows |

---

## DVC Pipeline

All six stages are defined in `dvc.yaml`. Run the full pipeline with:

```bash
dvc repro
```

| Stage | Script | Description |
|-------|--------|-------------|
| validate | validate_data.py | Schema check, null detection, fraud rate logging |
| preprocess | preprocess.py | Outlier removal, StandardScaler on Amount+Time |
| feature_engineering | feature_engineering.py | Drift baseline stats saved to JSON |
| split | split_data.py | Stratified 70/15/15 train/val/test split |
| train | train.py | XGBoost + SMOTE + MLflow logging |
| evaluate | evaluate.py | Metrics, ROC curve, test_metrics.json |

---

## Airflow DAG

The `fraud_data_ingestion_pipeline` DAG runs the first four pipeline stages automatically on a daily schedule. Access the Airflow UI at http://localhost:8082 (admin/admin) to trigger or monitor runs.

---

## Reproducibility

Every experiment is reproducible via:

```bash
# Reproduce from a specific git commit
git checkout <commit-hash>
dvc repro

# View experiment in MLflow
# MLflow Run ID is logged at the end of each training run
# e.g. Run ID: 9f25f1461f8e4d04ba3e2383b00fd64b
```

Each training run logs to MLflow:
- **Parameters:** all hyperparameters, SMOTE settings, decision threshold
- **Metrics:** ROC-AUC, AUC-PR, F1, precision, recall, accuracy, latency
- **Artifacts:** confusion matrix, ROC curve, feature importance chart
- **Tags:** git commit hash, DVC hash, dataset name

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Liveness check |
| GET | /ready | Readiness check (503 if model not loaded) |
| POST | /api/v1/predict | Single transaction fraud classification |
| POST | /api/v1/predict/batch | Batch prediction (max 100 transactions) |
| POST | /api/v1/reload-model | Hot-reload model from MLflow registry |
| GET | /metrics | Prometheus metrics endpoint |

Full interactive docs: http://localhost:8800/docs

---

## Monitoring

Prometheus scrapes the backend every 10 seconds. Custom metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_api_requests_total` | Counter | Requests by method, endpoint, status |
| `fraud_api_request_duration_seconds` | Histogram | Request latency |
| `fraud_predictions_total` | Counter | Predictions by outcome |
| `fraud_model_confidence_last` | Gauge | Last prediction confidence |
| `fraud_active_model_version` | Gauge | Current model version |

---

## Running Tests

```bash
# Activate virtual environment first
venv\Scripts\activate

# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest backend/tests/ -v

# Run with coverage
pytest backend/tests/ -v --cov=backend/app
```

Expected output: **9 passed, 0 failed**

---

## CI/CD Pipeline

GitHub Actions runs 4 jobs on every push to `main`:

| Job | Description |
|-----|-------------|
| Lint & Unit Tests | ruff linting + 9 pytest cases |
| DVC Pipeline Status | validates dvc.yaml DAG |
| Build Docker Images | builds backend + frontend images |
| Integration Smoke Test | starts stack, calls /health |

---

