# Credit Card Fraud Detection — MLOps Project

Real-time fraud detection system built with end-to-end MLOps practices.

## Architecture Overview

| Layer | Tools |
|-------|-------|
| Data Engineering | Apache Airflow, DVC |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI + MLflow Model Server |
| Frontend | React (Vite) |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions + DVC |

## Quick Start

```bash
# 1. Clone & setup
git clone <your-repo-url>
cd fraud-detection
cp .env.example .env

# 2. Download dataset (Kaggle Credit Card Fraud)
python scripts/download_data.py

# 3. Initialize DVC
dvc init
dvc add data/raw/creditcard.csv
git add .dvc data/.gitignore
git commit -m "feat: add raw dataset under DVC"

# 4. Run full stack
docker-compose up --build

# 5. Access services
# Frontend:        http://localhost:3000
# Backend API:     http://localhost:8000/docs
# MLflow UI:       http://localhost:5000
# Airflow UI:      http://localhost:8080
# Grafana:         http://localhost:3001 (admin/admin)
# Prometheus:      http://localhost:9090
```

## Project Structure

```
fraud-detection/
├── backend/               # FastAPI inference engine
│   ├── app/
│   │   ├── api/           # Route handlers
│   │   ├── models/        # Pydantic schemas
│   │   ├── services/      # ML inference, monitoring
│   │   └── utils/         # Logging, helpers
│   ├── tests/             # Unit + integration tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/              # React web portal
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── pages/         # Page views
│   │   └── services/      # API client
│   └── Dockerfile
├── airflow/               # Data pipeline DAGs
│   └── dags/
├── monitoring/            # Prometheus + Grafana config
├── mlflow/                # MLflow artifacts storage
├── data/                  # DVC-tracked data
├── notebooks/             # EDA and experiments
├── scripts/               # Utility scripts
├── .github/workflows/     # CI/CD pipelines
├── dvc.yaml               # DVC pipeline definition
├── MLproject              # MLflow project spec
├── docker-compose.yml
└── docs/                  # Design documents
```

## Dataset

Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
File: `creditcard.csv` → place in `data/raw/`

Features: V1–V28 (PCA-transformed), Amount, Time  
Target: Class (0=legitimate, 1=fraud)  
Size: 284,807 transactions | 492 fraud (0.17%)

## Reproducibility

Every experiment is reproducible via:
- Git commit hash
- MLflow run ID
- DVC pipeline stage hash

```bash
# Reproduce a specific experiment
git checkout <commit-hash>
dvc repro
mlflow run . --run-id <mlflow-run-id>
```

## MLOps Pipeline Stages

```
dvc repro
```

Stages (defined in dvc.yaml):
1. `data_ingestion`  — Airflow triggers, raw → validated
2. `preprocessing`   — Clean, scale, SMOTE oversample
3. `feature_eng`     — Feature selection, drift baseline
4. `train`           — Model training + MLflow logging
5. `evaluate`        — Metrics, threshold tuning
6. `register`        — Push to MLflow model registry
