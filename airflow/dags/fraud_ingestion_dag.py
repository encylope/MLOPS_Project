"""
Airflow DAG: Credit Card Fraud Data Ingestion Pipeline

Stages:
  1. validate_raw_data   — schema checks, null detection, class distribution
  2. preprocess_data     — scaling, outlier removal
  3. feature_engineering — feature selection, baseline stats for drift detection
  4. split_dataset       — stratified train/val/test split
  5. trigger_training    — signal DVC to run the training pipeline

Schedule: Daily at midnight (or trigger manually after new data arrives)
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

RAW_DATA_PATH = "/data/raw/creditcard.csv"
PROCESSED_DIR = "/data/processed"
FEATURES_DIR = "/data/features"

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ── Task functions ────────────────────────────────────────────────────────────

def validate_raw_data(**context) -> dict:
    """
    Validate raw CSV: schema, nulls, class distribution, row count.
    Fails the DAG if data quality checks do not pass.
    """
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    # Schema check
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Null check
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values detected:\n{null_counts[null_counts > 0]}")

    # Row count
    if len(df) < 1000:
        raise ValueError(f"Dataset too small: {len(df)} rows")

    # Class distribution
    fraud_rate = df["Class"].mean()
    logger.info(f"Fraud rate: {fraud_rate:.4%} | Total rows: {len(df)}")

    context["ti"].xcom_push(key="row_count", value=len(df))
    context["ti"].xcom_push(key="fraud_rate", value=float(fraud_rate))
    return {"status": "valid", "rows": len(df), "fraud_rate": fraud_rate}


def preprocess_data(**context) -> None:
    """
    Clean and scale features. Save to /data/processed/.
    - Remove extreme outliers (Amount > 99.9th percentile)
    - Standard-scale Amount and Time
    - Save scaler params for inference-time use
    """
    import json
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)

    # Remove extreme outliers from Amount
    amount_cap = df["Amount"].quantile(0.999)
    df = df[df["Amount"] <= amount_cap].copy()
    logger.info(f"Rows after outlier removal: {len(df)}")

    # Scale Amount and Time
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    # Save scaler params for inference
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": ["Amount", "Time"],
    }
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f)

    df.to_csv(os.path.join(PROCESSED_DIR, "creditcard_processed.csv"), index=False)
    logger.info("Preprocessing complete.")


def feature_engineering(**context) -> None:
    """
    Feature engineering and drift baseline computation.
    - Compute baseline statistics (mean, std, min, max) per feature
    - Save baseline for later Prometheus-based drift detection
    """
    import json

    os.makedirs(FEATURES_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "creditcard_processed.csv"))
    feature_cols = [c for c in df.columns if c != "Class"]

    # Compute baseline stats (used later for drift detection)
    baseline = {}
    for col in feature_cols:
        baseline[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "p25": float(df[col].quantile(0.25)),
            "p75": float(df[col].quantile(0.75)),
        }

    with open(os.path.join(FEATURES_DIR, "feature_baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    # Save feature matrix (labels remain in Class column)
    df.to_csv(os.path.join(FEATURES_DIR, "features.csv"), index=False)
    logger.info("Feature engineering and baseline stats saved.")


def split_dataset(**context) -> None:
    """Stratified train/val/test split — 70/15/15."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(os.path.join(FEATURES_DIR, "features.csv"))
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    for name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        split_df = X_split.copy()
        split_df["Class"] = y_split.values
        path = os.path.join(FEATURES_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        logger.info(f"Saved {name}: {len(split_df)} rows | fraud={y_split.sum()}")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="fraud_data_ingestion_pipeline",
    default_args=default_args,
    description="End-to-end data ingestion pipeline for fraud detection",
    schedule_interval="@daily",
    catchup=False,
    tags=["fraud", "mlops", "data-engineering"],
) as dag:

    t1_validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    t2_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t3_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    t4_split = PythonOperator(
        task_id="split_dataset",
        python_callable=split_dataset,
    )

    # Pipeline dependency chain
    t1_validate >> t2_preprocess >> t3_features >> t4_split
