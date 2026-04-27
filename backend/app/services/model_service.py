"""
Model service — singleton that loads the fraud detection model
from MLflow Model Registry and runs inference.
"""

import logging
import os
import re
import time
import uuid
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from app.models.schemas import (
    PredictionResponse,
    TransactionRequest,
    BatchTransactionRequest,
    BatchPredictionResponse,
)

logger = logging.getLogger(__name__)

# Feature order must match training pipeline
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Risk thresholds
RISK_LOW_THRESHOLD = 0.3
RISK_HIGH_THRESHOLD = 0.7


class ModelService:
    _instance: Optional["ModelService"] = None
    _model = None
    _model_version: str = "unknown"

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        model_name = os.getenv("MODEL_NAME", "fraud-detector")
        model_stage = os.getenv("MODEL_STAGE", "Production")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

        mlflow.set_tracking_uri(tracking_uri)

        logger.info(f"Loading model for {model_name}@{model_stage.lower()}")
        try:
            client = mlflow.MlflowClient()
            mv = client.get_model_version_by_alias(model_name, model_stage.lower())
            self._model_version = mv.version

            # Fix Windows path to Docker Linux path
            source = mv.source
            source = source.replace("file:///", "").replace("file:", "")
            source = re.sub(r"[A-Za-z]:[/\\].*?mlflow", "/mlflow", source)
            source = source.replace("\\", "/")

            logger.info(f"Loading model from fixed path: {source}")
            self._model = mlflow.sklearn.load_model(source)
            logger.info(f"Model v{self._model_version} loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load model from registry: {exc}")
            raise RuntimeError(f"Model load failed: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load_model()

    def _build_feature_vector(self, transaction: TransactionRequest) -> pd.DataFrame:
        data = {col: getattr(transaction, col) for col in FEATURE_COLUMNS}
        return pd.DataFrame([data], columns=FEATURE_COLUMNS)

    def _compute_risk_level(self, probability: float) -> str:
        if probability < RISK_LOW_THRESHOLD:
            return "LOW"
        if probability < RISK_HIGH_THRESHOLD:
            return "MEDIUM"
        return "HIGH"

    def predict(self, transaction: TransactionRequest) -> PredictionResponse:
        self._ensure_loaded()
        features = self._build_feature_vector(transaction)
        start = time.perf_counter()
        proba = self._model.predict_proba(features)[0][1]
        latency_ms = (time.perf_counter() - start) * 1000
        is_fraud = bool(proba >= 0.5)
        risk = self._compute_risk_level(float(proba))
        logger.info(
            f"Prediction: fraud={is_fraud} proba={proba:.4f} "
            f"latency={latency_ms:.2f}ms model_v={self._model_version}"
        )
        return PredictionResponse(
            transaction_id=str(uuid.uuid4()),
            is_fraud=is_fraud,
            fraud_probability=round(float(proba), 6),
            risk_level=risk,
            model_version=self._model_version,
            inference_time_ms=round(latency_ms, 3),
            amount=transaction.Amount,
        )

    def predict_batch(self, batch: BatchTransactionRequest) -> BatchPredictionResponse:
        self._ensure_loaded()
        features_df = pd.concat(
            [self._build_feature_vector(t) for t in batch.transactions],
            ignore_index=True,
        )
        start = time.perf_counter()
        probas = self._model.predict_proba(features_df)[:, 1]
        total_ms = (time.perf_counter() - start) * 1000
        predictions = []
        for txn, proba in zip(batch.transactions, probas):
            is_fraud = bool(proba >= 0.5)
            predictions.append(
                PredictionResponse(
                    transaction_id=str(uuid.uuid4()),
                    is_fraud=is_fraud,
                    fraud_probability=round(float(proba), 6),
                    risk_level=self._compute_risk_level(float(proba)),
                    model_version=self._model_version,
                    inference_time_ms=round(total_ms / len(batch.transactions), 3),
                    amount=txn.Amount,
                )
            )
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            fraud_count=fraud_count,
            processing_time_ms=round(total_ms, 3),
        )


model_service = ModelService()