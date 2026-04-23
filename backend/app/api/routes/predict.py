"""
Prediction API routes.
POST /api/v1/predict       — single transaction
POST /api/v1/predict/batch — batch transactions
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    PredictionResponse,
    TransactionRequest,
)
from app.services.model_service import model_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict fraud for a single transaction",
)
async def predict_single(transaction: TransactionRequest) -> PredictionResponse:
    """
    Submit a single credit card transaction for fraud classification.

    Returns a fraud probability score, risk level (LOW/MEDIUM/HIGH),
    and the model version that generated the prediction.
    """
    try:
        result = model_service.predict(transaction)
        return result
    except RuntimeError as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during inference.",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch fraud prediction (max 100 transactions)",
)
async def predict_batch(batch: BatchTransactionRequest) -> BatchPredictionResponse:
    """
    Submit up to 100 transactions for batch fraud classification.
    Returns predictions for all transactions plus aggregate statistics.
    """
    try:
        return model_service.predict_batch(batch)
    except RuntimeError as exc:
        logger.error(f"Batch inference error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during batch prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch inference.",
        )


@router.post(
    "/reload-model",
    status_code=status.HTTP_200_OK,
    summary="Hot-reload model from MLflow registry",
)
async def reload_model() -> dict:
    """
    Trigger a hot-reload of the model from MLflow Model Registry.
    Use after promoting a new model version to Production.
    """
    try:
        model_service.load_model()
        return {
            "status": "reloaded",
            "model_version": model_service._model_version,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model reload failed: {exc}",
        )
