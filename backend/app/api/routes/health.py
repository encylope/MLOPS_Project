"""Health and readiness endpoints for orchestration."""

import os
import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.models.schemas import HealthResponse
from app.services.model_service import model_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
)
async def health() -> HealthResponse:
    """Basic liveness check — confirms the API process is running."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded,
        model_version=model_service._model_version if model_service.is_loaded else None,
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check — model must be loaded",
)
async def ready():
    """
    Readiness check for Kubernetes/Docker orchestration.
    Returns 503 if the model has not been loaded yet.
    """
    if not model_service.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "model_not_loaded"},
        )
    return {"status": "ready", "model_version": model_service._model_version}
