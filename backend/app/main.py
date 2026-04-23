"""
Credit Card Fraud Detection - FastAPI Backend
Main application entry point.
"""

import logging
import os
import time

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from app.api.routes import predict, health, pipeline
from app.utils.logging_config import setup_logging

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint"],
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total fraud predictions made",
    ["prediction"],  # "fraud" or "legitimate"
)
MODEL_CONFIDENCE = Gauge(
    "fraud_model_confidence_last",
    "Confidence score of the last prediction",
)
ACTIVE_MODEL_VERSION = Gauge(
    "fraud_active_model_version",
    "Currently loaded model version",
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="MLOps-powered fraud detection inference engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (strict — only allow frontend origin) ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Prometheus ASGI sub-app ───────────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])


# ── Middleware: request timing + counting ─────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    return response


# ── Startup: load model from MLflow registry ─────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fraud Detection API...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    logger.info("MLflow tracking URI set.")
    # Model is loaded lazily on first request via ModelService singleton


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Fraud Detection API.")
