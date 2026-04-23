"""Pipeline info routes — exposes pipeline status for the ML Pipeline UI screen."""

import logging
import os
import subprocess

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/pipeline/status", summary="DVC pipeline status")
async def pipeline_status() -> dict:
    """Return the current DVC pipeline stage statuses."""
    try:
        result = subprocess.run(
            ["dvc", "status", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        return {"dvc_status": result.stdout, "error": result.stderr or None}
    except FileNotFoundError:
        return {"dvc_status": None, "error": "DVC not found in container PATH"}
    except subprocess.TimeoutExpired:
        return {"dvc_status": None, "error": "DVC status timed out"}


@router.get("/pipeline/dag", summary="DVC DAG structure")
async def pipeline_dag() -> dict:
    """Return the DVC DAG as text for visualization."""
    try:
        result = subprocess.run(
            ["dvc", "dag"],
            capture_output=True, text=True, timeout=10,
        )
        return {"dag": result.stdout}
    except Exception as exc:
        return {"dag": None, "error": str(exc)}
