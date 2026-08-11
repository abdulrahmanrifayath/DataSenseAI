"""Health Check and System Diagnostics API Router."""

import os
import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import psutil

from configuration.settings import settings
from datasense.database.connection import check_db_health, get_db_session
from datasense.database.models import DatasetMetadata, MLModelRecord
from datasense.utilities.logger import get_logger

logger = get_logger("api.health")

router = APIRouter(tags=["Health & Diagnostics"])


class DatabaseHealthResponse(BaseModel):
    connected: bool
    message: str
    details: Dict[str, Any]


class SystemMetricsResponse(BaseModel):
    cpu_usage_pct: float = Field(..., description="Current system CPU utilization percentage")
    memory_usage_pct: float = Field(..., description="Current system RAM memory utilization percentage")
    disk_usage_pct: float = Field(..., description="Current disk utilization percentage")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Overall health status: healthy, degraded, or unhealthy")
    app_name: str
    version: str
    environment: str
    timestamp: str
    database: DatabaseHealthResponse
    system_metrics: SystemMetricsResponse
    registered_datasets_count: int = Field(0, description="Total registered datasets in database")
    active_models_count: int = Field(0, description="Total trained models stored in model registry")


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Platform System Health Check & Diagnostics",
    description="Returns backend API service health, CPU/RAM/Disk metrics, database connection state, and entity counts.",
)
@router.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Platform System Health Check & Diagnostics (v1)",
    description="Returns backend API service health, CPU/RAM/Disk metrics, database connection state, and entity counts.",
)
def get_system_health(db: Session = Depends(get_db_session)) -> HealthCheckResponse:
    """Check health of the backend API, system resources, and persistent database connection."""
    db_connected, db_message, db_details = check_db_health()

    # Query system resource metrics
    cpu_pct = float(psutil.cpu_percent(interval=None))
    mem_pct = float(psutil.virtual_memory().percent)
    try:
        disk_pct = float(psutil.disk_usage(os.getcwd()).percent)
    except Exception:
        disk_pct = 0.0

    # Query database entity counts
    dataset_cnt = 0
    model_cnt = 0
    if db_connected:
        try:
            dataset_cnt = db.query(DatasetMetadata).count()
            model_cnt = db.query(MLModelRecord).count()
        except Exception as e:
            logger.warning(f"Error querying entity counts for health check: {e}")

    overall_status = "healthy" if db_connected else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        app_name=settings.APP_NAME,
        version=settings.API_VERSION,
        environment=settings.APP_ENV,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        database=DatabaseHealthResponse(
            connected=db_connected,
            message=db_message,
            details=db_details,
        ),
        system_metrics=SystemMetricsResponse(
            cpu_usage_pct=cpu_pct,
            memory_usage_pct=mem_pct,
            disk_usage_pct=disk_pct,
        ),
        registered_datasets_count=dataset_cnt,
        active_models_count=model_cnt,
    )
