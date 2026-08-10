"""Health Check API Router."""

import datetime
from typing import Dict, Any
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from configuration.settings import settings
from datasense.database.connection import check_db_health
from datasense.utilities.logger import get_logger

logger = get_logger("api.health")

router = APIRouter(tags=["Health & Diagnostics"])


class DatabaseHealthResponse(BaseModel):
    connected: bool
    message: str
    details: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Overall health status: healthy, degraded, or unhealthy")
    app_name: str
    version: str
    environment: str
    timestamp: str
    database: DatabaseHealthResponse


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Platform System Health Check",
    description="Returns backend API service health, version metadata, and PostgreSQL database connection state.",
)
def get_system_health() -> HealthCheckResponse:
    """Check health of the backend API and persistent database connection."""
    db_connected, db_message, db_details = check_db_health()

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
    )
