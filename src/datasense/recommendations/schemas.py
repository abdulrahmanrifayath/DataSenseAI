"""Pydantic contracts and data structures for Business Recommendation Engine."""

from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class RecommendationPriority(str, Enum):
    """Priority / Severity levels for actionable business recommendations."""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class RecommendationItem(BaseModel):
    """Individual actionable business recommendation."""

    id: str = Field(..., description="Unique recommendation item ID")
    title: str = Field(..., description="Short descriptive title of recommendation")
    explanation: str = Field(..., description="Detailed natural language rationale and context")
    evidence: str = Field(..., description="Empirical computed data proof backing recommendation")
    severity_priority: RecommendationPriority = Field(..., description="Priority severity level")
    affected_metric: str = Field(..., description="Key business/data metric impacted")
    suggested_action: str = Field(..., description="Concrete, actionable recommended next step")
    source_module: str = Field(..., description="Origin module: EDA, ML, Forecasting, Anomaly, BI, SHAP")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RecommendationReport(BaseModel):
    """Comprehensive Business Recommendation Report."""

    run_id: str = Field(..., description="Unique recommendation run ID")
    dataset_id: Optional[int] = Field(None, description="Registered dataset ID if applicable")
    total_recommendations: int = Field(..., description="Total count of generated recommendations")
    critical_count: int = Field(..., description="Count of critical priority recommendations")
    high_count: int = Field(..., description="Count of high priority recommendations")
    items: List[RecommendationItem] = Field(default_factory=list, description="List of actionable recommendations")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp of report"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RecommendationRequest(BaseModel):
    """Payload to trigger business recommendation synthesis."""

    dataset_id: Optional[int] = Field(None, description="Registered dataset ID")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Direct feature records if dataset_id is not provided")
    eda_report: Optional[Dict[str, Any]] = Field(None, description="Computed EDA report if available")
    ml_report: Optional[Dict[str, Any]] = Field(None, description="Computed ML model report if available")
    forecast_report: Optional[Dict[str, Any]] = Field(None, description="Computed time-series forecast report if available")
    anomaly_report: Optional[Dict[str, Any]] = Field(None, description="Computed anomaly detection report if available")
    bi_report: Optional[Dict[str, Any]] = Field(None, description="Computed BI report if available")
    xai_report: Optional[Dict[str, Any]] = Field(None, description="Computed XAI / SHAP report if available")


class RecommendationResponse(BaseModel):
    """Response payload returned from recommendation endpoint."""

    run_id: str = Field(..., description="Unique run ID")
    report: RecommendationReport = Field(..., description="Recommendation report payload")
