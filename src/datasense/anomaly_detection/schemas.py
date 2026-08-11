"""Pydantic schemas and contracts for Anomaly Detection Engine."""

from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class AnomalyMethod(str, Enum):
    """Supported Anomaly Detection algorithm methods."""

    ISOLATION_FOREST = "isolation_forest"
    ZSCORE = "zscore"
    IQR = "iqr"
    ENSEMBLE = "ensemble"


class SeverityLevel(str, Enum):
    """Severity classification for detected anomalies."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AnomalyRecordDetail(BaseModel):
    """Details for a single anomalous row observation."""

    row_index: int = Field(..., description="DataFrame row index")
    anomaly_score: float = Field(..., description="Normalized anomaly confidence score (0.0 to 1.0)")
    severity: SeverityLevel = Field(..., description="Severity classification level")
    contributing_features: Dict[str, float] = Field(
        default_factory=dict, description="Top feature deviation scores attributing to anomaly"
    )
    feature_values: Dict[str, Any] = Field(default_factory=dict, description="Raw feature key-value pairs")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnomalyConfig(BaseModel):
    """Configuration options for AnomalyDetector execution."""

    features: Optional[List[str]] = Field(
        default=None, description="Subset of numerical features to evaluate. If empty, all numerical columns are used."
    )
    method: AnomalyMethod = Field(default=AnomalyMethod.ENSEMBLE, description="Algorithm method to execute")
    contamination: float = Field(default=0.05, ge=0.001, le=0.4, description="Expected proportion of anomalies")
    z_threshold: float = Field(default=3.0, ge=1.5, le=5.0, description="Z-score threshold multiplier")
    iqr_multiplier: float = Field(default=1.5, ge=1.0, le=4.0, description="IQR multiplier factor")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnomalyReport(BaseModel):
    """Comprehensive report returned from Anomaly Detection run."""

    run_id: str = Field(..., description="Unique anomaly detection run ID")
    dataset_id: Optional[int] = Field(None, description="Registered dataset ID if applicable")
    method: str = Field(..., description="Executed detection method")
    total_rows: int = Field(..., description="Total rows analyzed")
    affected_rows_count: int = Field(..., description="Count of detected anomalous rows")
    anomaly_percentage: float = Field(..., description="Percentage of dataset flagged as anomalous")
    max_severity: SeverityLevel = Field(..., description="Highest severity level detected")
    affected_row_indices: List[int] = Field(default_factory=list, description="Row indices of anomalies")
    anomalous_records: List[AnomalyRecordDetail] = Field(
        default_factory=list, description="Detailed list of anomalous records"
    )
    feature_importance_ranking: Dict[str, float] = Field(
        default_factory=dict, description="Overall feature contribution score ranking"
    )
    chart_plotly_json: Optional[str] = Field(None, description="Serialized Plotly chart JSON")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp of report"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnomalyRequest(BaseModel):
    """Payload to trigger anomaly detection."""

    dataset_id: Optional[int] = Field(None, description="Registered dataset ID")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Direct records list if not using dataset_id")
    features: Optional[List[str]] = Field(default=None, description="Features to evaluate")
    method: AnomalyMethod = Field(default=AnomalyMethod.ENSEMBLE, description="Algorithm method")
    contamination: float = Field(default=0.05, ge=0.001, le=0.4, description="Contamination rate")
    z_threshold: float = Field(default=3.0, ge=1.5, le=5.0, description="Z-score threshold")
    iqr_multiplier: float = Field(default=1.5, ge=1.0, le=4.0, description="IQR factor")


class AnomalyResponse(BaseModel):
    """Response returned from anomaly detection endpoint."""

    run_id: str = Field(..., description="Unique run ID")
    report: AnomalyReport = Field(..., description="Anomaly report payload")
