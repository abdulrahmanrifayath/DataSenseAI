"""Pydantic schemas and data contracts for Time-Series Forecasting."""

from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class ForecastModelType(str, Enum):
    """Supported Forecasting model algorithms."""

    BASELINE = "baseline"
    PROPHET = "prophet"
    XGBOOST = "xgboost"


class FrequencyOption(str, Enum):
    """Time-series frequency sampling grid options."""

    AUTO = "auto"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    HOURLY = "H"


class ForecastMetrics(BaseModel):
    """Evaluation metrics for time-series forecasting."""

    mae: float = Field(..., description="Mean Absolute Error")
    mse: float = Field(..., description="Mean Squared Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error (%)")
    smape: Optional[float] = Field(None, description="Symmetric Mean Absolute Percentage Error (%)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ForecastItem(BaseModel):
    """Individual forecast point with confidence interval."""

    timestamp: str = Field(..., description="ISO datetime string for prediction step")
    predicted_value: float = Field(..., description="Point forecast value")
    lower_bound: Optional[float] = Field(None, description="Lower confidence interval boundary")
    upper_bound: Optional[float] = Field(None, description="Upper confidence interval boundary")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelForecastResult(BaseModel):
    """Forecasting result for a single algorithm."""

    model_type: str = Field(..., description="Algorithm identifier key")
    model_name: str = Field(..., description="Human-readable model name")
    test_metrics: ForecastMetrics = Field(..., description="Holdout test set evaluation metrics")
    future_forecast: List[ForecastItem] = Field(default_factory=list, description="Future predictions horizon list")
    is_best: bool = Field(False, description="Whether this model achieved lowest forecast RMSE")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ForecastingConfig(BaseModel):
    """Configuration for ForecastingEngine execution."""

    date_column: str = Field(..., description="Datetime column name")
    target_column: str = Field(..., description="Numerical target column name to forecast")
    forecast_horizon: int = Field(default=30, ge=1, le=365, description="Number of future periods to predict")
    frequency: str = Field(default="auto", description="Frequency grid: auto, D, W, M, H")
    selected_models: Optional[List[str]] = Field(
        default=None, description="Subset of models to run: baseline, prophet, xgboost"
    )
    test_size_ratio: float = Field(default=0.2, ge=0.05, le=0.4, description="Holdout test set ratio")
    confidence_level: float = Field(default=0.95, ge=0.80, le=0.99, description="Confidence interval coverage")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ForecastingReport(BaseModel):
    """Comprehensive Time-Series Forecasting report."""

    run_id: str = Field(..., description="Unique forecasting run ID")
    dataset_id: Optional[int] = Field(None, description="Registered dataset ID if applicable")
    date_column: str = Field(..., description="Resolved date column name")
    target_column: str = Field(..., description="Resolved target column name")
    inferred_frequency: str = Field(..., description="Detected sampling frequency code")
    total_records: int = Field(..., description="Total rows in clean time series")
    train_records: int = Field(..., description="Training split sample count")
    test_records: int = Field(..., description="Holdout test split sample count")
    forecast_horizon: int = Field(..., description="Future forecast step count")
    missing_dates_detected: int = Field(0, description="Count of imputed/filled missing date gaps")
    results: List[ModelForecastResult] = Field(default_factory=list, description="Results per forecasting model")
    best_model_name: str = Field(..., description="Model name of top performer")
    chart_plotly_json: Optional[str] = Field(None, description="Serialized Plotly chart JSON")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp of report"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ForecastRequest(BaseModel):
    """Payload to trigger time-series forecasting."""

    dataset_id: Optional[int] = Field(None, description="Registered dataset ID")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Direct feature records if not using dataset_id")
    date_column: str = Field(..., description="Datetime column name")
    target_column: str = Field(..., description="Numerical target column name")
    forecast_horizon: int = Field(default=30, ge=1, le=365, description="Future forecast horizon")
    frequency: str = Field(default="auto", description="Frequency selection")
    selected_models: Optional[List[str]] = Field(default=None, description="Subset of models to run")


class ForecastResponse(BaseModel):
    """Response returned from forecasting endpoint."""

    run_id: str = Field(..., description="Unique forecasting run ID")
    report: ForecastingReport = Field(..., description="Forecasting report")
