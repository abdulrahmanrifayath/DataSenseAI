"""Pydantic contracts and data structures for Explainable AI (SHAP) module."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class FeatureContribution(BaseModel):
    """SHAP feature contribution for a specific instance."""

    feature_name: str = Field(..., description="Feature column name")
    feature_value: float = Field(..., description="Raw input value of feature for instance")
    shap_value: float = Field(..., description="Computed SHAP attribution value")
    direction: str = Field(..., description="Contribution direction: 'positive' (pushes prediction higher) or 'negative' (pushes prediction lower)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LocalExplanation(BaseModel):
    """SHAP explanation for a single prediction instance."""

    instance_index: int = Field(..., description="Row index of instance")
    base_value: float = Field(..., description="Model expected base value E[f(X)]")
    prediction_value: float = Field(..., description="Model output prediction f(X_i)")
    top_positive_features: List[FeatureContribution] = Field(default_factory=list, description="Top positive contributing features")
    top_negative_features: List[FeatureContribution] = Field(default_factory=list, description="Top negative contributing features")
    all_contributions: List[FeatureContribution] = Field(default_factory=list, description="All feature contributions sorted by absolute SHAP value")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GlobalFeatureImportance(BaseModel):
    """Global feature importance derived from mean absolute SHAP values."""

    feature_name: str = Field(..., description="Feature column name")
    mean_abs_shap_value: float = Field(..., description="Mean absolute SHAP value across dataset")
    rank: int = Field(..., description="Importance rank (1 = highest impact)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class XAIReport(BaseModel):
    """Comprehensive Explainable AI report."""

    run_id: str = Field(..., description="Unique XAI run ID")
    model_id: Optional[str] = Field(None, description="Trained model ID if applicable")
    task_type: str = Field(..., description="Task type: classification or regression")
    global_importance: List[GlobalFeatureImportance] = Field(default_factory=list, description="Global feature importances")
    sample_local_explanations: List[LocalExplanation] = Field(default_factory=list, description="Sample local per-instance explanations")
    summary_chart_plotly_json: Optional[str] = Field(None, description="Plotly SHAP summary bar chart JSON")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp of report"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class XAIRequest(BaseModel):
    """Payload to trigger SHAP model explanation."""

    model_id: Optional[str] = Field(None, description="Trained model ID from model registry")
    dataset_id: Optional[int] = Field(None, description="Dataset ID for background data matrix X")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Direct feature records list if dataset_id is not provided")
    target_column: Optional[str] = Field(None, description="Target column name")
    instance_indices: Optional[List[int]] = Field(default=[0, 1, 2], description="Specific row indices for local explanations")


class XAIResponse(BaseModel):
    """Response returned from XAI endpoint."""

    run_id: str = Field(..., description="Unique run ID")
    report: XAIReport = Field(..., description="XAI Report payload")
