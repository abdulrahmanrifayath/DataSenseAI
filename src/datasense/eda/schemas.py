"""Pydantic schemas for Exploratory Data Analysis (EDA) engine and reports."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class InsightItem(BaseModel):
    category: str = Field(..., description="Category: CORRELATION, DISTRIBUTION, MISSINGNESS, OUTLIER, CATEGORICAL, TEMPORAL, TARGET")
    severity: str = Field(..., description="Severity level: HIGH, MEDIUM, INFO")
    title: str = Field(..., description="Short summary title of the insight")
    description: str = Field(..., description="Detailed explanation of the detected data pattern")
    affected_columns: List[str] = Field(default_factory=list, description="Columns involved in the insight")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quantitative values supporting the insight")


class EDASummary(BaseModel):
    row_count: int = Field(..., description="Total rows in dataset")
    column_count: int = Field(..., description="Total columns in dataset")
    total_cells: int = Field(..., description="Total cells (rows * cols)")
    total_missing_cells: int = Field(..., description="Total missing cells")
    missing_percentage: float = Field(..., description="Percentage of missing cells overall")
    duplicate_rows: int = Field(..., description="Number of duplicate rows")
    memory_usage_bytes: int = Field(..., description="Estimated dataset memory usage in bytes")
    feature_counts: Dict[str, int] = Field(default_factory=dict, description="Count of numerical, categorical, datetime, id columns")


class NumericalStats(BaseModel):
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    skewness: float
    kurtosis: float
    missing_count: int
    zero_count: int


class CategoricalStats(BaseModel):
    count: int
    unique_count: int
    top_value: Optional[str] = None
    top_freq: int
    top_ratio: float
    missing_count: int


class OutlierStats(BaseModel):
    iqr_outliers: int
    zscore_outliers: int
    iqr_lower_bound: float
    iqr_upper_bound: float
    zscore_lower_bound: float
    zscore_upper_bound: float


class EDAReport(BaseModel):
    dataset_name: Optional[str] = Field(default=None, description="Name of dataset")
    summary: EDASummary
    numerical_stats: Dict[str, NumericalStats] = Field(default_factory=dict)
    categorical_stats: Dict[str, CategoricalStats] = Field(default_factory=dict)
    missing_analysis: Dict[str, Any] = Field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    spearman_correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    top_correlation_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    outlier_analysis: Dict[str, OutlierStats] = Field(default_factory=dict)
    category_frequencies: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    time_trends: Dict[str, Any] = Field(default_factory=dict)
    target_analysis: Dict[str, Any] = Field(default_factory=dict)
    charts_plotly_json: Dict[str, str] = Field(default_factory=dict)
    insights: List[InsightItem] = Field(default_factory=list)


class EDARequest(BaseModel):
    target_column: Optional[str] = Field(default=None, description="Optional target column for target-focused analysis")
    sample_size: Optional[int] = Field(default=None, description="Optional max rows to sample for fast analysis")


class EDAResponse(BaseModel):
    id: int
    dataset_id: int
    created_at: str
    target_column: Optional[str] = None
    report: EDAReport
