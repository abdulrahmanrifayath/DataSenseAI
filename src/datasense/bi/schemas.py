"""Pydantic contracts and data structures for Business Intelligence and Customer Analytics."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class ColumnMappingConfig(BaseModel):
    """Configurable column mapping aliases for flexible schema recognition."""

    customer_id_col: Optional[str] = Field(None, description="Column identifying customer ID")
    order_date_col: Optional[str] = Field(None, description="Column identifying transaction date")
    revenue_col: Optional[str] = Field(None, description="Column identifying sales/revenue value")
    profit_col: Optional[str] = Field(None, description="Column identifying profit/margin value")
    quantity_col: Optional[str] = Field(None, description="Column identifying purchase quantity")
    churn_col: Optional[str] = Field(None, description="Column identifying churn status (0/1 or True/False)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BusinessKPIs(BaseModel):
    """Executive Business KPIs."""

    total_revenue: float = Field(0.0, description="Total monetary sales revenue")
    total_profit: Optional[float] = Field(None, description="Total net profit")
    average_order_value: float = Field(0.0, description="Average Order Value (AOV)")
    total_customers: int = Field(0, description="Total unique customer count")
    repeat_purchase_rate: float = Field(0.0, description="Percentage of repeat customers (%)")
    profit_margin_pct: Optional[float] = Field(None, description="Overall profit margin (%)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RFMSegmentSummary(BaseModel):
    """Summary metrics per RFM persona segment."""

    segment_name: str = Field(..., description="RFM Segment Name (e.g., Champions, At Risk)")
    customer_count: int = Field(..., description="Count of customers in segment")
    avg_recency_days: float = Field(..., description="Average days since last purchase")
    avg_frequency: float = Field(..., description="Average order count")
    avg_monetary_value: float = Field(..., description="Average customer spend")
    total_revenue_share_pct: float = Field(..., description="Percentage of total revenue contributed by segment")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusterEvaluationMetrics(BaseModel):
    """Automatic cluster quality evaluation metrics."""

    algorithm: str = Field(..., description="Clustering algorithm executed (K-Means / Hierarchical)")
    optimal_k: int = Field(..., description="Optimal number of clusters")
    silhouette_score: float = Field(..., description="Silhouette Score (-1 to +1, higher is better)")
    davies_bouldin_index: float = Field(..., description="Davies-Bouldin Index (lower is better)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CustomerSegmentDetail(BaseModel):
    """Customer cluster profile details."""

    cluster_id: int = Field(..., description="Cluster numerical index")
    segment_label: str = Field(..., description="Descriptive segment label")
    customer_count: int = Field(..., description="Count of customers in cluster")
    mean_recency: float = Field(..., description="Cluster mean recency in days")
    mean_frequency: float = Field(..., description="Cluster mean purchase frequency")
    mean_monetary: float = Field(..., description="Cluster mean total spend")
    revenue_share_pct: float = Field(..., description="Cluster contribution to total revenue (%)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChurnRiskSummary(BaseModel):
    """Customer Churn Risk Prediction Summary."""

    overall_churn_rate_pct: float = Field(..., description="Overall churn rate percentage (%)")
    high_risk_customer_count: int = Field(..., description="Count of customers with >70% churn risk")
    top_churn_drivers: Dict[str, float] = Field(default_factory=dict, description="Top feature importance drivers of churn")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CLVSummary(BaseModel):
    """Customer Lifetime Value (CLV) Estimation metrics."""

    average_historical_clv: float = Field(..., description="Average historical total spend per customer")
    average_projected_12m_clv: float = Field(..., description="Average projected 12-month future CLV")
    top_clv_segment: str = Field(..., description="Customer segment with highest projected CLV")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BIAnalysisReport(BaseModel):
    """Comprehensive Business Intelligence and Customer Analytics Report."""

    run_id: str = Field(..., description="Unique BI analysis run ID")
    dataset_id: Optional[int] = Field(None, description="Registered dataset ID if applicable")
    resolved_mapping: ColumnMappingConfig = Field(..., description="Resolved column mapping configuration")
    business_kpis: BusinessKPIs = Field(..., description="Business executive KPIs")
    rfm_segments: List[RFMSegmentSummary] = Field(default_factory=list, description="RFM persona segments")
    cluster_evaluation: Optional[ClusterEvaluationMetrics] = Field(None, description="Clustering evaluation metrics")
    customer_segments: List[CustomerSegmentDetail] = Field(default_factory=list, description="Customer clusters")
    churn_summary: Optional[ChurnRiskSummary] = Field(None, description="Churn prediction summary if applicable")
    clv_summary: Optional[CLVSummary] = Field(None, description="CLV estimation summary if applicable")
    business_insights: List[str] = Field(default_factory=list, description="Actionable business insights")
    charts_plotly_json: Dict[str, str] = Field(default_factory=dict, description="Plotly chart JSON strings dictionary")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Report timestamp"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BIAnalysisRequest(BaseModel):
    """Payload to trigger BI & Customer Analytics run."""

    dataset_id: Optional[int] = Field(None, description="Registered dataset ID")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Direct records list if dataset_id is not provided")
    column_mapping: Optional[ColumnMappingConfig] = Field(None, description="Manual column mapping overrides")
    clustering_algorithm: str = Field(default="kmeans", description="Clustering algorithm: kmeans or hierarchical")


class BIAnalysisResponse(BaseModel):
    """Response payload returned from BI analysis endpoint."""

    run_id: str = Field(..., description="Unique BI run ID")
    report: BIAnalysisReport = Field(..., description="BI Analysis Report")
