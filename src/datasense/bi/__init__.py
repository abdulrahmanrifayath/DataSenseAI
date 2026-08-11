"""Business Intelligence and Customer Analytics module."""

from datasense.utilities.logger import get_logger
from datasense.bi.schemas import (
    ColumnMappingConfig,
    BusinessKPIs,
    RFMSegmentSummary,
    ClusterEvaluationMetrics,
    CustomerSegmentDetail,
    ChurnRiskSummary,
    CLVSummary,
    BIAnalysisReport,
    BIAnalysisRequest,
    BIAnalysisResponse,
)
from datasense.bi.engine import BIEngine

logger = get_logger("bi")

__all__ = [
    "BIEngine",
    "ColumnMappingConfig",
    "BusinessKPIs",
    "RFMSegmentSummary",
    "ClusterEvaluationMetrics",
    "CustomerSegmentDetail",
    "ChurnRiskSummary",
    "CLVSummary",
    "BIAnalysisReport",
    "BIAnalysisRequest",
    "BIAnalysisResponse",
]
