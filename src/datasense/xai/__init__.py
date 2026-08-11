"""Explainable AI (XAI) module with SHAP support."""

from datasense.utilities.logger import get_logger
from datasense.xai.schemas import (
    FeatureContribution,
    LocalExplanation,
    GlobalFeatureImportance,
    XAIReport,
    XAIRequest,
    XAIResponse,
)
from datasense.xai.service import XAIExplanationService

logger = get_logger("xai")

__all__ = [
    "XAIExplanationService",
    "FeatureContribution",
    "LocalExplanation",
    "GlobalFeatureImportance",
    "XAIReport",
    "XAIRequest",
    "XAIResponse",
]
