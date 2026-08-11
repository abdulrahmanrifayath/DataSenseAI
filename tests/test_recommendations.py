"""Unit tests for Business Recommendation Engine."""

import pytest
import pandas as pd
from datasense.recommendations.engine import BusinessRecommendationEngine
from datasense.recommendations.schemas import RecommendationReport, RecommendationPriority


def test_recommendations_synthesis():
    """Test recommendation engine synthesis across EDA, ML, Forecast, Anomaly, BI, and SHAP findings."""
    engine = BusinessRecommendationEngine()

    df = pd.DataFrame({"col_a": [1, 2, None, None, 5], "col_b": [10, 20, 30, 40, 50]})
    eda_rep = {"outliers_detected": 12}
    ml_rep = {"best_model_name": "XGBoost", "task_type": "classification", "metrics": {"accuracy": 0.62}}
    fc_rep = {
        "best_model_name": "Prophet",
        "results": [{
            "is_best": True,
            "future_forecast": [
                {"predicted_value": 100.0},
                {"predicted_value": 70.0},
            ]
        }]
    }
    anom_rep = {"affected_rows_count": 15, "anomaly_percentage": 7.5, "max_severity": "Critical"}
    bi_rep = {
        "business_kpis": {"total_revenue": 50000.0, "total_customers": 100, "repeat_purchase_rate": 12.0},
        "churn_summary": {"high_risk_customer_count": 8, "overall_churn_rate_pct": 24.5},
    }
    xai_rep = {
        "global_importance": [
            {"feature_name": "price", "mean_abs_shap_value": 4.5},
            {"feature_name": "recency", "mean_abs_shap_value": 2.1},
        ]
    }

    report: RecommendationReport = engine.generate_recommendations(
        df=df,
        eda_report=eda_rep,
        ml_report=ml_rep,
        forecast_report=fc_rep,
        anomaly_report=anom_rep,
        bi_report=bi_rep,
        xai_report=xai_rep,
    )

    assert report.total_recommendations >= 5
    assert report.critical_count >= 1
    assert report.high_count >= 1

    # Check structure of recommendation items
    for item in report.items:
        assert item.title is not None
        assert item.explanation is not None
        assert item.evidence is not None
        assert item.severity_priority in list(RecommendationPriority)
        assert item.affected_metric is not None
        assert item.suggested_action is not None
        assert item.source_module is not None
