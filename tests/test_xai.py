"""Unit tests for Explainable AI (SHAP) Service."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from datasense.xai.service import XAIExplanationService
from datasense.xai.schemas import XAIReport, LocalExplanation, GlobalFeatureImportance


@pytest.fixture
def sample_ml_df():
    """Generates synthetic dataset for XAI testing."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(50, 10, n)
    x3 = np.random.uniform(0, 1, n)
    y = (x1 * 2.0 + x2 * 0.5 + np.random.normal(0, 1, n) > 40).astype(int)

    df = pd.DataFrame({"feat_1": x1, "feat_2": x2, "feat_3": x3})
    return df, y


def test_xai_explanation_classification(sample_ml_df):
    """Test SHAP explanation service for classification model."""
    df, y = sample_ml_df
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(df, y)

    service = XAIExplanationService()
    report: XAIReport = service.explain(
        model=model,
        X_df=df,
        feature_names=list(df.columns),
        task_type="classification",
        instance_indices=[0, 5],
    )

    assert report.task_type == "classification"
    assert len(report.global_importance) == 3
    assert report.global_importance[0].rank == 1
    assert report.global_importance[0].mean_abs_shap_value >= 0.0

    assert len(report.sample_local_explanations) == 2
    local_exp: LocalExplanation = report.sample_local_explanations[0]
    assert local_exp.instance_index == 0
    assert len(local_exp.all_contributions) == 3
    assert report.summary_chart_plotly_json is not None


def test_xai_explanation_regression(sample_ml_df):
    """Test SHAP explanation service for regression model."""
    df, _ = sample_ml_df
    y_reg = df["feat_1"] * 3.0 + df["feat_2"] * 1.5 + np.random.normal(0, 0.5, len(df))

    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(df, y_reg)

    service = XAIExplanationService()
    report: XAIReport = service.explain(
        model=model,
        X_df=df,
        feature_names=list(df.columns),
        task_type="regression",
        instance_indices=[1],
    )

    assert report.task_type == "regression"
    assert len(report.global_importance) == 3
    assert len(report.sample_local_explanations) == 1
