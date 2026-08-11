"""Unit and API integration tests for Exploratory Data Analysis (EDA) engine."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from datasense.api.main import create_app
from datasense.database.connection import init_db_tables, SessionLocal
from datasense.database.models import DatasetMetadata, EDARecord
from datasense.eda.engine import EDAEngine
from datasense.eda.schemas import EDAReport, InsightItem


@pytest.fixture
def sample_eda_df():
    """Generates a rich test dataset for EDA testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D").astype(str)

    val1 = np.random.normal(100, 15, size=n)
    val1[0] = 999.0  # outlier
    val1[10:15] = np.nan  # missing

    # Highly correlated feature
    val2 = val1 * 3.5 + np.random.normal(0, 1, size=n)

    # Categorical with dominant class
    cat1 = ["Standard"] * 80 + ["Premium"] * 20

    # Target column
    target = (val1 > 105).astype(int)

    return pd.DataFrame({
        "timestamp": dates,
        "amount": val1,
        "amount_scaled": val2,
        "tier": cat1,
        "target": target,
    })


def test_eda_engine_summary_and_stats(sample_eda_df):
    engine = EDAEngine(sample_eda_df, target_column="target")
    report: EDAReport = engine.generate_report()

    assert report.summary.row_count == 100
    assert report.summary.column_count == 5
    assert "amount" in report.numerical_stats
    assert "tier" in report.categorical_stats

    num_stat = report.numerical_stats["amount"]
    assert num_stat.missing_count == 5
    assert num_stat.mean > 0
    assert num_stat.skewness != 0


def test_eda_correlations_and_outliers(sample_eda_df):
    engine = EDAEngine(sample_eda_df)
    report: EDAReport = engine.generate_report()

    assert len(report.top_correlation_pairs) >= 1
    top_pair = report.top_correlation_pairs[0]
    assert top_pair["abs_pearson"] > 0.80

    assert "amount" in report.outlier_analysis
    outlier_stat = report.outlier_analysis["amount"]
    assert outlier_stat.iqr_outliers >= 1


def test_eda_time_trends_and_target_analysis(sample_eda_df):
    engine = EDAEngine(sample_eda_df, target_column="target")
    report: EDAReport = engine.generate_report()

    assert "trends" in report.time_trends
    assert report.target_analysis["target_column"] == "target"


def test_eda_automatic_insights_detection(sample_eda_df):
    engine = EDAEngine(sample_eda_df, target_column="target")
    report: EDAReport = engine.generate_report()

    assert len(report.insights) > 0
    insight_categories = [i.category for i in report.insights]
    assert any(c in insight_categories for c in ["CORRELATION", "CATEGORICAL", "OUTLIER"])


def test_eda_plotly_charts_serialization(sample_eda_df):
    engine = EDAEngine(sample_eda_df, target_column="target")
    report: EDAReport = engine.generate_report()

    assert len(report.charts_plotly_json) >= 2
    assert "missing_bar" in report.charts_plotly_json or "corr_heatmap" in report.charts_plotly_json


# API Integration Tests
app = create_app()


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db_tables()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seed_eda_dataset():
    db = SessionLocal()
    preview_data = {
        "columns": ["x", "y", "cat"],
        "rows": [
            {"x": 10.0, "y": 20.0, "cat": "A"},
            {"x": 15.0, "y": 30.0, "cat": "B"},
            {"x": 20.0, "y": 40.0, "cat": "A"},
        ],
    }

    record = DatasetMetadata(
        name="Test EDA Dataset",
        filename="test_eda.csv",
        file_type="csv",
        file_size_bytes=512,
        row_count=3,
        column_count=3,
        quality_score=100.0,
        preview_data=preview_data,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    dataset_id = record.id
    db.close()
    return dataset_id


def test_eda_api_analyze_and_retrieve_report(client, seed_eda_dataset):
    response = client.post(f"/api/v1/eda/datasets/{seed_eda_dataset}/analyze", json={"target_column": "cat"})
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["dataset_id"] == seed_eda_dataset
    assert data["target_column"] == "cat"
    assert "report" in data
    assert data["report"]["summary"]["row_count"] == 3

    # Test GET latest report
    resp_get = client.get(f"/api/v1/eda/datasets/{seed_eda_dataset}/report")
    assert resp_get.status_code == 200
    assert resp_get.json()["id"] == data["id"]
