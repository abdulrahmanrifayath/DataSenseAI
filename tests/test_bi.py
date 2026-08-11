"""Unit tests for Business Intelligence & Customer Analytics Engine."""

import pytest
import numpy as np
import pandas as pd

from datasense.bi.engine import BIEngine
from datasense.bi.schemas import ColumnMappingConfig, BIAnalysisReport


@pytest.fixture
def sample_ecom_df():
    """Generates synthetic e-commerce transaction dataset with custom column names."""
    np.random.seed(42)
    n_rows = 150
    cust_ids = [f"CUST_{i:03d}" for i in np.random.randint(1, 30, size=n_rows)]
    dates = pd.date_range(start="2025-01-01", periods=n_rows, freq="12h").strftime("%Y-%m-%d %H:%M:%S").tolist()

    sales = np.random.uniform(20.0, 500.0, size=n_rows)
    profits = sales * np.random.uniform(0.1, 0.4, size=n_rows)

    return pd.DataFrame({
        "client_identifier": cust_ids,
        "transaction_date": dates,
        "sales_revenue": sales,
        "net_margin": profits,
    })


def test_alias_resolution(sample_ecom_df):
    """Test automatic fuzzy alias matching for configurable column names."""
    engine = BIEngine()
    mapping = engine.resolve_column_mapping(sample_ecom_df)

    assert mapping.customer_id_col == "client_identifier"
    assert mapping.order_date_col == "transaction_date"
    assert mapping.revenue_col == "sales_revenue"
    assert mapping.profit_col == "net_margin"


def test_full_bi_analysis(sample_ecom_df):
    """Test end-to-end BI analysis pipeline."""
    engine = BIEngine()
    report: BIAnalysisReport = engine.analyze(sample_ecom_df)

    assert report.business_kpis.total_revenue > 0.0
    assert report.business_kpis.total_profit > 0.0
    assert report.business_kpis.total_customers > 0
    assert report.business_kpis.average_order_value > 0.0

    # RFM Persona Segments
    assert len(report.rfm_segments) > 0

    # Cluster Evaluation
    assert report.cluster_evaluation is not None
    assert report.cluster_evaluation.optimal_k >= 2
    assert -1.0 <= report.cluster_evaluation.silhouette_score <= 1.0
    assert report.cluster_evaluation.davies_bouldin_index >= 0.0

    # Churn & CLV
    assert report.churn_summary is not None
    assert report.clv_summary is not None

    # Visualizations
    assert "customer_segments_scatter" in report.charts_plotly_json
    assert "revenue_by_segment_pie" in report.charts_plotly_json


def test_graceful_degradation_non_customer_dataset():
    """Test graceful handling of datasets missing customer and datetime columns."""
    df_generic = pd.DataFrame({
        "product": ["A", "B", "C", "D"],
        "sales": [100.0, 200.0, 150.0, 300.0],
    })

    engine = BIEngine()
    report: BIAnalysisReport = engine.analyze(df_generic)

    assert report.business_kpis.total_revenue == 750.0
    assert report.business_kpis.average_order_value == 187.5
    assert len(report.business_insights) > 0
