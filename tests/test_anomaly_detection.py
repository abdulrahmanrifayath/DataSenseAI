"""Unit tests for Anomaly Detection Engine."""

import pytest
import numpy as np
import pandas as pd

from datasense.anomaly_detection.detector import AnomalyDetector
from datasense.anomaly_detection.schemas import AnomalyConfig, AnomalyMethod, SeverityLevel, AnomalyReport


@pytest.fixture
def sample_anomaly_df():
    """Generates synthetic dataset with known outliers."""
    np.random.seed(42)
    normal = np.random.normal(loc=10.0, scale=1.0, size=(100, 3))
    outliers = np.array([
        [100.0, 105.0, 95.0],
        [-50.0, -40.0, -60.0],
        [200.0, 210.0, 190.0],
        [10.0, 10.0, 500.0],
        [12.0, -300.0, 11.0],
    ])
    X = np.vstack([normal, outliers])
    return pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])


def test_isolation_forest_detection(sample_anomaly_df):
    """Test Isolation Forest anomaly detection method."""
    detector = AnomalyDetector()
    config = AnomalyConfig(method=AnomalyMethod.ISOLATION_FOREST, contamination=0.05)

    report: AnomalyReport = detector.detect(sample_anomaly_df, config)

    assert report.total_rows == 105
    assert report.affected_rows_count >= 3
    assert report.anomaly_percentage > 0.0
    assert len(report.anomalous_records) == report.affected_rows_count

    for rec in report.anomalous_records:
        assert 0.0 <= rec.anomaly_score <= 1.0
        assert rec.severity in list(SeverityLevel)


def test_zscore_detection(sample_anomaly_df):
    """Test Statistical Z-Score anomaly detection method."""
    detector = AnomalyDetector()
    config = AnomalyConfig(method=AnomalyMethod.ZSCORE, z_threshold=3.0)

    report: AnomalyReport = detector.detect(sample_anomaly_df, config)

    assert report.affected_rows_count >= 3
    assert 100 in report.affected_row_indices or 101 in report.affected_row_indices or 102 in report.affected_row_indices


def test_iqr_detection(sample_anomaly_df):
    """Test Interquartile Range (IQR) anomaly detection method."""
    detector = AnomalyDetector()
    config = AnomalyConfig(method=AnomalyMethod.IQR, iqr_multiplier=1.5)

    report: AnomalyReport = detector.detect(sample_anomaly_df, config)

    assert report.affected_rows_count >= 3


def test_ensemble_detection(sample_anomaly_df):
    """Test Ensemble Voting anomaly detection method."""
    detector = AnomalyDetector()
    config = AnomalyConfig(method=AnomalyMethod.ENSEMBLE, contamination=0.05)

    report: AnomalyReport = detector.detect(sample_anomaly_df, config)

    assert report.affected_rows_count >= 3
    assert report.chart_plotly_json is not None
    assert "feat_a" in report.feature_importance_ranking or "feat_b" in report.feature_importance_ranking
