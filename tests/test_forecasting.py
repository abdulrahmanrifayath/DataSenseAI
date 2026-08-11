"""Unit tests for Time-Series Forecasting Engine."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from datasense.forecasting.engine import ForecastingEngine, calculate_smape, calculate_mape
from datasense.forecasting.schemas import ForecastingConfig, ForecastingReport, ForecastModelType


@pytest.fixture
def sample_timeseries_df():
    """Generates daily time series DataFrame with trend and seasonality."""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    t = np.arange(100)
    values = 50.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 7.0) + np.random.normal(0, 1.0, 100)
    return pd.DataFrame({"timestamp": dates, "sales": values})


def test_calculate_smape_and_mape():
    """Test SMAPE and MAPE metric calculations."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)

    assert 0.0 <= mape <= 20.0
    assert 0.0 <= smape <= 20.0


def test_validate_and_preprocess_timeseries(sample_timeseries_df):
    """Test frequency inference and chronological validation."""
    engine = ForecastingEngine()
    df_clean, freq, missing = engine.validate_and_preprocess_timeseries(
        sample_timeseries_df, date_col="timestamp", target_col="sales", freq_option="auto"
    )

    assert len(df_clean) >= 100
    assert freq in ["D", "1D"]
    assert missing == 0


def test_forecasting_pipeline(sample_timeseries_df):
    """Test end-to-end forecasting pipeline across selected models."""
    engine = ForecastingEngine()
    config = ForecastingConfig(
        date_column="timestamp",
        target_column="sales",
        forecast_horizon=14,
        frequency="D",
        selected_models=["baseline", "xgboost"],
        test_size_ratio=0.2,
    )

    report: ForecastingReport = engine.run_forecasting(sample_timeseries_df, config)

    assert report.total_records == 100
    assert report.forecast_horizon == 14
    assert len(report.results) >= 2
    assert report.best_model_name is not None
    assert report.chart_plotly_json is not None

    for res in report.results:
        assert res.test_metrics.rmse >= 0.0
        assert len(res.future_forecast) == 14
        first_item = res.future_forecast[0]
        assert first_item.timestamp is not None
        assert first_item.lower_bound <= first_item.predicted_value <= first_item.upper_bound
