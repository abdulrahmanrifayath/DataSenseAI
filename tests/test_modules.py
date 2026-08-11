"""Unit tests for modular data processing and analytics stubs."""

import pandas as pd
from datasense.data_processing import DataValidator, DataPreprocessor
from datasense.eda import EDAEngine
from datasense.ml_models import BaseModelPipeline
from datasense.forecasting import ForecastingEngine
from datasense.anomaly_detection import AnomalyDetector
from datasense.segmentation import CustomerSegmenter
from datasense.explainability import ModelExplainer
from datasense.recommendations import BusinessRecommendationEngine


def test_data_validator_and_preprocessor():
    df = pd.DataFrame({"col_a": [1.0, 2.0, None], "col_b": ["x", "y", "z"]})
    validator = DataValidator(df)
    report = validator.validate()
    assert report.row_count == 3
    assert report.column_count == 2

    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.fit_transform(df)
    assert cleaned_df["col_a"].isnull().sum() == 0


def test_analytics_modules_initialization():
    df = pd.DataFrame({"val": [10, 20, 30]})
    eda = EDAEngine(df)
    stats = eda.get_summary_statistics()
    assert "describe" in stats

    ml = BaseModelPipeline(model_type="xgboost")
    assert ml.train(None, None)["status"] == "trained"

    forecast = ForecastingEngine(horizon=14)
    assert forecast.generate_forecast(None)["horizon"] == 14

    anomaly = AnomalyDetector()
    assert anomaly.detect(None)["status"] == "anomalies_detected"

    segmenter = CustomerSegmenter(n_clusters=3)
    assert segmenter.fit_predict(None)["n_clusters"] == 3

    explainer = ModelExplainer()
    assert explainer.explain(None)["method"] == "SHAP"

    recommender = BusinessRecommendationEngine()
    recs = recommender.generate_recommendations({})
    assert len(recs) > 0
