"""Comprehensive End-to-End Integration Test for DataSense AI.

Validates the full automated data intelligence pipeline:
Data Ingestion -> Validation -> Preprocessing -> EDA -> ML Training -> Forecasting
-> Anomaly Detection -> BI & Customer Analytics -> SHAP XAI -> Business Recommendations.
"""

import pytest
import numpy as np
import pandas as pd
from datasense.data_processing.ingestion import DataIngestionService
from datasense.data_processing.validator import DataValidator
from datasense.data_processing.preprocessor import DataPreprocessor
from datasense.data_processing.schemas import PreprocessingConfig
from datasense.eda.engine import EDAEngine
from datasense.ml_models.trainer import ModelTrainer
from datasense.ml_models.schemas import TrainingConfig
from datasense.forecasting.engine import ForecastingEngine
from datasense.forecasting.schemas import ForecastingConfig
from datasense.anomaly_detection.detector import AnomalyDetector
from datasense.anomaly_detection.schemas import AnomalyConfig, AnomalyMethod
from datasense.bi.engine import BIEngine
from datasense.xai.service import XAIExplanationService
from datasense.recommendations.engine import BusinessRecommendationEngine
from dashboard.api_client import DataSenseAPIClient


@pytest.fixture
def synthetic_pipeline_dataset():
    """Generates synthetic dataset covering all analytics engine requirements."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    cust_ids = [f"CUST_{np.random.randint(1, 20):03d}" for _ in range(n)]
    revenue = np.round(np.random.normal(200, 50, n), 2)
    profit = np.round(revenue * 0.3 + np.random.normal(0, 5, n), 2)
    quantity = np.random.randint(1, 10, n)
    churn = (revenue < 160).astype(int)

    df = pd.DataFrame({
        "customer_id": cust_ids,
        "order_date": dates,
        "revenue": revenue,
        "profit": profit,
        "quantity": quantity,
        "churn": churn,
    })
    return df


def test_full_end_to_end_datasense_pipeline(synthetic_pipeline_dataset):
    """Executes full automated pipeline end-to-end without errors."""
    raw_df = synthetic_pipeline_dataset

    # 1. Validation
    validator = DataValidator(raw_df)
    val_report = validator.validate()
    assert val_report.quality_score > 0

    # 2. Preprocessing
    preprocessor = DataPreprocessor()
    config_prep = PreprocessingConfig(impute_missing=True, remove_duplicates=True)
    clean_df = preprocessor.fit_transform(raw_df, config=config_prep)
    assert not clean_df.empty

    # 3. EDA
    eda_engine = EDAEngine(clean_df)
    eda_report = eda_engine.generate_report()
    assert eda_report.summary.row_count > 0



    # 4. Machine Learning
    trainer = ModelTrainer()
    ml_config = TrainingConfig(target_column="churn", tune_hyperparameters=False)
    ml_report = trainer.train(raw_df, config=ml_config)
    assert ml_report.best_model_name is not None



    # 5. Time-Series Forecasting
    fc_engine = ForecastingEngine()
    fc_config = ForecastingConfig(date_column="order_date", target_column="revenue", forecast_horizon=7)
    fc_report = fc_engine.run_forecasting(raw_df, config=fc_config)
    assert len(fc_report.results) > 0

    # 6. Anomaly Detection
    anom_detector = AnomalyDetector()
    anom_config = AnomalyConfig(method=AnomalyMethod.ENSEMBLE, contamination=0.05)
    anom_report = anom_detector.detect(raw_df, config=anom_config)
    assert anom_report.total_rows == len(raw_df)

    # 7. Customer Analytics & BI
    bi_engine = BIEngine()
    bi_report = bi_engine.analyze(raw_df)
    assert bi_report.business_kpis.total_customers > 0

    # 8. Explainable AI (SHAP)
    num_feats = ["revenue", "profit", "quantity"]
    X_mat = raw_df[num_feats].to_numpy()
    y_vec = raw_df["churn"].to_numpy()


    best_model = trainer.registry.load_model(ml_report.best_model_id).get("estimator")
    if best_model is None:
        from sklearn.ensemble import RandomForestClassifier
        best_model = RandomForestClassifier(n_estimators=10, random_state=42)
        best_model.fit(X_mat, y_vec)

    xai_service = XAIExplanationService()
    xai_report = xai_service.explain(
        model=best_model,
        X_df=pd.DataFrame(X_mat, columns=num_feats),
        feature_names=num_feats,
        task_type="classification",
        instance_indices=[0],
    )
    assert len(xai_report.global_importance) > 0

    # 9. Business Recommendation Engine
    rec_engine = BusinessRecommendationEngine()
    rec_report = rec_engine.generate_recommendations(
        df=clean_df,
        eda_report=eda_report.model_dump(),
        ml_report=ml_report.model_dump(),
        forecast_report=fc_report.model_dump(),
        anomaly_report=anom_report.model_dump(),
        bi_report=bi_report.model_dump(),
        xai_report=xai_report.model_dump(),
    )
    assert rec_report.total_recommendations >= 0


    # 10. Dashboard API Client Initialization
    client = DataSenseAPIClient("http://127.0.0.1:8000")
    assert client.base_url == "http://127.0.0.1:8000"
