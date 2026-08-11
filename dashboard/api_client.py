"""DataSense AI API Client for communicating between Streamlit frontend and FastAPI backend."""

import requests
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from datasense.utilities.logger import get_logger

logger = get_logger("dashboard.api_client")


class DataSenseAPIClient:
    """HTTP Client connecting Streamlit UI to FastAPI REST endpoints with graceful offline fallback."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Optional[Dict[str, Any]]:
        """Pings API health endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/api/v1/health", timeout=3)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"FastAPI backend health check failed: {e}")
        return None

    def upload_dataset(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """POST /api/v1/datasets/upload"""
        try:
            files = {"file": (filename, file_bytes, "application/octet-stream")}
            resp = requests.post(f"{self.base_url}/api/v1/datasets/upload", files=files, timeout=10)
            if resp.status_code == 201:
                return resp.json()
        except Exception as e:
            logger.warning(f"Upload dataset API error: {e}")
        return None

    def preprocess_dataset(self, dataset_id: int, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """POST /api/v1/preprocessing/clean"""
        try:
            payload = {"dataset_id": dataset_id, "config": config or {}}
            resp = requests.post(f"{self.base_url}/api/v1/preprocessing/clean", json=payload, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Preprocess dataset API error: {e}")
        return None

    def run_eda(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """POST /api/v1/eda/analyze"""
        try:
            resp = requests.post(f"{self.base_url}/api/v1/eda/analyze", json={"dataset_id": dataset_id}, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"EDA API error: {e}")
        return None

    def train_ml_models(self, dataset_id: int, target_column: str, task_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """POST /api/v1/ml/train"""
        try:
            payload = {"dataset_id": dataset_id, "target_column": target_column}
            if task_type:
                payload["task_type"] = task_type
            resp = requests.post(f"{self.base_url}/api/v1/ml/train", json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Train ML models API error: {e}")
        return None

    def predict_ml_model(self, model_id: str, input_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """POST /api/v1/ml/predict"""
        try:
            payload = {"model_id": model_id, "input_data": input_data}
            resp = requests.post(f"{self.base_url}/api/v1/ml/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Predict ML model API error: {e}")
        return None

    def run_forecasting(self, dataset_id: int, date_column: str, target_column: str, horizon: int = 14) -> Optional[Dict[str, Any]]:
        """POST /api/v1/forecasting/predict"""
        try:
            payload = {
                "dataset_id": dataset_id,
                "date_column": date_column,
                "target_column": target_column,
                "forecast_horizon": horizon,
            }
            resp = requests.post(f"{self.base_url}/api/v1/forecasting/predict", json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Forecasting API error: {e}")
        return None

    def detect_anomalies(self, dataset_id: int, method: str = "ensemble") -> Optional[Dict[str, Any]]:
        """POST /api/v1/anomaly/detect"""
        try:
            payload = {"dataset_id": dataset_id, "method": method}
            resp = requests.post(f"{self.base_url}/api/v1/anomaly/detect", json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Anomaly detection API error: {e}")
        return None

    def run_bi(self, dataset_id: int, column_mapping: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """POST /api/v1/bi/analyze"""
        try:
            payload = {"dataset_id": dataset_id, "column_mapping": column_mapping or {}}
            resp = requests.post(f"{self.base_url}/api/v1/bi/analyze", json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"BI API error: {e}")
        return None

    def explain_model(self, dataset_id: int, target_column: str, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """POST /api/v1/xai/explain"""
        try:
            payload = {"dataset_id": dataset_id, "target_column": target_column, "model_id": model_id}
            resp = requests.post(f"{self.base_url}/api/v1/xai/explain", json=payload, timeout=25)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"XAI API error: {e}")
        return None

    def generate_recommendations(
        self,
        dataset_id: Optional[int] = None,
        eda_report: Optional[Dict[str, Any]] = None,
        ml_report: Optional[Dict[str, Any]] = None,
        forecast_report: Optional[Dict[str, Any]] = None,
        anomaly_report: Optional[Dict[str, Any]] = None,
        bi_report: Optional[Dict[str, Any]] = None,
        xai_report: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """POST /api/v1/recommendations/generate"""
        try:
            payload = {
                "dataset_id": dataset_id,
                "eda_report": eda_report,
                "ml_report": ml_report,
                "forecast_report": forecast_report,
                "anomaly_report": anomaly_report,
                "bi_report": bi_report,
                "xai_report": xai_report,
            }
            resp = requests.post(f"{self.base_url}/api/v1/recommendations/generate", json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Recommendations API error: {e}")
        return None
