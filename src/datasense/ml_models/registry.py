"""Model Registry abstraction and implementations for persistent storage and future MLflow integration."""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import joblib

from datasense.database.connection import get_db_context
from datasense.database.models import MLModelRecord, MLExperimentRecord
from datasense.utilities.logger import get_logger

logger = get_logger("ml_models.registry")


class BaseModelRegistry(ABC):
    """Abstract Base Class for Machine Learning Model Registries.
    
    Provides standardized interface for model persistence, metadata logging,
    retrieval, and experiment tracking so MLflow or other tracking systems
    can be integrated seamlessly.
    """

    @abstractmethod
    def save_model(
        self,
        model_id: str,
        artifact: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """Save model estimator artifact package and log metadata."""
        pass

    @abstractmethod
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load model estimator artifact package by model ID."""
        pass

    @abstractmethod
    def list_models(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered model metadata records."""
        pass

    @abstractmethod
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata by model ID."""
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Remove model artifact and database record."""
        pass

    @abstractmethod
    def save_experiment_report(self, run_id: str, report: Dict[str, Any]) -> None:
        """Persist experiment run comparison report."""
        pass

    @abstractmethod
    def get_experiment_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment run comparison report by run_id."""
        pass


class LocalModelRegistry(BaseModelRegistry):
    """Local file-system and database implementation of BaseModelRegistry."""

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path("data/models")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory backup cache for environment fallback
        self._memory_index: Dict[str, Dict[str, Any]] = {}
        self._memory_experiments: Dict[str, Dict[str, Any]] = {}

    def save_model(
        self,
        model_id: str,
        artifact: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """Save model artifact package safely to disk and persist record in DB/index."""
        file_path = self.storage_dir / f"{model_id}.joblib"
        
        # Save complete artifact package using joblib safely
        joblib.dump(artifact, file_path)
        logger.info(f"Persisted model artifact package to: {file_path}")

        meta_payload = {
            "model_id": model_id,
            "run_id": metadata.get("run_id", "default_run"),
            "dataset_id": metadata.get("dataset_id"),
            "model_name": metadata.get("model_name", "Unknown Model"),
            "model_type": metadata.get("model_type", "unknown"),
            "task_type": metadata.get("task_type", "classification"),
            "target_column": metadata.get("target_column", ""),
            "feature_columns": metadata.get("feature_columns", []),
            "hyperparameters": metadata.get("hyperparameters", {}),
            "metrics": metadata.get("metrics", {}),
            "training_time_seconds": metadata.get("training_time_seconds", 0.0),
            "model_path": str(file_path),
            "is_best": metadata.get("is_best", False),
            "created_at": metadata.get("created_at"),
        }

        self._memory_index[model_id] = meta_payload

        # Attempt Database Persistence
        try:
            with get_db_context() as db:
                record = db.query(MLModelRecord).filter(MLModelRecord.model_id == model_id).first()
                if not record:
                    record = MLModelRecord(
                        model_id=model_id,
                        run_id=meta_payload["run_id"],
                        dataset_id=meta_payload["dataset_id"],
                        model_name=meta_payload["model_name"],
                        model_type=meta_payload["model_type"],
                        task_type=meta_payload["task_type"],
                        target_column=meta_payload["target_column"],
                        feature_columns=meta_payload["feature_columns"],
                        hyperparameters=meta_payload["hyperparameters"],
                        metrics=meta_payload["metrics"],
                        training_time_seconds=meta_payload["training_time_seconds"],
                        model_path=str(file_path),
                        is_best=meta_payload["is_best"],
                    )
                    db.add(record)
                    db.commit()
                    logger.info(f"Registered model record '{model_id}' in database.")
        except Exception as e:
            logger.warning(f"Could not persist model record to DB (fallback to local cache): {e}")

        return str(file_path)

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load model estimator artifact package by model ID."""
        file_path = self.storage_dir / f"{model_id}.joblib"
        
        # If not found directly, check metadata for custom path
        if not file_path.exists():
            meta = self.get_model_metadata(model_id)
            if meta and "model_path" in meta and Path(meta["model_path"]).exists():
                file_path = Path(meta["model_path"])

        if not file_path.exists():
            raise FileNotFoundError(f"Model artifact for model_id '{model_id}' not found at {file_path}")

        artifact = joblib.load(file_path)
        return artifact

    def list_models(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by task type."""
        models = []

        try:
            with get_db_context() as db:
                query = db.query(MLModelRecord)
                if task_type:
                    query = query.filter(MLModelRecord.task_type == task_type)
                records = query.order_by(MLModelRecord.created_at.desc()).all()
                
                for r in records:
                    models.append({
                        "model_id": r.model_id,
                        "run_id": r.run_id,
                        "dataset_id": r.dataset_id,
                        "model_name": r.model_name,
                        "model_type": r.model_type,
                        "task_type": r.task_type,
                        "target_column": r.target_column,
                        "feature_columns": r.feature_columns,
                        "hyperparameters": r.hyperparameters,
                        "metrics": r.metrics,
                        "training_time_seconds": r.training_time_seconds,
                        "model_path": r.model_path,
                        "is_best": r.is_best,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    })
        except Exception as e:
            logger.warning(f"DB list models failed, using memory fallback: {e}")
            for m_id, meta in self._memory_index.items():
                if task_type is None or meta.get("task_type") == task_type:
                    models.append(meta)

        # Merge memory cache entries if missing from DB list
        db_ids = {m["model_id"] for m in models}
        for m_id, meta in self._memory_index.items():
            if m_id not in db_ids:
                if task_type is None or meta.get("task_type") == task_type:
                    models.append(meta)

        return models

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata dictionary by model_id."""
        if model_id in self._memory_index:
            return self._memory_index[model_id]

        try:
            with get_db_context() as db:
                record = db.query(MLModelRecord).filter(MLModelRecord.model_id == model_id).first()
                if record:
                    meta = {
                        "model_id": record.model_id,
                        "run_id": record.run_id,
                        "dataset_id": record.dataset_id,
                        "model_name": record.model_name,
                        "model_type": record.model_type,
                        "task_type": record.task_type,
                        "target_column": record.target_column,
                        "feature_columns": record.feature_columns,
                        "hyperparameters": record.hyperparameters,
                        "metrics": record.metrics,
                        "training_time_seconds": record.training_time_seconds,
                        "model_path": record.model_path,
                        "is_best": record.is_best,
                        "created_at": record.created_at.isoformat() if record.created_at else None,
                    }
                    self._memory_index[model_id] = meta
                    return meta
        except Exception as e:
            logger.warning(f"DB get_model_metadata failed: {e}")

        return None

    def delete_model(self, model_id: str) -> bool:
        """Delete model artifact file and registry record."""
        file_path = self.storage_dir / f"{model_id}.joblib"
        deleted = False

        if file_path.exists():
            file_path.unlink()
            deleted = True

        self._memory_index.pop(model_id, None)

        try:
            with get_db_context() as db:
                record = db.query(MLModelRecord).filter(MLModelRecord.model_id == model_id).first()
                if record:
                    db.delete(record)
                    db.commit()
                    deleted = True
        except Exception as e:
            logger.warning(f"DB delete_model failed: {e}")

        return deleted

    def save_experiment_report(self, run_id: str, report: Dict[str, Any]) -> None:
        """Persist experiment run report."""
        self._memory_experiments[run_id] = report

        try:
            with get_db_context() as db:
                record = db.query(MLExperimentRecord).filter(MLExperimentRecord.run_id == run_id).first()
                if not record:
                    record = MLExperimentRecord(
                        run_id=run_id,
                        dataset_id=report.get("dataset_id"),
                        target_column=report.get("target_column", ""),
                        task_type=report.get("task_type", ""),
                        best_model_id=report.get("best_model_id"),
                        best_model_name=report.get("best_model_name"),
                        report=report,
                    )
                    db.add(record)
                    db.commit()
        except Exception as e:
            logger.warning(f"DB save_experiment_report failed: {e}")

    def get_experiment_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment run report by run_id."""
        if run_id in self._memory_experiments:
            return self._memory_experiments[run_id]

        try:
            with get_db_context() as db:
                record = db.query(MLExperimentRecord).filter(MLExperimentRecord.run_id == run_id).first()
                if record:
                    return record.report
        except Exception as e:
            logger.warning(f"DB get_experiment_report failed: {e}")

        return None


class MLflowModelRegistry(BaseModelRegistry):
    """Abstraction wrapper for future MLflow Model Registry integration."""

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._local_fallback = LocalModelRegistry()
        logger.info(f"Initialized MLflowModelRegistry with URI: {self.tracking_uri}")

    def save_model(
        self,
        model_id: str,
        artifact: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """Save model artifact using local fallback and log MLflow run parameters/metrics if available."""
        local_path = self._local_fallback.save_model(model_id, artifact, metadata)
        
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            with mlflow.start_run(run_name=metadata.get("model_name", model_id)):
                if "hyperparameters" in metadata and metadata["hyperparameters"]:
                    mlflow.log_params(metadata["hyperparameters"])
                if "metrics" in metadata and metadata["metrics"]:
                    val_m = metadata["metrics"].get("validation_metrics", {})
                    test_m = metadata["metrics"].get("test_metrics", {})
                    for k, v in val_m.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"val_{k}", v)
                    for k, v in test_m.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"test_{k}", v)
        except Exception as e:
            logger.debug(f"MLflow logging bypassed (MLflow server not active or optional): {e}")

        return local_path

    def load_model(self, model_id: str) -> Dict[str, Any]:
        return self._local_fallback.load_model(model_id)

    def list_models(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._local_fallback.list_models(task_type)

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._local_fallback.get_model_metadata(model_id)

    def delete_model(self, model_id: str) -> bool:
        return self._local_fallback.delete_model(model_id)

    def save_experiment_report(self, run_id: str, report: Dict[str, Any]) -> None:
        self._local_fallback.save_experiment_report(run_id, report)

    def get_experiment_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._local_fallback.get_experiment_report(run_id)
