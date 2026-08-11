"""Machine Learning Engine - ModelTrainer Service for automated classification and regression."""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False


from datasense.data_processing.preprocessor import DataPreprocessor
from datasense.ml_models.schemas import (
    TaskType,
    ClassificationAlgorithm,
    RegressionAlgorithm,
    ClassificationMetrics,
    RegressionMetrics,
    ModelEvaluationResult,
    TrainingConfig,
    ModelComparisonReport,
    PredictionResponse,
)
from datasense.ml_models.registry import BaseModelRegistry, LocalModelRegistry
from datasense.utilities.logger import get_logger

logger = get_logger("ml_models.trainer")


def determine_task_type(df: pd.DataFrame, target_col: str) -> TaskType:
    """Automatically determine whether a target column is suitable for classification or regression.
    
    Rule heuristics:
    1. String, object, boolean, or categorical dtypes -> Classification.
    2. Integer/float dtypes with <= 10 unique values -> Classification.
    3. Integer/float dtypes with <= 20 unique values AND unique_ratio < 5% -> Classification.
    4. Otherwise -> Regression.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    series = df[target_col].dropna()
    if series.empty:
        raise ValueError(f"Target column '{target_col}' contains only missing values.")

    dtype_str = str(series.dtype)
    n_unique = series.nunique()
    total_len = len(series)
    unique_ratio = n_unique / total_len if total_len > 0 else 1.0

    if pd.api.types.is_bool_dtype(series) or dtype_str in ["object", "string", "category"]:
        return TaskType.CLASSIFICATION

    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 10:
            return TaskType.CLASSIFICATION
        elif n_unique <= 20 and unique_ratio < 0.05:
            return TaskType.CLASSIFICATION
        else:
            return TaskType.REGRESSION

    return TaskType.CLASSIFICATION


class ModelTrainer:
    """Orchestrator for dataset splitting, data leakage prevention preprocessing,
    classification and regression model training, cross-validation, hyperparameter tuning,
    evaluation reporting, and model registry persistence.
    """

    def __init__(self, registry: Optional[BaseModelRegistry] = None):
        self.registry = registry or LocalModelRegistry()

    def train(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        dataset_id: Optional[int] = None,
    ) -> ModelComparisonReport:
        """Run complete end-to-end ML training pipeline."""
        target_col = config.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not present in provided DataFrame.")

        # Drop rows where target is missing
        df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
        if df_clean.empty:
            raise ValueError(f"Dataset has no valid rows after dropping missing values in target '{target_col}'.")

        # 1. Determine or validate task type
        resolved_task = config.task_type
        if resolved_task == TaskType.AUTO or resolved_task is None:
            resolved_task = determine_task_type(df_clean, target_col)
            logger.info(f"Auto-detected ML task type: {resolved_task.value} for target '{target_col}'")

        # 2. Extract X and y
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        # Target Encoding for Classification if non-numeric
        target_encoder: Optional[LabelEncoder] = None
        if resolved_task == TaskType.CLASSIFICATION:
            if not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                # Fit target encoder ONLY on full y for clean numeric representation,
                # but split target values consistently.
                y = pd.Series(target_encoder.fit_transform(y), name=target_col)

        # 3. Train / Validation / Test Split (DATA LEAKAGE PREVENTION)
        test_size = config.test_size
        val_size = config.val_size
        random_state = config.random_state

        # Stratification strategy for classification
        stratify_y = None
        if resolved_task == TaskType.CLASSIFICATION:
            val_counts = y.value_counts()
            if (val_counts >= 2).all() and len(val_counts) > 1:
                stratify_y = y

        # Initial Train+Val vs Test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
        )

        # Calculate relative validation ratio
        val_relative_ratio = val_size / (1.0 - test_size)
        stratify_train_val = None
        if resolved_task == TaskType.CLASSIFICATION:
            tv_counts = y_train_val.value_counts()
            if (tv_counts >= 2).all() and len(tv_counts) > 1:
                stratify_train_val = y_train_val

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_relative_ratio,
            random_state=random_state,
            stratify=stratify_train_val,
        )

        logger.info(
            f"Dataset splits - Total: {len(df_clean)}, Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # 4. Preprocessing Integration (FITTED STRICTLY ON X_TRAIN TO PREVENT LEAKAGE)
        preprocessor = DataPreprocessor(config=config.preprocessing_config)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test)

        feature_names = list(X_train_proc.columns)

        # Convert to numpy arrays for sklearn/xgboost estimators
        X_train_arr = X_train_proc.to_numpy()
        X_val_arr = X_val_proc.to_numpy()
        X_test_arr = X_test_proc.to_numpy()
        y_train_arr = y_train.to_numpy()
        y_val_arr = y_val.to_numpy()
        y_test_arr = y_test.to_numpy()

        # 5. Resolve Algorithms to Train
        selected_model_keys = config.selected_models or []
        estimators = self._get_estimators(resolved_task, selected_model_keys, random_state)

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        results: List[ModelEvaluationResult] = []
        artifacts_to_save = []

        # 6. Train and Evaluate each Model
        for model_key, (model_name, estimator, param_grid) in estimators.items():
            logger.info(f"Training model: '{model_name}' ({model_key}) for task '{resolved_task.value}'...")
            start_time = time.time()
            best_params = None

            # Hyperparameter Tuning
            if config.enable_tuning and param_grid:
                logger.info(f"Executing RandomizedSearchCV for '{model_name}'...")
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions=param_grid,
                    n_iter=min(config.n_tuning_trials, 15),
                    cv=config.cross_validation_folds,
                    random_state=random_state,
                    n_jobs=-1,
                )
                search.fit(X_train_arr, y_train_arr)
                fitted_model = search.best_estimator_
                best_params = search.best_params_
            else:
                fitted_model = estimator
                fitted_model.fit(X_train_arr, y_train_arr)

            training_time = round(time.time() - start_time, 4)

            # Cross-Validation Score on Training split
            scoring_metric = "f1_weighted" if resolved_task == TaskType.CLASSIFICATION else "neg_root_mean_squared_error"
            try:
                cv_scores = cross_val_score(
                    fitted_model,
                    X_train_arr,
                    y_train_arr,
                    cv=config.cross_validation_folds,
                    scoring=scoring_metric,
                    n_jobs=-1,
                )
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
            except Exception as cv_err:
                logger.warning(f"Cross validation error for {model_name}: {cv_err}")
                cv_mean, cv_std = None, None

            # Evaluate on Validation set and Test set
            val_metrics = self._evaluate(fitted_model, X_val_arr, y_val_arr, resolved_task)
            test_metrics = self._evaluate(fitted_model, X_test_arr, y_test_arr, resolved_task)

            model_id = f"model_{model_key}_{uuid.uuid4().hex[:8]}"

            eval_result = ModelEvaluationResult(
                model_id=model_id,
                run_id=run_id,
                model_name=model_name,
                model_type=model_key,
                task_type=resolved_task,
                training_time_seconds=training_time,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                cv_scores_mean=cv_mean,
                cv_scores_std=cv_std,
                best_params=best_params,
                is_best=False,
            )

            # Artifact Package
            artifact_package = {
                "estimator": fitted_model,
                "preprocessor": preprocessor,
                "target_encoder": target_encoder,
                "feature_names": feature_names,
                "task_type": resolved_task.value,
                "target_column": target_col,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "metadata": {
                    "model_id": model_id,
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "model_name": model_name,
                    "model_type": model_key,
                    "task_type": resolved_task.value,
                    "target_column": target_col,
                    "feature_columns": feature_names,
                    "hyperparameters": best_params or getattr(fitted_model, "get_params", lambda: {})(),
                    "metrics": {"validation_metrics": val_metrics, "test_metrics": test_metrics},
                    "training_time_seconds": training_time,
                },
            }

            results.append(eval_result)
            artifacts_to_save.append((model_id, artifact_package))

        # 7. Identify Best Model
        selection_metric = "f1" if resolved_task == TaskType.CLASSIFICATION else "rmse"
        best_idx = 0
        best_score = -float("inf") if resolved_task == TaskType.CLASSIFICATION else float("inf")

        for idx, res in enumerate(results):
            val_m = res.validation_metrics
            if resolved_task == TaskType.CLASSIFICATION:
                score = val_m.get("f1", val_m.get("accuracy", 0.0))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            else:
                score = val_m.get("rmse", float("inf"))
                if score < best_score:
                    best_score = score
                    best_idx = idx

        results[best_idx].is_best = True
        best_model_id = results[best_idx].model_id
        best_model_name = results[best_idx].model_name

        # 8. Save all models in ModelRegistry
        for model_id, artifact_package in artifacts_to_save:
            is_best = (model_id == best_model_id)
            artifact_package["metadata"]["is_best"] = is_best
            self.registry.save_model(
                model_id=model_id,
                artifact=artifact_package,
                metadata=artifact_package["metadata"],
            )

        # 9. Build Report & Save Experiment
        report = ModelComparisonReport(
            run_id=run_id,
            dataset_id=dataset_id,
            target_column=target_col,
            task_type=resolved_task,
            total_samples=len(df_clean),
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            feature_names=feature_names,
            results=results,
            best_model_id=best_model_id,
            best_model_name=best_model_name,
            selection_metric=selection_metric,
        )

        self.registry.save_experiment_report(run_id, report.model_dump())
        logger.info(f"Experiment run '{run_id}' completed. Best model: '{best_model_name}' ({best_model_id})")

        return report

    def predict(self, model_id: str, df_features: pd.DataFrame) -> PredictionResponse:
        """Make real-time predictions using a stored model from the registry."""
        artifact = self.registry.load_model(model_id)
        
        estimator = artifact["estimator"]
        preprocessor: DataPreprocessor = artifact["preprocessor"]
        target_encoder: Optional[LabelEncoder] = artifact.get("target_encoder")
        task_type_str = artifact.get("task_type", "classification")
        model_name = artifact.get("metadata", {}).get("model_name", "Trained Model")
        target_col = artifact.get("target_column", "target")

        # Drop target column if user passed full dataframe containing target
        if target_col in df_features.columns:
            df_input = df_features.drop(columns=[target_col])
        else:
            df_input = df_features.copy()

        # Transform inputs using preprocessor fitted on training set
        df_proc = preprocessor.transform(df_input)
        X_arr = df_proc.to_numpy()

        raw_preds = estimator.predict(X_arr)

        probabilities = None
        if task_type_str == TaskType.CLASSIFICATION.value:
            if hasattr(estimator, "predict_proba"):
                try:
                    probas_arr = estimator.predict_proba(X_arr)
                    classes = getattr(estimator, "classes_", list(range(probas_arr.shape[1])))
                    probabilities = []
                    for row_prob in probas_arr:
                        prob_dict = {}
                        for c_idx, class_val in enumerate(classes):
                            class_label = str(class_val)
                            if target_encoder is not None:
                                try:
                                    class_label = str(target_encoder.inverse_transform([class_val])[0])
                                except Exception:
                                    pass
                            prob_dict[class_label] = float(row_prob[c_idx])
                        probabilities.append(prob_dict)
                except Exception as prob_err:
                    logger.warning(f"Failed to calculate class probabilities: {prob_err}")

            if target_encoder is not None:
                try:
                    predictions_list = target_encoder.inverse_transform(raw_preds).tolist()
                except Exception:
                    predictions_list = raw_preds.tolist()
            else:
                predictions_list = raw_preds.tolist()
        else:
            predictions_list = [float(val) for val in raw_preds]

        return PredictionResponse(
            model_id=model_id,
            model_name=model_name,
            task_type=TaskType(task_type_str),
            predictions=predictions_list,
            probabilities=probabilities,
            row_count=len(predictions_list),
        )

    def _get_estimators(
        self,
        task_type: TaskType,
        selected_keys: List[str],
        random_state: int,
    ) -> Dict[str, Tuple[str, Any, Optional[Dict[str, List[Any]]]]]:
        """Construct dictionary of requested model estimators and hyperparameter search grids."""
        dict_estimators = {}

        if task_type == TaskType.CLASSIFICATION:
            all_clf = {
                ClassificationAlgorithm.LOGISTIC_REGRESSION.value: (
                    "Logistic Regression",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                    {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]},
                ),
                ClassificationAlgorithm.RANDOM_FOREST.value: (
                    "Random Forest Classifier",
                    RandomForestClassifier(n_estimators=100, random_state=random_state),
                    {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
                ),
                ClassificationAlgorithm.GRADIENT_BOOSTING.value: (
                    "Gradient Boosting Classifier",
                    GradientBoostingClassifier(random_state=random_state),
                    {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 8]},
                ),
            }
            if HAS_XGBOOST:
                all_clf[ClassificationAlgorithm.XGBOOST.value] = (
                    "XGBoost Classifier",
                    xgb.XGBClassifier(random_state=random_state, eval_metric="logloss"),
                    {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 6, 10]},
                )

            for key, val in all_clf.items():
                if not selected_keys or key in selected_keys:
                    dict_estimators[key] = val

        else:
            all_reg = {
                RegressionAlgorithm.LINEAR_REGRESSION.value: (
                    "Linear Regression",
                    LinearRegression(),
                    {"fit_intercept": [True, False]},
                ),
                RegressionAlgorithm.RANDOM_FOREST.value: (
                    "Random Forest Regressor",
                    RandomForestRegressor(n_estimators=100, random_state=random_state),
                    {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
                ),
                RegressionAlgorithm.GRADIENT_BOOSTING.value: (
                    "Gradient Boosting Regressor",
                    GradientBoostingRegressor(random_state=random_state),
                    {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 8]},
                ),
            }
            if HAS_XGBOOST:
                all_reg[RegressionAlgorithm.XGBOOST.value] = (
                    "XGBoost Regressor",
                    xgb.XGBRegressor(random_state=random_state),
                    {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 6, 10]},
                )

            for key, val in all_reg.items():

                if not selected_keys or key in selected_keys:
                    dict_estimators[key] = val

        if not dict_estimators:
            raise ValueError(f"No valid models selected for task type '{task_type.value}'.")

        return dict_estimators

    def _evaluate(
        self,
        model: Any,
        X_arr: np.ndarray,
        y_true: np.ndarray,
        task_type: TaskType,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics for classification or regression."""
        y_pred = model.predict(X_arr)

        if task_type == TaskType.CLASSIFICATION:
            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

            cm = confusion_matrix(y_true, y_pred).tolist()

            roc_auc = None
            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(X_arr)
                    n_classes = probas.shape[1]
                    if n_classes == 2:
                        roc_auc = float(roc_auc_score(y_true, probas[:, 1]))
                    else:
                        roc_auc = float(roc_auc_score(y_true, probas, multi_class="ovr"))
                except Exception:
                    roc_auc = None

            return ClassificationMetrics(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                roc_auc=roc_auc,
                confusion_matrix=cm,
            ).model_dump()

        else:
            mae = float(mean_absolute_error(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_true, y_pred))

            try:
                mape = float(mean_absolute_percentage_error(y_true, y_pred))
            except Exception:
                safe_denom = np.maximum(np.abs(y_true), 1e-8)
                mape = float(np.mean(np.abs((y_true - y_pred) / safe_denom)) * 100.0)

            return RegressionMetrics(
                mae=mae,
                mse=mse,
                rmse=rmse,
                r2=r2,
                mape=mape,
            ).model_dump()
