"""Time-Series Forecasting Engine supporting baseline, Prophet, and XGBoost models."""

import time
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

from datasense.forecasting.schemas import (
    ForecastModelType,
    ForecastMetrics,
    ForecastItem,
    ModelForecastResult,
    ForecastingConfig,
    ForecastingReport,
)
from datasense.utilities.logger import get_logger

logger = get_logger("forecasting.engine")

# Check Prophet availability
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    Prophet = None
    HAS_PROPHET = False

# Check XGBoost availability
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE) in %."""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) in % with zero division handling."""
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


class ForecastingEngine:
    """Automated Time-Series Forecasting Pipeline Engine.
    
    Performs chronological sorting, frequency inference/resampling, temporal feature engineering,
    leakage-free train/test splitting, baseline/prophet/xgboost forecasting, metrics evaluation,
    confidence interval calculation, and Plotly chart generation.
    """

    def __init__(self, horizon: Optional[int] = None):
        self.horizon = horizon
        logger.info(f"Initialized ForecastingEngine (horizon fallback: {horizon})")

    def generate_forecast(self, data=None):
        """Legacy helper method returning status dict for compatibility."""
        return {"status": "forecast_generated", "horizon": self.horizon or 30}



    def validate_and_preprocess_timeseries(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        freq_option: str = "auto",
    ) -> Tuple[pd.DataFrame, str, int]:
        """Validates datetime ordering, infers frequency, and resamples/aggregates missing dates."""
        if date_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Columns '{date_col}' and '{target_col}' must be present in DataFrame.")

        df_clean = df.copy()
        
        # 1. Coerce Date & Target
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce")

        df_clean = df_clean.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

        if len(df_clean) < 10:
            raise ValueError("Time series requires at least 10 valid data points for forecasting.")

        # 2. Infer Frequency
        inferred_freq = None
        if freq_option != "auto":
            inferred_freq = freq_option
        else:
            inferred_freq = pd.infer_freq(df_clean[date_col])
            if not inferred_freq:
                timedeltas = df_clean[date_col].diff().dropna()
                median_seconds = timedeltas.dt.total_seconds().median()
                if median_seconds <= 3600:
                    inferred_freq = "h"
                elif median_seconds <= 86400 * 2:
                    inferred_freq = "D"
                elif median_seconds <= 86400 * 10:
                    inferred_freq = "W"
                else:
                    inferred_freq = "MS"

        # 3. Detect Missing Dates & Resample Grid
        missing_dates_count = 0
        try:
            full_date_range = pd.date_range(
                start=df_clean[date_col].min(),
                end=df_clean[date_col].max(),
                freq=inferred_freq,
            )
            missing_dates_count = max(0, len(full_date_range) - len(df_clean[date_col].unique()))

            if missing_dates_count > 0:
                logger.info(f"Detected {missing_dates_count} missing date steps. Resampling to frequency '{inferred_freq}'...")
                df_resampled = (
                    df_clean.set_index(date_col)
                    .resample(inferred_freq)
                    .agg({target_col: "mean"})
                    .interpolate(method="time")
                    .reset_index()
                )
                df_clean = df_resampled
        except Exception as resample_err:
            logger.warning(f"Frequency resampling notice: {resample_err}")

        logger.info(
            f"Preprocessed time series: {len(df_clean)} records from {df_clean[date_col].min()} to {df_clean[date_col].max()} (freq: '{inferred_freq}')"
        )
        return df_clean, str(inferred_freq), missing_dates_count

    def create_time_features(self, df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """Generates temporal sub-features, lags, and rolling window statistics without future leakage."""
        df_feat = df.copy()
        dt_s = df_feat[date_col].dt

        df_feat["year"] = dt_s.year
        df_feat["month"] = dt_s.month
        df_feat["day"] = dt_s.day
        df_feat["dayofweek"] = dt_s.dayofweek
        df_feat["quarter"] = dt_s.quarter
        df_feat["is_weekend"] = (dt_s.dayofweek >= 5).astype(int)

        # Lag features (strictly past observations)
        df_feat["lag_1"] = df_feat[target_col].shift(1)
        df_feat["lag_2"] = df_feat[target_col].shift(2)
        df_feat["lag_7"] = df_feat[target_col].shift(7)

        # Rolling window features (strictly past observations using shift(1))
        df_feat["rolling_mean_7"] = df_feat[target_col].shift(1).rolling(7, min_periods=1).mean()
        df_feat["rolling_std_7"] = df_feat[target_col].shift(1).rolling(7, min_periods=1).std().fillna(0)

        return df_feat

    def run_forecasting(
        self,
        df: pd.DataFrame,
        config: ForecastingConfig,
        dataset_id: Optional[int] = None,
    ) -> ForecastingReport:
        """Runs time-series forecasting across requested models and generates comparison report."""
        date_col = config.date_column
        target_col = config.target_column
        horizon = config.forecast_horizon

        # 1. Validate and preprocess
        df_clean, inferred_freq, missing_dates = self.validate_and_preprocess_timeseries(
            df, date_col, target_col, config.frequency
        )

        # 2. Chronological Split (No future data leakage!)
        test_ratio = config.test_size_ratio
        n_total = len(df_clean)
        n_test = max(3, min(horizon, int(n_total * test_ratio)))
        n_train = n_total - n_test

        train_df = df_clean.iloc[:n_train].reset_index(drop=True)
        test_df = df_clean.iloc[n_train:].reset_index(drop=True)

        logger.info(f"Chronological split - Total: {n_total}, Train: {n_train}, Test: {n_test}, Horizon: {horizon}")

        # 3. Selected Models
        selected_model_keys = config.selected_models or ["baseline", "xgboost"]
        run_id = f"fc_{uuid.uuid4().hex[:10]}"
        results: List[ModelForecastResult] = []

        # Future Date Range for horizon forecast steps
        last_date = df_clean[date_col].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=inferred_freq)[1:]

        # Execute Baseline Model
        if "baseline" in selected_model_keys:
            res_base = self._fit_predict_baseline(train_df, test_df, future_dates, date_col, target_col, config)
            results.append(res_base)

        # Execute Prophet Model
        if "prophet" in selected_model_keys:
            res_prophet = self._fit_predict_prophet(train_df, test_df, future_dates, date_col, target_col, config, inferred_freq)
            results.append(res_prophet)

        # Execute XGBoost Model
        if "xgboost" in selected_model_keys:
            res_xgb = self._fit_predict_xgboost(train_df, test_df, future_dates, date_col, target_col, config, inferred_freq)
            results.append(res_xgb)

        if not results:
            # Fallback to baseline if none selected
            results.append(self._fit_predict_baseline(train_df, test_df, future_dates, date_col, target_col, config))

        # 4. Identify Best Model (Lowest RMSE on holdout test set)
        best_idx = 0
        best_rmse = float("inf")
        for idx, res in enumerate(results):
            if res.test_metrics.rmse < best_rmse:
                best_rmse = res.test_metrics.rmse
                best_idx = idx

        results[best_idx].is_best = True
        best_model_name = results[best_idx].model_name

        # 5. Build Plotly Chart
        fig = self._build_plotly_forecast_chart(df_clean, date_col, target_col, test_df, results[best_idx], future_dates)
        chart_json = fig.to_json() if fig else None

        # 6. Report Construction
        report = ForecastingReport(
            run_id=run_id,
            dataset_id=dataset_id,
            date_column=date_col,
            target_column=target_col,
            inferred_frequency=inferred_freq,
            total_records=n_total,
            train_records=n_train,
            test_records=n_test,
            forecast_horizon=horizon,
            missing_dates_detected=missing_dates,
            results=results,
            best_model_name=best_model_name,
            chart_plotly_json=chart_json,
        )

        logger.info(f"Forecasting completed run_id '{run_id}'. Best Model: '{best_model_name}' (RMSE: {best_rmse:.4f})")
        return report

    def _fit_predict_baseline(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
        date_col: str,
        target_col: str,
        config: ForecastingConfig,
    ) -> ModelForecastResult:
        """Baseline Statistical Approach using Moving Average + Holt-style Trend & Exponential Smoothing."""
        y_train = train_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()

        # Fit Holt's exponential smoothing baseline
        n_tr = len(y_train)
        alpha, beta = 0.3, 0.1
        level = y_train[0]
        trend = (y_train[-1] - y_train[0]) / max(1, n_tr - 1)

        for val in y_train:
            prev_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        # Predict holdout test set
        test_preds = np.array([level + (h + 1) * trend for h in range(len(y_test))])

        # Predict future horizon
        future_preds = np.array([level + (len(y_test) + h + 1) * trend for h in range(len(future_dates))])

        # Residual std for 95% confidence intervals
        train_fitted = np.array([y_train[0]] + [y_train[i-1] for i in range(1, n_tr)])
        residual_std = float(np.std(y_train - train_fitted)) if n_tr > 1 else 1.0
        z_mult = 1.96

        future_items = []
        for h, (dt, val) in enumerate(zip(future_dates, future_preds)):
            margin = z_mult * residual_std * np.sqrt(1 + 0.05 * (h + 1))
            future_items.append(
                ForecastItem(
                    timestamp=dt.isoformat(),
                    predicted_value=float(round(val, 4)),
                    lower_bound=float(round(val - margin, 4)),
                    upper_bound=float(round(val + margin, 4)),
                )
            )

        metrics = ForecastMetrics(
            mae=float(mean_absolute_error(y_test, test_preds)),
            mse=float(mean_squared_error(y_test, test_preds)),
            rmse=float(np.sqrt(mean_squared_error(y_test, test_preds))),
            mape=calculate_mape(y_test, test_preds),
            smape=calculate_smape(y_test, test_preds),
        )

        return ModelForecastResult(
            model_type=ForecastModelType.BASELINE.value,
            model_name="Baseline Exponential Smoothing",
            test_metrics=metrics,
            future_forecast=future_items,
            is_best=False,
        )

    def _fit_predict_prophet(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
        date_col: str,
        target_col: str,
        config: ForecastingConfig,
        inferred_freq: str,
    ) -> ModelForecastResult:
        """Prophet Forecasting model with automatic trend + seasonality decomposition."""
        y_test = test_df[target_col].to_numpy()

        if HAS_PROPHET:
            try:
                df_p = pd.DataFrame({"ds": train_df[date_col], "y": train_df[target_col]})
                m = Prophet(interval_width=config.confidence_level, daily_seasonality=False, weekly_seasonality=True)
                m.fit(df_p)

                # Test set evaluation
                df_test_p = pd.DataFrame({"ds": test_df[date_col]})
                fc_test = m.predict(df_test_p)
                test_preds = fc_test["yhat"].to_numpy()

                # Future forecast
                df_future_p = pd.DataFrame({"ds": future_dates})
                fc_future = m.predict(df_future_p)

                future_items = []
                for idx, row in fc_future.iterrows():
                    future_items.append(
                        ForecastItem(
                            timestamp=pd.to_datetime(row["ds"]).isoformat(),
                            predicted_value=float(round(row["yhat"], 4)),
                            lower_bound=float(round(row["yhat_lower"], 4)),
                            upper_bound=float(round(row["yhat_upper"], 4)),
                        )
                    )

                metrics = ForecastMetrics(
                    mae=float(mean_absolute_error(y_test, test_preds)),
                    mse=float(mean_squared_error(y_test, test_preds)),
                    rmse=float(np.sqrt(mean_squared_error(y_test, test_preds))),
                    mape=calculate_mape(y_test, test_preds),
                    smape=calculate_smape(y_test, test_preds),
                )

                return ModelForecastResult(
                    model_type=ForecastModelType.PROPHET.value,
                    model_name="Prophet Time-Series",
                    test_metrics=metrics,
                    future_forecast=future_items,
                    is_best=False,
                )
            except Exception as prophet_err:
                logger.warning(f"Prophet execution error: {prophet_err}. Falling back to Harmonic Seasonal model.")

        # Fallback Harmonic Trend + Seasonality model when Prophet is unavailable
        return self._fit_predict_harmonic_fallback(train_df, test_df, future_dates, date_col, target_col, config)

    def _fit_predict_harmonic_fallback(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
        date_col: str,
        target_col: str,
        config: ForecastingConfig,
    ) -> ModelForecastResult:
        """Harmonic Seasonal + Polynomial Trend model fallback for Prophet."""
        y_train = train_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()

        n_tr = len(y_train)
        t_tr = np.arange(n_tr)
        sin_tr = np.sin(2 * np.pi * t_tr / 7.0)
        cos_tr = np.cos(2 * np.pi * t_tr / 7.0)

        X_tr = np.column_stack([np.ones(n_tr), t_tr, sin_tr, cos_tr])
        coefs, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)

        # Test prediction
        t_te = np.arange(n_tr, n_tr + len(y_test))
        X_te = np.column_stack([np.ones(len(y_test)), t_te, np.sin(2 * np.pi * t_te / 7.0), np.cos(2 * np.pi * t_te / 7.0)])
        test_preds = X_te @ coefs

        # Future prediction
        t_fut = np.arange(n_tr + len(y_test), n_tr + len(y_test) + len(future_dates))
        X_fut = np.column_stack([np.ones(len(future_dates)), t_fut, np.sin(2 * np.pi * t_fut / 7.0), np.cos(2 * np.pi * t_fut / 7.0)])
        future_preds = X_fut @ coefs

        residual_std = float(np.std(y_train - (X_tr @ coefs)))
        z_mult = 1.96

        future_items = []
        for h, (dt, val) in enumerate(zip(future_dates, future_preds)):
            margin = z_mult * residual_std * np.sqrt(1 + 0.08 * (h + 1))
            future_items.append(
                ForecastItem(
                    timestamp=dt.isoformat(),
                    predicted_value=float(round(val, 4)),
                    lower_bound=float(round(val - margin, 4)),
                    upper_bound=float(round(val + margin, 4)),
                )
            )

        metrics = ForecastMetrics(
            mae=float(mean_absolute_error(y_test, test_preds)),
            mse=float(mean_squared_error(y_test, test_preds)),
            rmse=float(np.sqrt(mean_squared_error(y_test, test_preds))),
            mape=calculate_mape(y_test, test_preds),
            smape=calculate_smape(y_test, test_preds),
        )

        return ModelForecastResult(
            model_type=ForecastModelType.PROPHET.value,
            model_name="Prophet/Seasonal Decomposition",
            test_metrics=metrics,
            future_forecast=future_items,
            is_best=False,
        )

    def _fit_predict_xgboost(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
        date_col: str,
        target_col: str,
        config: ForecastingConfig,
        inferred_freq: str,
    ) -> ModelForecastResult:
        """XGBoost Time-Series Forecasting using autoregressive lags and datetime features."""
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        feat_df = self.create_time_features(full_df, date_col, target_col)

        feature_cols = ["year", "month", "day", "dayofweek", "quarter", "is_weekend", "lag_1", "lag_2", "lag_7", "rolling_mean_7", "rolling_std_7"]

        train_feat = feat_df.iloc[:len(train_df)].dropna(subset=feature_cols)
        test_feat = feat_df.iloc[len(train_df):].fillna(0)

        X_train_arr = train_feat[feature_cols].to_numpy()
        y_train_arr = train_feat[target_col].to_numpy()
        X_test_arr = test_feat[feature_cols].to_numpy()
        y_test = test_df[target_col].to_numpy()

        if HAS_XGBOOST:
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

        model.fit(X_train_arr, y_train_arr)
        test_preds = model.predict(X_test_arr)

        # Recursive Multi-Step Future Forecasting
        future_preds = []
        recent_series = list(full_df[target_col].values[-30:])

        residual_std = float(np.std(y_train_arr - model.predict(X_train_arr))) if len(y_train_arr) > 0 else 1.0
        z_mult = 1.96

        future_items = []
        for h, dt in enumerate(future_dates):
            # Construct time features for future step dt
            dt_s = pd.Timestamp(dt)
            yr, mo, dy, dow, qtr, wknd = dt_s.year, dt_s.month, dt_s.day, dt_s.dayofweek, dt_s.quarter, int(dt_s.dayofweek >= 5)

            l1 = recent_series[-1] if len(recent_series) >= 1 else 0
            l2 = recent_series[-2] if len(recent_series) >= 2 else l1
            l7 = recent_series[-7] if len(recent_series) >= 7 else l1

            r_mean7 = float(np.mean(recent_series[-7:])) if len(recent_series) >= 1 else l1
            r_std7 = float(np.std(recent_series[-7:])) if len(recent_series) >= 2 else 0.0

            x_step = np.array([[yr, mo, dy, dow, qtr, wknd, l1, l2, l7, r_mean7, r_std7]])
            pred_val = float(model.predict(x_step)[0])

            recent_series.append(pred_val)
            margin = z_mult * residual_std * np.sqrt(1 + 0.1 * (h + 1))

            future_items.append(
                ForecastItem(
                    timestamp=dt_s.isoformat(),
                    predicted_value=float(round(pred_val, 4)),
                    lower_bound=float(round(pred_val - margin, 4)),
                    upper_bound=float(round(pred_val + margin, 4)),
                )
            )

        metrics = ForecastMetrics(
            mae=float(mean_absolute_error(y_test, test_preds)),
            mse=float(mean_squared_error(y_test, test_preds)),
            rmse=float(np.sqrt(mean_squared_error(y_test, test_preds))),
            mape=calculate_mape(y_test, test_preds),
            smape=calculate_smape(y_test, test_preds),
        )

        return ModelForecastResult(
            model_type=ForecastModelType.XGBOOST.value,
            model_name="XGBoost Autoregressive",
            test_metrics=metrics,
            future_forecast=future_items,
            is_best=False,
        )

    def _build_plotly_forecast_chart(
        self,
        full_df: pd.DataFrame,
        date_col: str,
        target_col: str,
        test_df: pd.DataFrame,
        best_result: ModelForecastResult,
        future_dates: pd.DatetimeIndex,
    ) -> go.Figure:
        """Build interactive Plotly chart showing historical data, holdout test predictions, and future forecasts."""
        fig = go.Figure()

        # 1. Historical Actuals Trace
        fig.add_trace(
            go.Scatter(
                x=full_df[date_col],
                y=full_df[target_col],
                mode="lines+markers",
                name="Historical Actuals",
                line=dict(color="#0284C7", width=2),
            )
        )

        # 2. Future Forecast Trace
        fut_dates = [pd.to_datetime(item.timestamp) for item in best_result.future_forecast]
        fut_vals = [item.predicted_value for item in best_result.future_forecast]
        fut_lower = [item.lower_bound for item in best_result.future_forecast]
        fut_upper = [item.upper_bound for item in best_result.future_forecast]

        # Upper bound line
        fig.add_trace(
            go.Scatter(
                x=fut_dates,
                y=fut_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound line with shaded confidence band
        fig.add_trace(
            go.Scatter(
                x=fut_dates,
                y=fut_lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.15)",
                line=dict(width=0),
                name="95% Confidence Band",
            )
        )

        # Point forecast line
        fig.add_trace(
            go.Scatter(
                x=fut_dates,
                y=fut_vals,
                mode="lines+markers",
                name=f"Future Forecast ({best_result.model_name})",
                line=dict(color="#10B981", width=3, dash="dash"),
            )
        )

        fig.update_layout(
            title=f"Time-Series Forecast for '{target_col}' ({best_result.model_name})",
            xaxis_title="Date",
            yaxis_title=target_col,
            hovermode="x unified",
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        return fig
