"""Anomaly Detection Engine supporting Isolation Forest, Z-Score, IQR, and Ensemble methods."""

from enum import Enum
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

from datasense.anomaly_detection.schemas import (
    AnomalyMethod,
    SeverityLevel,
    AnomalyRecordDetail,
    AnomalyConfig,
    AnomalyReport,
)
from datasense.utilities.logger import get_logger

logger = get_logger("anomaly_detection.detector")


class AnomalyDetector:
    """Multi-method Anomaly Detection Engine.
    
    Supports Isolation Forest, Statistical Z-Score, IQR-based detection, and Ensemble voting,
    returning anomaly flags, normalized confidence scores, severity levels, affected rows,
    top contributing features, and Plotly visualization charts.
    """

    def __init__(self, contamination: Optional[float] = None):
        self.contamination = contamination
        logger.info(f"Initialized AnomalyDetector (contamination fallback: {contamination})")


    def detect(
        self,
        df: Optional[pd.DataFrame],
        config: Optional[AnomalyConfig] = None,
        dataset_id: Optional[int] = None,
    ) -> Any:
        """Executes anomaly detection pipeline on DataFrame."""
        if df is None:
            logger.info("Detecting anomalies stub for None input.")
            return {"status": "anomalies_detected", "contamination": self.contamination or 0.05}

        if df.empty:
            raise ValueError("Input DataFrame for anomaly detection is empty.")


        config = config or AnomalyConfig()
        run_id = f"anom_{uuid.uuid4().hex[:10]}"

        # 1. Resolve Numerical Features
        num_cols = config.features
        if not num_cols:
            num_cols = [
                col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col]) and not col.lower().endswith("_id") and col.lower() != "id"
            ]

        if not num_cols:
            raise ValueError("No valid numerical features found in dataset for anomaly detection.")

        # Impute missing values with median for feature matrix
        df_num = df[num_cols].copy()
        for c in num_cols:
            if df_num[c].isnull().any():
                df_num[c] = df_num[c].fillna(df_num[c].median())

        X = df_num.to_numpy()
        n_rows, n_feats = X.shape

        # 2. Compute Mean & Std per feature for Z-score feature attribution
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1e-8
        Z_matrix = np.abs((X - means) / stds)

        # 3. Execute Method
        method = config.method
        if method == AnomalyMethod.ISOLATION_FOREST:
            flags, scores = self._detect_isolation_forest(X, config.contamination)
        elif method == AnomalyMethod.ZSCORE:
            flags, scores = self._detect_zscore(Z_matrix, config.z_threshold)
        elif method == AnomalyMethod.IQR:
            flags, scores = self._detect_iqr(X, config.iqr_multiplier)
        else:
            flags, scores = self._detect_ensemble(X, Z_matrix, config)

        # 4. Construct Anomaly Records & Severities
        anomalous_records: List[AnomalyRecordDetail] = []
        affected_indices: List[int] = []
        max_severity = SeverityLevel.LOW
        severity_order = {SeverityLevel.LOW: 1, SeverityLevel.MEDIUM: 2, SeverityLevel.HIGH: 3, SeverityLevel.CRITICAL: 4}

        feat_contrib_sums = {col: 0.0 for col in num_cols}

        for i in range(n_rows):
            if flags[i]:
                affected_indices.append(i)
                score = float(np.clip(scores[i], 0.0, 1.0))

                # Assign Severity Level
                if score >= 0.85:
                    sev = SeverityLevel.CRITICAL
                elif score >= 0.70:
                    sev = SeverityLevel.HIGH
                elif score >= 0.50:
                    sev = SeverityLevel.MEDIUM
                else:
                    sev = SeverityLevel.LOW

                if severity_order[sev] > severity_order[max_severity]:
                    max_severity = sev

                # Determine top 3 contributing features based on Z-score deviation
                row_z = Z_matrix[i]
                top_feat_indices = np.argsort(row_z)[::-1][:min(3, n_feats)]
                contrib_dict = {}
                for idx in top_feat_indices:
                    f_name = num_cols[idx]
                    f_z = float(round(row_z[idx], 3))
                    contrib_dict[f_name] = f_z
                    feat_contrib_sums[f_name] += f_z

                # Raw feature values for context
                raw_values = {col: (float(df.iloc[i][col]) if isinstance(df.iloc[i][col], (int, float, np.integer, np.floating)) else str(df.iloc[i][col])) for col in num_cols[:5]}

                record = AnomalyRecordDetail(
                    row_index=i,
                    anomaly_score=round(score, 4),
                    severity=sev,
                    contributing_features=contrib_dict,
                    feature_values=raw_values,
                )
                anomalous_records.append(record)

        # Overall Feature Importance Ranking
        total_anom = max(1, len(affected_indices))
        feature_importance = {col: round(sum_z / total_anom, 4) for col, sum_z in feat_contrib_sums.items()}
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

        # 5. Build Plotly Chart
        fig = self._build_plotly_anomaly_chart(df_num, num_cols, flags, scores)
        chart_json = fig.to_json() if fig else None

        affected_count = len(affected_indices)
        pct = round((affected_count / n_rows) * 100.0, 2)

        report = AnomalyReport(
            run_id=run_id,
            dataset_id=dataset_id,
            method=method.value if isinstance(method, Enum) else str(method),
            total_rows=n_rows,
            affected_rows_count=affected_count,
            anomaly_percentage=pct,
            max_severity=max_severity if affected_count > 0 else SeverityLevel.LOW,
            affected_row_indices=affected_indices,
            anomalous_records=anomalous_records,
            feature_importance_ranking=sorted_importance,
            chart_plotly_json=chart_json,
        )

        logger.info(
            f"Anomaly Detection run_id '{run_id}' ({method.value}) finished. "
            f"Flagged {affected_count}/{n_rows} rows ({pct}%) as anomalous. Max Severity: '{report.max_severity.value}'"
        )
        return report

    def _detect_isolation_forest(self, X: np.ndarray, contamination: float) -> Tuple[np.ndarray, np.ndarray]:
        """Isolation Forest Detection."""
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(X)
        flags = (preds == -1)

        dec_func = iso.decision_function(X)
        # Normalize decision function into score range [0, 1]
        scores = 1.0 - ((dec_func - dec_func.min()) / (dec_func.max() - dec_func.min() + 1e-8))
        return flags, scores

    def _detect_zscore(self, Z_matrix: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Statistical Z-Score Detection."""
        max_z = np.max(Z_matrix, axis=1)
        flags = (max_z > threshold)
        scores = np.clip(max_z / (threshold * 2.0), 0.0, 1.0)
        return flags, scores

    def _detect_iqr(self, X: np.ndarray, multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
        """IQR Outlier Detection."""
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1e-8

        lower_bound = q25 - (multiplier * iqr)
        upper_bound = q75 + (multiplier * iqr)

        over_upper = np.maximum(0, X - upper_bound)
        under_lower = np.maximum(0, lower_bound - X)
        dev_matrix = (over_upper + under_lower) / iqr

        max_dev = np.max(dev_matrix, axis=1)
        flags = (max_dev > 0)
        scores = np.clip(max_dev / (multiplier * 3.0), 0.0, 1.0)
        return flags, scores

    def _detect_ensemble(self, X: np.ndarray, Z_matrix: np.ndarray, config: AnomalyConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble Voting across Isolation Forest, Z-Score, and IQR."""
        f_iso, s_iso = self._detect_isolation_forest(X, config.contamination)
        f_z, s_z = self._detect_zscore(Z_matrix, config.z_threshold)
        f_iqr, s_iqr = self._detect_iqr(X, config.iqr_multiplier)

        # Average normalized scores
        avg_scores = (s_iso + s_z + s_iqr) / 3.0
        # Flag if at least 2 methods vote True OR average score is top contamination quantile
        vote_counts = f_iso.astype(int) + f_z.astype(int) + f_iqr.astype(int)
        score_thresh = float(np.percentile(avg_scores, 100.0 * (1.0 - config.contamination)))

        flags = (vote_counts >= 2) | (avg_scores >= score_thresh)
        return flags, avg_scores

    def _build_plotly_anomaly_chart(
        self,
        df_num: pd.DataFrame,
        num_cols: List[str],
        flags: np.ndarray,
        scores: np.ndarray,
    ) -> go.Figure:
        """Build Plotly 2D Scatter or PCA projection chart for visual anomaly inspection."""
        X = df_num.to_numpy()
        n_rows, n_feats = X.shape

        if n_feats >= 2:
            if n_feats == 2:
                x_vals = X[:, 0]
                y_vals = X[:, 1]
                x_title, y_title = num_cols[0], num_cols[1]
            else:
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X)
                x_vals, y_vals = X_pca[:, 0], X_pca[:, 1]
                x_title, y_title = "PCA Component 1", "PCA Component 2"
        else:
            x_vals = np.arange(n_rows)
            y_vals = X[:, 0]
            x_title, y_title = "Row Index", num_cols[0]

        plot_df = pd.DataFrame({
            "x": x_vals,
            "y": y_vals,
            "Anomaly": np.where(flags, "Anomalous", "Normal"),
            "Score": np.round(scores, 4),
            "Row": np.arange(n_rows),
        })

        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="Anomaly",
            size="Score",
            hover_data=["Row", "Score"],
            color_discrete_map={"Normal": "#0284C7", "Anomalous": "#EF4444"},
            title="Anomaly Detection Visual Scatter Plot",
            labels={"x": x_title, "y": y_title},
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig
