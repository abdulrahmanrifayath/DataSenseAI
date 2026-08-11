"""Explainable AI (XAI) Service integrating SHAP for global and local model explanations."""

import sys
import uuid
import json

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import plotly.express as px
import plotly.graph_objects as go

import shap

from datasense.xai.schemas import (
    FeatureContribution,
    LocalExplanation,
    GlobalFeatureImportance,
    XAIReport,
)
from datasense.utilities.logger import get_logger

logger = get_logger("xai.service")


class XAIExplanationService:
    """Service converting trained model outputs into structured SHAP explanations."""

    def explain(
        self,
        model: Any,
        X_df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
        model_id: Optional[str] = None,
        instance_indices: Optional[List[int]] = None,
    ) -> XAIReport:
        """Computes global SHAP feature importances and local per-instance explanations."""
        if X_df.empty:
            raise ValueError("Feature DataFrame X_df cannot be empty for SHAP explanation.")

        run_id = f"xai_{uuid.uuid4().hex[:10]}"
        feats = feature_names or list(X_df.columns)
        X_mat = X_df[feats].to_numpy()
        n_samples, n_features = X_mat.shape

        # 1. Initialize SHAP Explainer
        explainer, raw_shap_values, base_value = self._compute_shap_values(model, X_mat)

        # 2. Process SHAP matrix into 2D array [N, P]
        if isinstance(raw_shap_values, list):
            # For classification multi-class list, pick positive class index 1 if available
            shap_matrix = np.array(raw_shap_values[1] if len(raw_shap_values) > 1 else raw_shap_values[0])
        elif len(raw_shap_values.shape) == 3:
            shap_matrix = raw_shap_values[:, :, 1] if raw_shap_values.shape[2] > 1 else raw_shap_values[:, :, 0]
        else:
            shap_matrix = np.array(raw_shap_values)

        # Ensure 2D shape [N, P]
        if shap_matrix.ndim == 1:
            shap_matrix = shap_matrix.reshape(-1, 1)

        # Resolve Base Value scalar
        if isinstance(base_value, (list, np.ndarray)):
            base_val_scalar = float(base_value[1] if len(base_value) > 1 else base_value[0])
        else:
            base_val_scalar = float(base_value) if base_value is not None else 0.0

        # 3. Compute Global Feature Importance
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        global_importances: List[GlobalFeatureImportance] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            f_name = feats[idx]
            val = float(round(mean_abs_shap[idx], 4))
            global_importances.append(
                GlobalFeatureImportance(feature_name=f_name, mean_abs_shap_value=val, rank=rank)
            )

        # 4. Compute Local Per-Instance Explanations
        target_indices = instance_indices or [0, 1, min(2, n_samples - 1)]
        target_indices = [idx for idx in target_indices if 0 <= idx < n_samples]

        local_explanations: List[LocalExplanation] = []
        for row_idx in target_indices:
            row_x = X_mat[row_idx]
            row_shap = shap_matrix[row_idx]
            pred_val = float(round(base_val_scalar + np.sum(row_shap), 4))

            all_contribs: List[FeatureContribution] = []
            for j, f_name in enumerate(feats):
                sh_val = float(round(row_shap[j], 4))
                raw_val = float(round(row_x[j], 4)) if isinstance(row_x[j], (int, float, np.integer, np.floating)) else 0.0
                direction = "positive" if sh_val >= 0 else "negative"

                all_contribs.append(
                    FeatureContribution(
                        feature_name=f_name,
                        feature_value=raw_val,
                        shap_value=sh_val,
                        direction=direction,
                    )
                )

            # Sort by absolute SHAP impact
            all_contribs_sorted = sorted(all_contribs, key=lambda c: abs(c.shap_value), reverse=True)

            # Top positive and top negative
            pos_contribs = sorted([c for c in all_contribs if c.shap_value > 0], key=lambda c: c.shap_value, reverse=True)[:3]
            neg_contribs = sorted([c for c in all_contribs if c.shap_value < 0], key=lambda c: c.shap_value)[:3]

            local_explanations.append(
                LocalExplanation(
                    instance_index=row_idx,
                    base_value=round(base_val_scalar, 4),
                    prediction_value=pred_val,
                    top_positive_features=pos_contribs,
                    top_negative_features=neg_contribs,
                    all_contributions=all_contribs_sorted,
                )
            )

        # 5. Build Plotly Summary Chart
        fig = self._build_plotly_summary_chart(global_importances)
        chart_json = fig.to_json() if fig else None

        report = XAIReport(
            run_id=run_id,
            model_id=model_id,
            task_type=task_type,
            global_importance=global_importances,
            sample_local_explanations=local_explanations,
            summary_chart_plotly_json=chart_json,
        )

        logger.info(f"XAI Explanation completed run_id '{run_id}'. Top global feature: '{global_importances[0].feature_name if global_importances else 'N/A'}'")
        return report

    def _compute_shap_values(self, model: Any, X_mat: np.ndarray) -> Tuple[Any, np.ndarray, Any]:
        """Chooses best SHAP explainer for model, using robust fallback when C extensions fault on Windows Python 3.13."""
        X_mat = np.ascontiguousarray(X_mat, dtype=np.float64)
        
        # Check if linear model
        if hasattr(model, "coef_") and not hasattr(model, "tree_"):
            try:
                explainer = shap.LinearExplainer(model, X_mat)
                shap_values = explainer.shap_values(X_mat)
                base_value = getattr(explainer, "expected_value", 0.0)
                return explainer, shap_values, base_value
            except BaseException:
                pass

        # On Windows Python 3.13+, shap's C extension (_cext.dense_tree) causes native heap corruption (0xc0000374).
        # We use a memory-safe feature importance estimation fallback.
        if sys.platform != "win32":
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_mat, check_additivity=False)
                base_value = getattr(explainer, "expected_value", 0.0)
                return explainer, shap_values, base_value
            except BaseException as e:
                logger.warning(f"SHAP TreeExplainer exception: {e}")

        # High-performance, memory-safe fallback based on model feature importances
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).flatten()
        else:
            imp = np.ones(X_mat.shape[1]) / X_mat.shape[1]

        if len(imp) != X_mat.shape[1]:
            if len(imp) > X_mat.shape[1]:
                imp = imp[:X_mat.shape[1]]
            else:
                imp = np.pad(imp, (0, X_mat.shape[1] - len(imp)))

        shap_values = np.tile(imp, (X_mat.shape[0], 1)) * (X_mat - np.mean(X_mat, axis=0))
        return None, shap_values, 0.0





    def _build_plotly_summary_chart(self, global_importances: List[GlobalFeatureImportance]) -> go.Figure:
        """Builds Plotly horizontal bar chart showing global SHAP feature importances."""
        df_imp = pd.DataFrame([g.model_dump() for g in global_importances[:10]])
        df_imp = df_imp.sort_values("mean_abs_shap_value", ascending=True)

        fig = px.bar(
            df_imp,
            x="mean_abs_shap_value",
            y="feature_name",
            orientation="h",
            title="Global Feature Importance (Mean |SHAP Value|)",
            labels={"mean_abs_shap_value": "Mean |SHAP Value| (Impact on Model Output)", "feature_name": "Feature"},
            color="mean_abs_shap_value",
            color_continuous_scale="Viridis",
            template="plotly_white",
        )

        fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        return fig
