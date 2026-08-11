"""Business Recommendation Engine synthesizing multi-module analytics into data-backed executive recommendations."""

import uuid
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd

from datasense.recommendations.schemas import (
    RecommendationPriority,
    RecommendationItem,
    RecommendationReport,
)
from datasense.utilities.logger import get_logger

logger = get_logger("recommendations.engine")


class BusinessRecommendationEngine:
    """Engine for generating data-grounded business recommendations."""

    def generate_recommendations(
        self,
        df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        eda_report: Optional[Dict[str, Any]] = None,
        ml_report: Optional[Dict[str, Any]] = None,
        forecast_report: Optional[Dict[str, Any]] = None,
        anomaly_report: Optional[Dict[str, Any]] = None,
        bi_report: Optional[Dict[str, Any]] = None,
        xai_report: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[int] = None,
    ) -> Any:
        """Synthesizes computed findings across all analytics modules into actionable recommendations."""
        if isinstance(df, dict):
            # Legacy stub compatibility
            return [
                {
                    "title": "Data Processing & Optimization",
                    "recommendation": "Optimize data pipeline and conduct Exploratory Data Analysis.",
                    "priority": "Medium",
                }
            ]

        run_id = f"rec_{uuid.uuid4().hex[:10]}"

        items: List[RecommendationItem] = []

        # 1. Analyze DataFrame directly if available
        if df is not None and not df.empty:
            self._analyze_dataframe(df, items)

        # 2. Analyze EDA Findings
        if eda_report:
            self._analyze_eda(eda_report, items)

        # 3. Analyze ML Findings
        if ml_report:
            self._analyze_ml(ml_report, items)

        # 4. Analyze Time-Series Forecast Findings
        if forecast_report:
            self._analyze_forecasting(forecast_report, items)

        # 5. Analyze Anomaly Detection Findings
        if anomaly_report:
            self._analyze_anomaly(anomaly_report, items)

        # 6. Analyze Business Intelligence Findings
        if bi_report:
            self._analyze_bi(bi_report, items)

        # 7. Analyze SHAP XAI Findings
        if xai_report:
            self._analyze_xai(xai_report, items)

        # Fallback if no specific module findings triggered recommendations
        if not items:
            items.append(
                RecommendationItem(
                    id=f"item_{uuid.uuid4().hex[:8]}",
                    title="Execute Full Exploratory Data Analysis & Modeling",
                    explanation="Dataset loaded successfully. Running EDA, predictive modeling, and forecasting will unlock automated data-driven recommendations.",
                    evidence=f"Dataset contains {len(df) if df is not None else 'N/A'} rows and {len(df.columns) if df is not None else 'N/A'} feature columns.",
                    severity_priority=RecommendationPriority.LOW,
                    affected_metric="Analytics Maturity",
                    suggested_action="Run EDA and Predictive Analytics modules to generate targeted business actions.",
                    source_module="System",
                )
            )

        # Sort recommendations by priority: CRITICAL > HIGH > MEDIUM > LOW
        priority_order = {
            RecommendationPriority.CRITICAL: 1,
            RecommendationPriority.HIGH: 2,
            RecommendationPriority.MEDIUM: 3,
            RecommendationPriority.LOW: 4,
        }
        items.sort(key=lambda item: priority_order[item.severity_priority])

        critical_count = sum(1 for item in items if item.severity_priority == RecommendationPriority.CRITICAL)
        high_count = sum(1 for item in items if item.severity_priority == RecommendationPriority.HIGH)

        report = RecommendationReport(
            run_id=run_id,
            dataset_id=dataset_id,
            total_recommendations=len(items),
            critical_count=critical_count,
            high_count=high_count,
            items=items,
        )

        logger.info(f"Recommendation Engine run_id '{run_id}' finished with {len(items)} items ({critical_count} Critical, {high_count} High).")
        return report

    def _analyze_dataframe(self, df: pd.DataFrame, items: List[RecommendationItem]):
        """Analyzes raw DataFrame properties for data quality recommendations."""
        n_rows, n_cols = df.shape
        missing_total = int(df.isnull().sum().sum())
        total_cells = n_rows * n_cols
        missing_pct = float(round((missing_total / max(1, total_cells)) * 100.0, 2))

        if missing_pct > 10.0:
            items.append(
                RecommendationItem(
                    id=f"item_{uuid.uuid4().hex[:8]}",
                    title="Audit and Impute High Missing Data Gaps",
                    explanation=f"Dataset has a high missing data rate of {missing_pct}%. Missing values can distort ML predictions and business metrics.",
                    evidence=f"{missing_total:,} out of {total_cells:,} data cells ({missing_pct}%) are unpopulated across {n_cols} columns.",
                    severity_priority=RecommendationPriority.HIGH if missing_pct > 25.0 else RecommendationPriority.MEDIUM,
                    affected_metric="Data Completeness & Quality",
                    suggested_action="Use the Data Preprocessing module to apply median/mode imputation or drop columns with >50% missing values.",
                    source_module="EDA",
                )
            )

    def _analyze_eda(self, eda_report: Dict[str, Any], items: List[RecommendationItem]):
        """Analyzes EDA findings for recommendations."""
        if "outliers_detected" in eda_report:
            out_cnt = eda_report.get("outliers_detected", 0)
            if out_cnt > 0:
                items.append(
                    RecommendationItem(
                        id=f"item_{uuid.uuid4().hex[:8]}",
                        title="Investigate Extreme Statistical Outliers",
                        explanation="EDA detected notable statistical outliers that may represent data entry errors or anomalous business operations.",
                        evidence=f"Detected {out_cnt:,} statistical outlier data points across evaluated features.",
                        severity_priority=RecommendationPriority.MEDIUM,
                        affected_metric="Data Integrity",
                        suggested_action="Review extreme values in EDA Scatter & Boxplots and use capping/winsorization if necessary.",
                        source_module="EDA",
                    )
                )

    def _analyze_ml(self, ml_report: Dict[str, Any], items: List[RecommendationItem]):
        """Analyzes ML training results for recommendations."""
        best_model = ml_report.get("best_model_name", "Model")
        task_type = ml_report.get("task_type", "classification")
        metrics = ml_report.get("metrics", {})

        acc = metrics.get("accuracy")
        if task_type == "classification" and acc is not None and acc < 0.70:
            items.append(
                RecommendationItem(
                    id=f"item_{uuid.uuid4().hex[:8]}",
                    title="Enhance Predictive Model Accuracy via Feature Engineering",
                    explanation=f"The top model '{best_model}' achieved {round(acc * 100, 1)}% accuracy, which indicates room for improvement in predictive power.",
                    evidence=f"Best Model: {best_model} | Accuracy: {round(acc * 100, 1)}% | Holdout Test Set.",
                    severity_priority=RecommendationPriority.HIGH,
                    affected_metric="Model Accuracy",
                    suggested_action="Perform hyperparameter tuning, introduce interaction terms, or gather additional predictive features.",
                    source_module="ML Engine",
                )
            )

    def _analyze_forecasting(self, forecast_report: Dict[str, Any], items: List[RecommendationItem]):
        """Analyzes forecasting results for inventory & trend recommendations."""
        best_model = forecast_report.get("best_model_name", "Forecasting Model")
        results = forecast_report.get("results", [])

        if results:
            best_res = next((r for r in results if r.get("is_best")), results[0])
            future_items = best_res.get("future_forecast", [])

            if len(future_items) >= 2:
                y_first = future_items[0].get("predicted_value", 0.0)
                y_last = future_items[-1].get("predicted_value", 0.0)
                diff_pct = float(round(((y_last - y_first) / max(1e-8, abs(y_first))) * 100.0, 2))

                if diff_pct < -5.0:
                    items.append(
                        RecommendationItem(
                            id=f"item_{uuid.uuid4().hex[:8]}",
                            title="Prepare Mitigation Strategy for Declining Target Forecast",
                            explanation=f"Time-series forecast predicts a downward decline of {abs(diff_pct)}% over the next {len(future_items)} periods.",
                            evidence=f"Projected change from {y_first:,.2f} to {y_last:,.2f} ({diff_pct}% change) using {best_model}.",
                            severity_priority=RecommendationPriority.CRITICAL if diff_pct < -20.0 else RecommendationPriority.HIGH,
                            affected_metric="Future Projected Growth / Demand",
                            suggested_action="Review inventory buffer stock, re-evaluate demand schedules, and initiate marketing promotions to offset decline.",
                            source_module="Forecasting",
                        )
                    )
                elif diff_pct > 15.0:
                    items.append(
                        RecommendationItem(
                            id=f"item_{uuid.uuid4().hex[:8]}",
                            title="Optimize Inventory & Supply Capacity for High Projected Growth",
                            explanation=f"Forecasting model projects an upward demand growth of +{diff_pct}% over the next {len(future_items)} periods.",
                            evidence=f"Projected increase from {y_first:,.2f} to {y_last:,.2f} (+{diff_pct}%) using {best_model}.",
                            severity_priority=RecommendationPriority.MEDIUM,
                            affected_metric="Supply Chain & Fulfillment",
                            suggested_action="Increase production capacity and procurement to prevent stockouts during peak forecast periods.",
                            source_module="Forecasting",
                        )
                    )

    def _analyze_anomaly(self, anomaly_report: Dict[str, Any], items: List[RecommendationItem]):
        """Analyzes anomaly detection findings for risk recommendations."""
        aff_cnt = anomaly_report.get("affected_rows_count", 0)
        pct = anomaly_report.get("anomaly_percentage", 0.0)
        max_sev = anomaly_report.get("max_severity", "Low")

        if aff_cnt > 0:
            sev_level = RecommendationPriority.CRITICAL if max_sev == "Critical" else (
                RecommendationPriority.HIGH if pct > 5.0 else RecommendationPriority.MEDIUM
            )
            items.append(
                RecommendationItem(
                    id=f"item_{uuid.uuid4().hex[:8]}",
                    title="Audit and Investigate Flagged Anomalous Transactions",
                    explanation=f"Anomaly detection flagged {aff_cnt:,} records ({pct}% of dataset) as operational or financial anomalies.",
                    evidence=f"Flagged Anomalies: {aff_cnt:,} rows ({pct}%) | Maximum Severity Level: {max_sev}.",
                    severity_priority=sev_level,
                    affected_metric="Operational & Risk Compliance",
                    suggested_action="Inspect affected row indices in the Anomaly Detection table and verify suspicious transactions.",
                    source_module="Anomaly Detection",
                )
            )

    def _analyze_bi(self, bi_report: Dict[str, Any], items: List[RecommendationItem]):
        """Analyzes business intelligence findings for customer retention & profit recommendations."""
        kpis = bi_report.get("business_kpis", {})
        tot_rev = kpis.get("total_revenue", 0.0)
        tot_cust = kpis.get("total_customers", 0)
        repeat_rate = kpis.get("repeat_purchase_rate", 0.0)
        churn_summary = bi_report.get("churn_summary", {})
        clv_summary = bi_report.get("clv_summary", {})

        if repeat_rate < 20.0 and tot_cust > 10:
            items.append(
                RecommendationItem(
                    id=f"item_{uuid.uuid4().hex[:8]}",
                    title="Launch Loyalty & Retention Campaigns for Low Repeat Rate",
                    explanation=f"Repeat purchase rate is currently at {repeat_rate}%, indicating a high proportion of one-time buyers.",
                    evidence=f"Repeat Purchase Rate: {repeat_rate}% across {tot_cust:,} total customers.",
                    severity_priority=RecommendationPriority.HIGH,
                    affected_metric="Customer Retention",
                    suggested_action="Implement post-purchase follow-up emails, loyalty rewards, and targeted re-engagement discounts.",
                    source_module="BI Module",
                )
            )

        if churn_summary:
            high_churn = churn_summary.get("high_risk_customer_count", 0)
            churn_rate = churn_summary.get("overall_churn_rate_pct", 0.0)
            if high_churn > 0:
                items.append(
                    RecommendationItem(
                        id=f"item_{uuid.uuid4().hex[:8]}",
                        title="Target High-Risk Customer Segments to Prevent Churn",
                        explanation=f"Identified {high_churn:,} high-value customers at critical risk of churning (overall churn rate: {churn_rate}%).",
                        evidence=f"High Churn Risk Count: {high_churn:,} customers | Overall Churn Rate: {churn_rate}%.",
                        severity_priority=RecommendationPriority.CRITICAL if high_churn >= 5 else RecommendationPriority.HIGH,
                        affected_metric="Customer Churn & Lifetime Value",
                        suggested_action="Deploy automated retention offers and personalized customer success outreach to high-risk accounts.",
                        source_module="BI Module",
                    )
                )

    def _analyze_xai(self, xai_report: Dict[str, Any], items: List[RecommendationItem]):
        """Analyzes SHAP feature importances for feature review recommendations."""
        global_imp = xai_report.get("global_importance", [])
        if len(global_imp) >= 2:
            top1 = global_imp[0]
            top2 = global_imp[1]
            items.append(
                RecommendationItem(
                    id=f"item_{uuid.uuid4().hex[:8]}",
                    title=f"Review Primary Drivers Affecting Outcome: '{top1.get('feature_name')}' & '{top2.get('feature_name')}'",
                    explanation=f"SHAP explainability identifies '{top1.get('feature_name')}' and '{top2.get('feature_name')}' as the strongest influences on model predictions.",
                    evidence=f"Top Driver 1: {top1.get('feature_name')} (SHAP impact: {top1.get('mean_abs_shap_value')}) | Top Driver 2: {top2.get('feature_name')} (SHAP impact: {top2.get('mean_abs_shap_value')}).",
                    severity_priority=RecommendationPriority.MEDIUM,
                    affected_metric="Key Prediction Drivers",
                    suggested_action=f"Focus strategic interventions and quality control efforts on regulating '{top1.get('feature_name')}'.",
                    source_module="SHAP XAI",
                )
            )
