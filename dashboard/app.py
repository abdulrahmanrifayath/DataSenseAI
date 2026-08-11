"""Streamlit Analytics Dashboard for DataSense AI - Unified Executive Decision Platform."""

import os
import sys
import io
import json
import datetime
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace directory to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configuration.settings import settings
from dashboard.api_client import DataSenseAPIClient
from datasense.data_processing.ingestion import DataIngestionService
from datasense.data_processing.validator import DataValidator
from datasense.data_processing.preprocessor import DataPreprocessor
from datasense.data_processing.schemas import ValidationReport, PreprocessingConfig, PreprocessingReport
from datasense.eda.engine import EDAEngine
from datasense.eda.schemas import EDAReport
from datasense.ml_models.trainer import ModelTrainer, determine_task_type
from datasense.ml_models.registry import LocalModelRegistry
from datasense.ml_models.schemas import (
    TaskType,
    TrainingConfig,
    ModelComparisonReport,
    ClassificationAlgorithm,
    RegressionAlgorithm,
    PredictionRequest,
)
from datasense.forecasting.engine import ForecastingEngine
from datasense.forecasting.schemas import ForecastingConfig, ForecastingReport
from datasense.anomaly_detection.detector import AnomalyDetector
from datasense.anomaly_detection.schemas import AnomalyConfig, AnomalyMethod, AnomalyReport
from datasense.bi.engine import BIEngine
from datasense.bi.schemas import ColumnMappingConfig, BIAnalysisReport
from datasense.xai.service import XAIExplanationService
from datasense.xai.schemas import XAIReport
from datasense.recommendations.engine import BusinessRecommendationEngine
from datasense.recommendations.schemas import RecommendationReport, RecommendationPriority


# Initialize API Client
api_client = DataSenseAPIClient()

# Streamlit Page Configuration
st.set_page_config(
    page_title="DataSense AI Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Executive Theme Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .status-badge-healthy {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_quality_gauge(score: float):
    """Renders Plotly gauge indicator for Data Quality Score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Data Quality Index", "font": {"size": 18}},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#2563EB"},
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 80], "color": "#FEF3C7"},
                    {"range": [80, 100], "color": "#DCFCE7"},
                ],
                "threshold": {
                    "line": {"color": "#15803D", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Sidebar Navigation System
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=56)
st.sidebar.title("DataSense AI")
st.sidebar.caption("Intelligent Business Intelligence & Predictive Analytics Platform")

nav_option = st.sidebar.radio(
    "Platform Navigation",
    [
        "Home / Executive Overview",
        "Dataset Upload",
        "Data Quality",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Machine Learning",
        "Forecasting",
        "Anomaly Detection",
        "Customer Segmentation",
        "Explainable AI",
        "Business Recommendations",
        "Model History",
    ],
)

st.sidebar.divider()

# Backend Health Check
health_data = api_client.check_health()
if health_data:
    st.sidebar.success(f"Backend API: {health_data.get('status', 'online').upper()}")
    db_info = health_data.get("database", {})
    if db_info.get("connected"):
        st.sidebar.caption(f"🟢 DB: Connected ({db_info.get('details', {}).get('backend', 'SQLite/PostgreSQL')})")
    sys_m = health_data.get("system_metrics", {})
    if sys_m:
        st.sidebar.caption(f"💻 CPU: {sys_m.get('cpu_usage_pct')} | RAM: {sys_m.get('memory_usage_pct')}%")
else:
    st.sidebar.info("⚡ Backend API: Offline (Running in direct python engine mode)")


# Header Banner
st.markdown('<div class="main-header">DataSense AI Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Unified Business Intelligence, Predictive Analytics, Forecasting, & Recommendation Engine</div>',
    unsafe_allow_html=True,
)


# ==========================================
# 1. HOME / EXECUTIVE OVERVIEW
# ==========================================
if nav_option == "Home / Executive Overview":
    st.header("🏠 Executive Overview Dashboard")
    st.write("Aggregated executive metrics, data quality scores, forecasting trends, anomaly alerts, best predictive models, and high-priority business recommendations.")

    active_df = st.session_state.get("active_df", None)

    # Calculate live aggregate metrics
    n_rows = len(active_df) if active_df is not None else 0
    n_cols = len(active_df.columns) if active_df is not None else 0
    
    # Calculate Data Quality Score
    if active_df is not None and not active_df.empty:
        val_rep = DataValidator(active_df).validate()
        dq_score = val_rep.quality_score
    else:
        dq_score = 0.0

    # Retrieve Module Reports from Session State
    eda_rep: Optional[EDAReport] = st.session_state.get("eda_report", None)
    ml_rep: Optional[ModelComparisonReport] = st.session_state.get("ml_report", None)
    fc_rep: Optional[ForecastingReport] = st.session_state.get("fc_report", None)
    anom_rep: Optional[AnomalyReport] = st.session_state.get("anom_report", None)
    bi_rep: Optional[BIAnalysisReport] = st.session_state.get("bi_report", None)
    rec_rep: Optional[RecommendationReport] = st.session_state.get("recommendation_report", None)

    # Top Executive Key Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dataset Size", f"{n_rows:,} × {n_cols:,}" if n_rows > 0 else "No Dataset")
    m2.metric("Data Quality Score", f"{dq_score:.1f}%" if dq_score > 0 else "N/A")
    
    tot_rev = bi_rep.business_kpis.total_revenue if bi_rep and bi_rep.business_kpis else 0.0
    m3.metric("Total Revenue", f"${tot_rev:,.2f}" if tot_rev > 0 else "$0.00")
    
    anom_cnt = anom_rep.affected_rows_count if anom_rep else 0
    m4.metric("Flagged Anomalies", f"{anom_cnt:,}" if anom_cnt > 0 else "0")
    
    best_mdl = ml_rep.best_model_name if ml_rep else "Not Trained"
    m5.metric("Best ML Model", best_mdl)

    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### 📈 Time-Series Forecast & Demand Trend")
        if fc_rep and fc_rep.results:
            best_f = next((r for r in fc_rep.results if r.is_best), fc_rep.results[0])
            st.info(f"**Best Forecasting Model:** {best_f.model_name} | **sMAPE:** {best_f.metrics.smape:.2f}% | **MAPE:** {best_f.metrics.mape:.2f}%")
            if best_f.future_forecast:
                fut_df = pd.DataFrame([f.model_dump() for f in best_f.future_forecast])
                fig_f = px.line(fut_df, x="timestamp", y="predicted_value", title=f"Projected Future Demand ({len(fut_df)} Periods)", markers=True)
                fig_f.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.info("ℹ️ Run the **Forecasting** module to populate projected demand trends here.")

        st.markdown("#### 🚨 Anomaly Detection Summary")
        if anom_rep:
            st.warning(f"**Flagged Anomalies:** {anom_rep.affected_rows_count:,} rows ({anom_rep.anomaly_percentage}%) | **Max Severity:** {anom_rep.max_severity}")
        else:
            st.info("ℹ️ Run the **Anomaly Detection** module to audit outliers and risk factors here.")

    with col_r:
        st.markdown("#### 🏆 Machine Learning Leaderboard Summary")
        if ml_rep and ml_rep.comparison_table:
            ld_df = pd.DataFrame([c.model_dump() for c in ml_rep.comparison_table])
            st.dataframe(ld_df[["model_name", "task_type", "primary_metric_name", "primary_metric_value", "is_best"]], use_container_width=True)
        else:
            st.info("ℹ️ Run the **Machine Learning** module to train predictive models here.")

        st.markdown("#### 💡 High-Priority Executive Recommendations")
        if rec_rep and rec_rep.items:
            high_recs = [item for item in rec_rep.items if item.severity_priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]][:3]
            if not high_recs:
                high_recs = rec_rep.items[:3]
            for item in high_recs:
                st.error(f"**[{item.severity_priority.value}] {item.title}**\n\n{item.explanation}")
        else:
            st.info("ℹ️ Run the **Business Recommendations** module to synthesize data-driven executive actions.")


# ==========================================
# 2. DATASET UPLOAD
# ==========================================
elif nav_option == "Dataset Upload":
    st.header("📥 Dataset Upload & Validation")
    st.write("Upload a business dataset (CSV or Excel) or select a pre-loaded benchmark e-commerce transaction dataset.")

    source_tab1, source_tab2 = st.tabs(["📁 File Upload (CSV / Excel)", "🗄️ Benchmark E-Commerce Demo Dataset"])

    uploaded_df = None
    uploaded_filename = "dataset.csv"

    with source_tab1:
        uploaded_file = st.file_uploader("Choose a CSV or Excel dataset file", type=["csv", "xlsx", "xls"], key="up_file")
        if uploaded_file is not None:
            uploaded_filename = uploaded_file.name
            try:
                bytes_data = uploaded_file.getvalue()
                uploaded_df = DataIngestionService.ingest_file(bytes_data, uploaded_filename)
                st.session_state["active_df"] = uploaded_df
                st.success(f"Successfully loaded '{uploaded_filename}' ({len(uploaded_df)} rows, {len(uploaded_df.columns)} columns).")
            except Exception as err:
                st.error(f"Failed to ingest file: {err}")

    with source_tab2:
        if st.button("🚀 Load Synthetic E-Commerce Benchmark Dataset", type="primary"):
            np.random.seed(42)
            n_samples = 300
            dates = pd.date_range(start="2025-01-01", periods=n_samples, freq="D")
            cust_ids = [f"CUST_{np.random.randint(100, 150):04d}" for _ in range(n_samples)]
            sales = np.round(np.random.gamma(shape=3.0, scale=40.0, size=n_samples) + 20, 2)
            profit = np.round(sales * np.random.uniform(0.15, 0.40, size=n_samples), 2)
            qty = np.random.randint(1, 10, size=n_samples)
            
            demo_df = pd.DataFrame({
                "customer_id": cust_ids,
                "order_date": dates,
                "revenue": sales,
                "profit": profit,
                "quantity": qty,
            })
            st.session_state["active_df"] = demo_df
            st.success(f"Loaded benchmark dataset ({len(demo_df)} rows × {len(demo_df.columns)} columns).")

    active_df = st.session_state.get("active_df", None)
    if active_df is not None and not active_df.empty:
        st.divider()
        st.subheader("📋 Dataset Preview & Column Schema")
        st.dataframe(active_df.head(10), use_container_width=True)


# ==========================================
# 3. DATA QUALITY
# ==========================================
elif nav_option == "Data Quality":
    st.header("🔍 Data Quality Profiling & Validation Report")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to perform data quality profiling.")
    else:
        validator = DataValidator(active_df)
        report: ValidationReport = validator.validate()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{report.row_count:,}")
        c2.metric("Total Columns", f"{report.column_count:,}")
        c3.metric("Missing Cells", f"{report.total_missing_cells:,} ({report.missing_cell_percentage}%)")
        c4.metric("Duplicate Rows", f"{report.duplicate_rows_count:,} ({report.duplicate_rows_percentage}%)")

        st.divider()

        col_g, col_w = st.columns([1, 2])
        with col_g:
            st.plotly_chart(render_quality_gauge(report.quality_score), use_container_width=True)
        with col_w:
            st.subheader("🚨 Data Quality Warnings")
            if not report.warnings:
                st.success("🎉 Excellent! No critical data quality issues or anomalies detected.")
            else:
                for warn in report.warnings:
                    st.warning(f"⚠️ **{warn.column or 'Dataset'}:** {warn.message}")

        st.subheader("📊 Detailed Column Quality Metrics")
        cols_df = pd.DataFrame([c.model_dump() for c in report.column_metrics])
        st.dataframe(cols_df[["column_name", "inferred_data_type", "null_count", "null_percentage", "distinct_values_count"]], use_container_width=True)


# ==========================================
# 4. DATA CLEANING
# ==========================================
elif nav_option == "Data Cleaning":
    st.header("🧹 Data Preprocessing & Cleaning Engine")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to run data cleaning.")
    else:
        st.write("Configure data cleaning rules (missing value imputation, outlier handling, duplicate removal, constant column filtering).")

        col1, col2, col3 = st.columns(3)
        with col1:
            impute_strat = st.selectbox("Missing Value Strategy:", ["median", "mean", "mode", "drop"])
        with col2:
            outlier_strat = st.selectbox("Outlier Handling Strategy:", ["clip", "drop", "none"])
        with col3:
            drop_dups = st.checkbox("Remove Duplicate Rows", value=True)

        if st.button("🧼 Run Data Preprocessing Pipeline", type="primary", use_container_width=True):
            with st.spinner("Cleaning dataset and transforming features..."):
                try:
                    preprocessor = DataPreprocessor()
                    config = PreprocessingConfig(
                        impute_missing=True,
                        missing_numeric_strategy=impute_strat,
                        handle_outliers=(outlier_strat != "none"),
                        outlier_method=outlier_strat if outlier_strat != "none" else "clip",
                        remove_duplicates=drop_dups,
                    )
                    cleaned_df = preprocessor.fit_transform(active_df, config=config)
                    st.session_state["cleaned_df"] = cleaned_df
                    st.session_state["active_df"] = cleaned_df
                    st.success(f"Preprocessing completed! Cleaned dataset: {len(cleaned_df)} rows × {len(cleaned_df.columns)} columns.")
                except Exception as clean_err:
                    st.error(f"Preprocessing failed: {clean_err}")

        cleaned_df = st.session_state.get("cleaned_df", None)
        if cleaned_df is not None:
            st.divider()
            st.subheader("✨ Cleaned Dataset Preview")
            st.dataframe(cleaned_df.head(10), use_container_width=True)


# ==========================================
# 5. EXPLORATORY DATA ANALYSIS
# ==========================================
elif nav_option == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to perform EDA.")
    else:
        if st.button("📈 Run Full Exploratory Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing dataset statistical distributions and correlations..."):
                try:
                    engine = EDAEngine(active_df)
                    report: EDAReport = engine.generate_report()
                    st.session_state["eda_report"] = report
                    st.success("EDA generated successfully!")
                except Exception as eda_err:
                    st.error(f"EDA failed: {eda_err}")

        if "eda_report" in st.session_state:
            report: EDAReport = st.session_state["eda_report"]
            st.divider()

            t1, t2, t3 = st.tabs(["📊 Summary Statistics", "🔥 Correlation Matrix", "💡 Automated Insights"])
            with t1:
                st.subheader("Numerical Feature Describe Table")
                num_stats_df = pd.DataFrame([{"feature": k, **v.model_dump()} for k, v in report.numerical_stats.items()])
                st.dataframe(num_stats_df, use_container_width=True)




            with t2:
                if report.correlation_matrix:
                    corr_df = pd.DataFrame(report.correlation_matrix)
                    fig_corr = px.imshow(corr_df, text_auto=True, title="Feature Correlation Heatmap", color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig_corr, use_container_width=True)
            with t3:
                for ins in report.automated_insights:
                    st.info(f"💡 **Insight:** {ins}")


# ==========================================
# 6. MACHINE LEARNING
# ==========================================
elif nav_option == "Machine Learning":
    st.header("🤖 Predictive Machine Learning Engine")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to train machine learning models.")
    else:
        num_cols = [c for c in active_df.columns if pd.api.types.is_numeric_dtype(active_df[c])]
        
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox("Select Target Variable (Y):", options=num_cols if num_cols else list(active_df.columns), index=min(len(num_cols)-1, 0))
        with c2:
            tune_hp = st.checkbox("Enable Hyperparameter Tuning", value=False)

        if st.button("🚀 Train Machine Learning Models", type="primary", use_container_width=True):
            with st.spinner("Training classification and regression ensemble models..."):
                try:
                    trainer = ModelTrainer()
                    config = TrainingConfig(target_column=target_col, tune_hyperparameters=tune_hp)
                    report: ModelComparisonReport = trainer.train_and_compare(active_df, config=config)
                    st.session_state["ml_report"] = report
                    st.success(f"Training completed! Best Model: {report.best_model_name}")
                except Exception as ml_err:
                    st.error(f"ML training failed: {ml_err}")

        if "ml_report" in st.session_state:
            report: ModelComparisonReport = st.session_state["ml_report"]
            st.divider()
            st.markdown(f"### 🏆 Best Model: `{report.best_model_name}` ({report.task_type.title()})")
            
            comp_df = pd.DataFrame([c.model_dump() for c in report.comparison_table])
            st.dataframe(comp_df, use_container_width=True)


# ==========================================
# 7. FORECASTING
# ==========================================
elif nav_option == "Forecasting":
    st.header("📈 Time-Series Forecasting Module")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to perform forecasting.")
    else:
        date_cols = [c for c in active_df.columns if "date" in c.lower() or "time" in c.lower()]
        num_cols = [c for c in active_df.columns if pd.api.types.is_numeric_dtype(active_df[c])]

        col1, col2, col3 = st.columns(3)
        with col1:
            dt_col = st.selectbox("Datetime Column:", options=date_cols if date_cols else list(active_df.columns), index=0)
        with col2:
            tgt_col = st.selectbox("Numerical Target Column:", options=num_cols if num_cols else list(active_df.columns), index=0)
        with col3:
            horizon = st.slider("Forecast Horizon (Periods):", min_value=7, max_value=60, value=14)

        if st.button("🔮 Generate Time-Series Forecast", type="primary", use_container_width=True):
            with st.spinner("Fitting forecasting models..."):
                try:
                    engine = ForecastingEngine()
                    config = ForecastingConfig(date_column=dt_col, target_column=tgt_col, forecast_horizon=horizon)
                    report: ForecastingReport = engine.run_forecasting(active_df, config=config)
                    st.session_state["fc_report"] = report
                    st.success(f"Forecasting complete! Best Model: {report.best_model_name}")
                except Exception as fc_err:
                    st.error(f"Forecasting failed: {fc_err}")

        if "fc_report" in st.session_state:
            report: ForecastingReport = st.session_state["fc_report"]
            st.divider()
            if report.results:
                best_res = next((r for r in report.results if r.is_best), report.results[0])
                if best_res.future_forecast:
                    fut_df = pd.DataFrame([f.model_dump() for f in best_res.future_forecast])
                    fig_fc = px.line(fut_df, x="timestamp", y="predicted_value", title=f"Future Forecast - {report.best_model_name}", markers=True)
                    st.plotly_chart(fig_fc, use_container_width=True)


# ==========================================
# 8. ANOMALY DETECTION
# ==========================================
elif nav_option == "Anomaly Detection":
    st.header("🚨 Anomaly Detection Module")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to perform anomaly detection.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Anomaly Detection Method:", ["ensemble", "isolation_forest", "zscore", "iqr"])
        with col2:
            contam = st.slider("Contamination Rate:", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

        if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
            with st.spinner("Running anomaly detection algorithms..."):
                try:
                    detector = AnomalyDetector()
                    config = AnomalyConfig(method=AnomalyMethod(method), contamination=contam)
                    report: AnomalyReport = detector.detect(active_df, config=config)
                    st.session_state["anom_report"] = report
                    st.success(f"Detected {report.affected_rows_count} anomalous rows ({report.anomaly_percentage}% of dataset).")
                except Exception as anom_err:
                    st.error(f"Anomaly detection failed: {anom_err}")

        if "anom_report" in st.session_state:
            report: AnomalyReport = st.session_state["anom_report"]
            st.divider()
            st.warning(f"**Flagged Anomalies:** {report.affected_rows_count:,} rows | **Max Severity:** {report.max_severity}")


# ==========================================
# 9. CUSTOMER SEGMENTATION
# ==========================================
elif nav_option == "Customer Segmentation":
    st.header("🎯 Customer Analytics & Segmentation (BI Engine)")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to run customer segmentation.")
    else:
        if st.button("🚀 Run Customer Analytics & RFM Segmentation", type="primary", use_container_width=True):
            with st.spinner("Computing RFM scores, K-Means & Hierarchical clustering..."):
                try:
                    engine = BIEngine()
                    report: BIAnalysisReport = engine.analyze(active_df)
                    st.session_state["bi_report"] = report
                    st.success("Customer Analytics & RFM Segmentation complete!")
                except Exception as bi_err:
                    st.error(f"BI analysis failed: {bi_err}")

        if "bi_report" in st.session_state:
            report: BIAnalysisReport = st.session_state["bi_report"]
            st.divider()
            kpis = report.business_kpis
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers", f"{kpis.total_customers:,}")
            c2.metric("Total Revenue", f"${kpis.total_revenue:,.2f}")
            c3.metric("Average Order Value (AOV)", f"${kpis.average_order_value:,.2f}")


# ==========================================
# 10. EXPLAINABLE AI
# ==========================================
elif nav_option == "Explainable AI":
    st.header("🧠 Model Explainability & SHAP Analytics")
    active_df = st.session_state.get("active_df", None)

    if active_df is None or active_df.empty:
        st.warning("⚠️ Please upload a dataset in 'Dataset Upload' to run Model Explainability.")
    else:
        num_cols = [c for c in active_df.columns if pd.api.types.is_numeric_dtype(active_df[c])]
        col1, col2 = st.columns(2)
        with col1:
            target_col_xai = st.selectbox("Target Column for Explanation:", options=num_cols if num_cols else list(active_df.columns), index=0)
        with col2:
            row_idx_xai = st.slider("Instance Row Index:", min_value=0, max_value=max(0, len(active_df) - 1), value=0)

        if st.button("🧠 Compute SHAP Model Explanations", type="primary", use_container_width=True):
            with st.spinner("Fitting surrogate model and computing SHAP values..."):
                try:
                    num_feats = [c for c in num_cols if c != target_col_xai]
                    X_num = active_df[num_feats].fillna(active_df[num_feats].median()).to_numpy()
                    y_num = active_df[target_col_xai].to_numpy()

                    task_type = determine_task_type(active_df, target_col_xai)
                    if task_type == "classification":
                        model_xai = RandomForestClassifier(n_estimators=50, random_state=42)
                    else:
                        model_xai = RandomForestRegressor(n_estimators=50, random_state=42)

                    model_xai.fit(X_num, y_num)

                    service = XAIExplanationService()
                    report: XAIReport = service.explain(
                        model=model_xai,
                        X_df=pd.DataFrame(X_num, columns=num_feats),
                        feature_names=num_feats,
                        task_type=task_type,
                        instance_indices=[row_idx_xai],
                    )
                    st.session_state["xai_report"] = report
                    st.success("SHAP model explanations computed successfully!")
                except Exception as xai_err:
                    st.error(f"SHAP explanation failed: {xai_err}")

        if "xai_report" in st.session_state:
            report: XAIReport = st.session_state["xai_report"]
            st.divider()
            if report.summary_chart_plotly_json:
                st.markdown("#### 🌐 Global Feature Importance (Mean |SHAP Value|)")
                fig_glob = go.Figure(json.loads(report.summary_chart_plotly_json))
                st.plotly_chart(fig_glob, use_container_width=True)


# ==========================================
# 11. BUSINESS RECOMMENDATIONS
# ==========================================
elif nav_option == "Business Recommendations":
    st.header("💡 Business Recommendation Engine")
    active_df = st.session_state.get("active_df", None)

    if st.button("💡 Synthesize & Generate Business Recommendations", type="primary", use_container_width=True):
        with st.spinner("Synthesizing multi-module analytics findings..."):
            try:
                engine = BusinessRecommendationEngine()
                eda_rep = st.session_state.get("eda_report", {}).model_dump() if hasattr(st.session_state.get("eda_report", None), "model_dump") else None
                ml_rep = st.session_state.get("ml_report", {}).model_dump() if hasattr(st.session_state.get("ml_report", None), "model_dump") else None
                fc_rep = st.session_state.get("fc_report", {}).model_dump() if hasattr(st.session_state.get("fc_report", None), "model_dump") else None
                anom_rep = st.session_state.get("anom_report", {}).model_dump() if hasattr(st.session_state.get("anom_report", None), "model_dump") else None
                bi_rep = st.session_state.get("bi_report", {}).model_dump() if hasattr(st.session_state.get("bi_report", None), "model_dump") else None
                xai_rep = st.session_state.get("xai_report", {}).model_dump() if hasattr(st.session_state.get("xai_report", None), "model_dump") else None

                report: RecommendationReport = engine.generate_recommendations(
                    df=active_df,
                    eda_report=eda_rep,
                    ml_report=ml_rep,
                    forecast_report=fc_rep,
                    anomaly_report=anom_rep,
                    bi_report=bi_rep,
                    xai_report=xai_rep,
                )
                st.session_state["recommendation_report"] = report
                st.success("Business recommendations generated successfully!")
            except Exception as rec_err:
                st.error(f"Recommendation generation failed: {rec_err}")

    if "recommendation_report" in st.session_state:
        report: RecommendationReport = st.session_state["recommendation_report"]
        st.divider()
        for item in report.items:
            priority_val = item.severity_priority.value if hasattr(item.severity_priority, "value") else str(item.severity_priority)
            st.error(f"**[{priority_val}] {item.title}** ({item.source_module})\n\n{item.explanation}\n\n💡 **Evidence:** {item.evidence}\n\n✅ **Action:** {item.suggested_action}")


# ==========================================
# 12. MODEL HISTORY & PREDICTION SANDBOX
# ==========================================
elif nav_option == "Model History":
    st.header("🗄️ Model Registry & Real-Time Prediction Sandbox")
    st.write("Browse persistent trained models in LocalModelRegistry and test real-time inference predictions.")

    registry = LocalModelRegistry()
    stored_models = registry.list_models()

    if not stored_models:
        st.info("ℹ️ No trained models stored in model registry yet. Train a model in the **Machine Learning** section!")
    else:
        st.subheader("📋 Registered Machine Learning Models")
        reg_rows = []
        for m in stored_models:
            reg_rows.append({
                "model_id": m.model_id,
                "model_name": m.model_name,
                "task_type": m.task_type,
                "target_column": m.target_column,
                "primary_metric": m.metrics.get("primary_metric_name", "N/A"),
                "score": round(m.metrics.get("primary_metric_value", 0.0), 4),
                "is_best": "🏆 Best" if m.is_best else "",
                "created_at": m.created_at,
            })
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True)

        st.divider()
        st.subheader("🧪 Real-Time Model Inference Predictor Sandbox")

        model_ids = [m.model_id for m in stored_models]
        selected_model_id = st.selectbox("Select Model for Inference Testing:", options=model_ids)

        selected_record = next((m for m in stored_models if m.model_id == selected_model_id), stored_models[0])
        st.info(f"**Selected Model:** {selected_record.model_name} | **Task Type:** {selected_record.task_type} | **Target Column:** `{selected_record.target_column}`")

        # Dynamically build feature input form
        feats = selected_record.feature_names or []
        if not feats:
            feats = ["feature_1", "feature_2"]

        st.markdown("##### Enter Sample Input Feature Values:")
        input_dict = {}
        cols_input = st.columns(min(4, max(1, len(feats))))
        for idx, f_name in enumerate(feats):
            with cols_input[idx % len(cols_input)]:
                input_dict[f_name] = st.number_input(f"`{f_name}`:", value=1.0, key=f"input_{f_name}")

        if st.button("⚡ Run Real-Time Prediction Inference", type="primary", use_container_width=True):
            with st.spinner("Executing model prediction..."):
                try:
                    loaded_model = registry.load_model(selected_model_id)
                    X_sample = pd.DataFrame([input_dict])[feats].to_numpy()
                    pred_val = loaded_model.predict(X_sample)[0]

                    if selected_record.task_type == "classification" and hasattr(loaded_model, "predict_proba"):
                        proba = loaded_model.predict_proba(X_sample)[0]
                        st.success(f"🎯 **Predicted Output Class:** `{pred_val}` | **Class Probabilities:** {np.round(proba, 4)}")
                    else:
                        st.success(f"🎯 **Predicted Value:** `{pred_val:,.4f}`")
                except Exception as pred_err:
                    st.error(f"Inference prediction failed: {pred_err}")
