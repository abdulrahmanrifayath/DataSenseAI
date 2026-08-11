"""Streamlit Analytics Dashboard for DataSense AI - Data Ingestion & Data Validation Engine."""

import os
import sys
import io
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace directory to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configuration.settings import settings
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
        font-weight: 700;
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
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .warning-critical {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .warning-warning {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .warning-info {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_backend_health():
    """Fetch backend API health check status."""
    try:
        url = f"{settings.BACKEND_API_URL}/health"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def render_quality_gauge(score: float):
    """Render a clean Plotly gauge chart for data quality score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Overall Data Quality Score", "font": {"size": 18, "color": "#1E293B"}},
            number={"suffix": "/100", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#0284C7" if score >= 80 else ("#F59E0B" if score >= 60 else "#EF4444")},
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
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
st.sidebar.title("DataSense AI")
st.sidebar.caption("Intelligent Business Intelligence & Analytics")

nav_option = st.sidebar.radio(
    "Navigation",
    [
        "Data Ingestion & Validation",
        "Data Preprocessing & Cleaning",
        "Platform Overview",
        "Exploratory Data Analysis",
        "Predictive Modeling & ML",
        "Time-Series Forecasting",
        "Anomaly Detection",
        "Business Intelligence & Analytics",
        "Model Explainability & SHAP",
        "Business Recommendations Engine",
        "System Diagnostics",
    ],
)




st.sidebar.divider()


# Backend Health Status Badge in Sidebar
health_data = fetch_backend_health()
if health_data:
    st.sidebar.success(f"Backend API: {health_data.get('status', 'online').upper()}")
    db_info = health_data.get("database", {})
    if db_info.get("connected"):
        st.sidebar.caption(f"🟢 DB: Connected ({db_info.get('details', {}).get('backend', 'PostgreSQL')})")
    else:
        st.sidebar.caption("🟡 DB: Standby Mode")
else:
    st.sidebar.warning("⚡ Backend API: Offline (Standalone UI Mode)")

# Header Section
st.markdown('<div class="main-header">DataSense AI Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Data Validation, Quality Profiling & Predictive Analytics Engine</div>',
    unsafe_allow_html=True,
)


if nav_option == "Data Ingestion & Validation":
    st.header("📥 Data Ingestion & Quality Validation Engine")
    st.write("Upload a business dataset (CSV or Excel) or ingest directly from PostgreSQL to run automated data profiling and quality validation.")

    source_tab1, source_tab2 = st.tabs(["📁 File Upload (CSV / Excel)", "🗄️ Database Ingestion"])

    uploaded_df = None
    uploaded_filename = "dataset.csv"

    with source_tab1:
        uploaded_file = st.file_uploader("Choose a CSV or Excel dataset file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            uploaded_filename = uploaded_file.name
            try:
                bytes_data = uploaded_file.getvalue()
                uploaded_df = DataIngestionService.ingest_file(bytes_data, uploaded_filename)
                st.success(f"Successfully ingested file '{uploaded_filename}' ({len(uploaded_df)} rows, {len(uploaded_df.columns)} columns)")
            except Exception as err:
                st.error(f"Failed to ingest file: {err}")

    with source_tab2:
        st.subheader("Extract Dataset from PostgreSQL Database")
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            db_conn_str = st.text_input("Database Connection String", value=settings.DATABASE_URL)
        with col_db2:
            db_query = st.text_input("SQL Query or Table Name", value="SELECT * FROM dataset_metadata")

        if st.button("Execute Database Extraction"):
            try:
                uploaded_df = DataIngestionService.load_from_db(db_conn_str, db_query)
                uploaded_filename = f"db_{db_query[:20]}"
                st.success(f"Successfully extracted dataset from DB query! ({len(uploaded_df)} rows)")
            except Exception as err:
                st.error(f"Database query extraction failed: {err}")

    st.divider()

    if uploaded_df is not None and not uploaded_df.empty:
        # Run DataValidator
        validator = DataValidator(uploaded_df)
        report: ValidationReport = validator.validate()

        # Top Executive Summary Metrics
        st.subheader("📊 Dataset Statistics & Quality Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Rows", f"{report.row_count:,}")
        with col2:
            st.metric("Total Columns", f"{report.column_count:,}")
        with col3:
            st.metric("Missing Cells", f"{report.total_missing_cells:,} ({report.missing_cell_percentage}%)")
        with col4:
            st.metric("Duplicate Rows", f"{report.duplicate_rows_count:,} ({report.duplicate_rows_percentage}%)")
        with col5:
            st.metric("Quality Score", f"{report.quality_score} / 100")

        # Quality Gauge & Warnings Side-by-Side
        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.plotly_chart(render_quality_gauge(report.quality_score), use_container_width=True)

        with c_right:
            st.subheader("🚨 Data Quality Issues & Warnings")
            if not report.warnings:
                st.success("🎉 Excellent! No critical data quality issues or anomalies detected.")
            else:
                for w in report.warnings:
                    badge_color = "warning-critical" if w.severity == "CRITICAL" else ("warning-warning" if w.severity == "WARNING" else "warning-info")
                    col_info = f" **(Column: `{w.column}`)**" if w.column else ""
                    st.markdown(
                        f'<div class="{badge_color}"><strong>[{w.severity}] {w.code}</strong>{col_info}: {w.message}</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()

        # Dataset Preview Section
        st.subheader("🔍 Dataset Rows Preview")
        preview_rows_count = st.slider("Select preview row count", min_value=5, max_value=100, value=10, step=5)
        st.dataframe(uploaded_df.head(preview_rows_count), use_container_width=True)

        st.divider()

        # Column Profiles Detail Table
        st.subheader("📋 Column Profiles & DataType Analysis")

        cols_summary = []
        for col_name, profile in report.column_profiles.items():
            cols_summary.append(
                {
                    "Column Name": profile.name,
                    "Pandas Dtype": profile.data_type,
                    "Inferred Type": profile.inferred_type.upper(),
                    "Missing Count": profile.missing_count,
                    "Missing %": f"{profile.missing_percentage}%",
                    "Unique Values": profile.unique_count,
                    "Is ID Key": "YES 🔑" if profile.is_potential_id else "NO",
                    "Is Constant": "YES ⚠️" if profile.is_constant else "NO",
                    "Sample Values": ", ".join(map(str, profile.sample_values[:3])),
                }
            )

        summary_df = pd.DataFrame(cols_summary)
        st.dataframe(summary_df, use_container_width=True)

        # Categorical vs Numerical Column Distribution Chart
        st.subheader("📈 Datatype Distribution Breakdown")
        type_counts = pd.Series(
            {
                "Numerical": len(report.numerical_columns),
                "Categorical": len(report.categorical_columns),
                "Datetime": len(report.datetime_columns),
                "Primary ID": len(report.potential_id_columns),
                "Constant": len(report.constant_columns),
            }
        )
        fig_types = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            labels={"x": "Column Type", "y": "Column Count"},
            color=type_counts.index,
            title="Column Datatype Classification Count",
            template="plotly_white",
        )
        st.plotly_chart(fig_types, use_container_width=True)
        # Store in session state for Preprocessing tab
        st.session_state["active_df"] = uploaded_df

    else:
        st.info("💡 Please upload a CSV or Excel file, or query a database table above to view data quality results.")

elif nav_option == "Data Preprocessing & Cleaning":
    st.header("⚙️ Automated Data Cleaning & Preprocessing Engine")
    st.write("Configure missing value imputation, outlier detection, scaling, encoding, and feature filtering parameters to build a Scikit-Learn preprocessing pipeline.")

    prep_df = st.session_state.get("active_df", None)

    if prep_df is None:
        st.info("💡 No active dataset found from Ingestion. Upload a dataset file below to configure preprocessing:")
        prep_file = st.file_uploader("Choose dataset file for preprocessing", type=["csv", "xlsx", "xls"], key="prep_uploader")
        if prep_file is not None:
            try:
                prep_df = DataIngestionService.ingest_file(prep_file.getvalue(), prep_file.name)
                st.session_state["active_df"] = prep_df
                st.success(f"Ingested '{prep_file.name}' ({len(prep_df)} rows, {len(prep_df.columns)} columns)")
            except Exception as e:
                st.error(f"Failed to ingest dataset: {e}")

    if prep_df is not None and not prep_df.empty:
        st.subheader("1. Configure Pipeline Options")

        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            st.markdown("##### 📌 Column Assignments")
            all_cols = list(prep_df.columns)
            target_col = st.selectbox("Target Column (Optional)", options=[None] + all_cols, index=0)
            id_cols = st.multiselect("Identifier Columns", options=all_cols, default=[c for c in all_cols if "id" in c.lower()])

            st.markdown("##### 🧹 Missing Values Imputation")
            num_impute = st.selectbox("Numerical Impute Strategy", options=["median", "mean", "most_frequent", "constant"], index=0)
            cat_impute = st.selectbox("Categorical Impute Strategy", options=["most_frequent", "constant"], index=0)
            remove_dups = st.checkbox("Remove Duplicate Rows", value=True)

        with cfg_col2:
            st.markdown("##### 🚨 Outlier Detection & Action")
            outlier_method = st.selectbox("Outlier Detection Method", options=["iqr", "zscore", "none"], index=0)
            outlier_thresh = st.slider("Outlier Threshold (IQR multiplier / Z-score)", min_value=1.0, max_value=5.0, value=1.5 if outlier_method == "iqr" else 3.0, step=0.1)
            outlier_action = st.selectbox("Outlier Action", options=["clip", "impute", "drop_rows", "none"], index=0)

            st.markdown("##### 📅 Datetime & Types")
            coerce_types = st.checkbox("Coerce Data Types", value=True)
            convert_dt = st.checkbox("Parse Datetimes & Extract Features", value=True)

        with cfg_col3:
            st.markdown("##### 🔍 Feature Filtering")
            drop_const = st.checkbox("Drop Constant Columns", value=True)
            drop_near_const = st.checkbox("Drop Near-Constant Columns", value=True)
            near_const_thresh = st.slider("Near-Constant Cutoff Ratio", min_value=0.80, max_value=0.99, value=0.98, step=0.01)
            drop_corr = st.checkbox("Drop Highly Correlated Features", value=True)
            corr_thresh = st.slider("Correlation Cutoff (|r|)", min_value=0.70, max_value=0.99, value=0.95, step=0.01)

            st.markdown("##### 📐 Encoding & Scaling")
            cat_encoding = st.selectbox("Categorical Encoding", options=["onehot", "ordinal", "none"], index=0)
            num_scaling = st.selectbox("Numerical Scaling", options=["standard", "minmax", "robust", "none"], index=0)

        st.divider()

        if st.button("🚀 Run Preprocessing Pipeline", type="primary", use_container_width=True):
            config = PreprocessingConfig(
                target_column=target_col,
                identifier_columns=id_cols,
                numerical_impute_strategy=num_impute,
                categorical_impute_strategy=cat_impute,
                remove_duplicates=remove_dups,
                coerce_types=coerce_types,
                convert_datetimes=convert_dt,
                datetime_extract_features=convert_dt,
                outlier_method=outlier_method,
                outlier_threshold=outlier_thresh,
                outlier_action=outlier_action,
                categorical_encoding=cat_encoding,
                numerical_scaling=num_scaling,
                drop_constant=drop_const,
                drop_near_constant=drop_near_const,
                near_constant_threshold=near_const_thresh,
                drop_high_correlation=drop_corr,
                correlation_threshold=corr_thresh,
            )

            try:
                with st.spinner("Executing Scikit-Learn Preprocessing Pipeline..."):
                    preprocessor = DataPreprocessor(config=config)
                    transformed_df = preprocessor.fit_transform(prep_df)
                    report: PreprocessingReport = preprocessor.get_report()

                st.success("✅ Preprocessing Pipeline Completed Successfully!")

                # Executive Summary Metrics
                st.subheader("📊 Preprocessing Results Summary")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Initial Shape", f"{report.initial_shape[0]} × {report.initial_shape[1]}")
                with m2:
                    st.metric("Final Shape", f"{report.final_shape[0]} × {report.final_shape[1]}")
                with m3:
                    st.metric("Missing Values Fixed", f"{report.missing_values_fixed:,}")
                with m4:
                    st.metric("Duplicates Removed", f"{report.duplicates_removed:,}")
                with m5:
                    st.metric("Outliers Detected", f"{report.outliers_detected:,}")

                st.divider()

                # Removed Columns & Decisions
                if report.columns_removed:
                    st.subheader("🗑️ Removed Columns & Recorded Decisions")
                    rem_df = pd.DataFrame(
                        [{"Column Name": k, "Reason for Removal": v} for k, v in report.columns_removed.items()]
                    )
                    st.dataframe(rem_df, use_container_width=True)
                else:
                    st.info("ℹ️ No columns were dropped during feature filtering.")

                # Transformation Steps Log
                st.subheader("📜 Transformation Pipeline Log")
                trans_df = pd.DataFrame(
                    [
                        {
                            "Step": t.step_name,
                            "Action": t.action,
                            "Columns Affected": ", ".join(t.columns_affected[:5]) + ("..." if len(t.columns_affected) > 5 else ""),
                            "Details": str(t.details),
                        }
                        for t in report.transformations
                    ]
                )
                st.dataframe(trans_df, use_container_width=True)

                # Preview Transformed Dataset
                st.subheader("✨ Transformed Dataset Preview")
                st.dataframe(transformed_df.head(20), use_container_width=True)

                # Download Transformed CSV
                csv_bytes = transformed_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Transformed Dataset CSV",
                    data=csv_bytes,
                    file_name="transformed_dataset.csv",
                    mime="text/csv",
                )

            except Exception as err:
                st.error(f"Preprocessing Execution Failed: {err}")


elif nav_option == "Platform Overview":
    st.header("Executive Summary & Core Capabilities")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="System Status", value="Operational" if health_data else "Standby")
    with col2:
        st.metric(label="API Version", value=settings.API_VERSION)
    with col3:
        st.metric(label="Target Host", value=settings.HOST)
    with col4:
        st.metric(label="Backend Port", value=settings.PORT)

    st.divider()

    st.subheader("Platform Module Architecture")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.info("### 🔍 Data Processing & EDA\n- Automated Schema Validation\n- Missing Value Profiling & Imputation\n- Data Quality Scoring Engine")

    with m2:
        st.success("### 🤖 Predictive Analytics\n- Classification & Regression (XGBoost)\n- Time-Series Demand Forecasting\n- Anomaly & Outlier Detection")

    with m3:
        st.warning("### 💡 Explainable AI & Insights\n- SHAP Feature Attribution\n- Customer Cohort Segmentation\n- Actionable Business Recommendations")

elif nav_option == "Exploratory Data Analysis":
    st.header("🔍 Automated Exploratory Data Analysis (EDA) Engine")
    st.write("Generates comprehensive statistical summaries, distribution profiles, correlation heatmaps, outlier analyses, time-series trends, target relationships, and automated textual insights.")

    eda_df = st.session_state.get("active_df", None)

    if eda_df is None:
        st.info("💡 No active dataset found. Upload a dataset file below to run Exploratory Data Analysis:")
        eda_file = st.file_uploader("Choose dataset file for EDA", type=["csv", "xlsx", "xls"], key="eda_uploader")
        if eda_file is not None:
            try:
                eda_df = DataIngestionService.ingest_file(eda_file.getvalue(), eda_file.name)
                st.session_state["active_df"] = eda_df
                st.success(f"Ingested dataset '{eda_file.name}' ({len(eda_df)} rows, {len(eda_df.columns)} columns)")
            except Exception as e:
                st.error(f"Failed to ingest file: {e}")

    if eda_df is not None and not eda_df.empty:
        col_target, col_btn = st.columns([3, 1])
        with col_target:
            all_cols = list(eda_df.columns)
            selected_target = st.selectbox("Select Target Variable for Target-Focused EDA (Optional)", options=[None] + all_cols, index=0)
        with col_btn:
            st.markdown("##")
            run_eda = st.button("🚀 Run Complete EDA Analysis", type="primary", use_container_width=True)

        if run_eda or "eda_report" in st.session_state:
            if run_eda or st.session_state.get("eda_target") != selected_target:
                with st.spinner("Generating automated statistical profiling & Plotly charts..."):
                    engine = EDAEngine(eda_df, target_column=selected_target)
                    report: EDAReport = engine.generate_report()
                    st.session_state["eda_report"] = report
                    st.session_state["eda_target"] = selected_target
            else:
                report: EDAReport = st.session_state["eda_report"]

            st.divider()

            # Sub-tab Navigation
            tab_summary, tab_insights, tab_stats, tab_dist, tab_corr, tab_cat, tab_time, tab_target = st.tabs([
                "📊 Overview",
                "💡 Automatic Insights",
                "📋 Statistics",
                "📈 Distributions & Outliers",
                "🔥 Correlation Heatmap",
                "🏷️ Categorical Analysis",
                "📅 Time Trends",
                "🎯 Target Analysis",
            ])

            with tab_summary:
                st.subheader("Dataset Structure & Metric Overview")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Total Rows", f"{report.summary.row_count:,}")
                with m2:
                    st.metric("Total Columns", f"{report.summary.column_count:,}")
                with m3:
                    st.metric("Missing Cells", f"{report.summary.total_missing_cells:,} ({report.summary.missing_percentage}%)")
                with m4:
                    st.metric("Duplicate Rows", f"{report.summary.duplicate_rows:,}")
                with m5:
                    st.metric("Memory Usage", f"{report.summary.memory_usage_bytes / (1024*1024):.2f} MB")

                st.divider()

                col_chart, col_types = st.columns([2, 1])
                with col_chart:
                    if "missing_bar" in report.charts_plotly_json:
                        fig = go.Figure(json.loads(report.charts_plotly_json["missing_bar"]))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("🎉 Zero missing values detected across dataset!")

                with col_types:
                    st.subheader("Feature Type Breakdown")
                    counts_df = pd.DataFrame(
                        [{"Type": k.upper(), "Count": v} for k, v in report.summary.feature_counts.items()]
                    )
                    st.dataframe(counts_df, use_container_width=True)

            with tab_insights:
                st.subheader("🤖 Automatically Detected Data Insights & Anomaly Alerts")
                if not report.insights:
                    st.info("No critical anomaly or distribution warnings detected.")
                else:
                    for ins in report.insights:
                        badge_class = "warning-critical" if ins.severity == "HIGH" else ("warning-warning" if ins.severity == "MEDIUM" else "warning-info")
                        col_str = f" **(Columns: `{', '.join(ins.affected_columns)}`)**" if ins.affected_columns else ""
                        st.markdown(
                            f'<div class="{badge_class}"><strong>[{ins.severity}] [{ins.category}] {ins.title}</strong>{col_str}<br>{ins.description}</div>',
                            unsafe_allow_html=True,
                        )

            with tab_stats:
                st.subheader("📋 Descriptive Statistics")
                if report.numerical_stats:
                    st.markdown("##### Numerical Feature Statistics")
                    num_list = []
                    for col, nstat in report.numerical_stats.items():
                        num_list.append({
                            "Feature": col,
                            "Mean": nstat.mean,
                            "Std": nstat.std,
                            "Min": nstat.min,
                            "Q25": nstat.q25,
                            "Median": nstat.median,
                            "Q75": nstat.q75,
                            "Max": nstat.max,
                            "Skewness": nstat.skewness,
                            "Kurtosis": nstat.kurtosis,
                            "Missing": nstat.missing_count,
                            "Zeros": nstat.zero_count,
                        })
                    st.dataframe(pd.DataFrame(num_list), use_container_width=True)

                if report.categorical_stats:
                    st.markdown("##### Categorical Feature Statistics")
                    cat_list = []
                    for col, cstat in report.categorical_stats.items():
                        cat_list.append({
                            "Feature": col,
                            "Count": cstat.count,
                            "Unique": cstat.unique_count,
                            "Top Value": cstat.top_value,
                            "Top Frequency": cstat.top_freq,
                            "Top Ratio": f"{cstat.top_ratio:.1%}",
                            "Missing": cstat.missing_count,
                        })
                    st.dataframe(pd.DataFrame(cat_list), use_container_width=True)

            with tab_dist:
                st.subheader("📈 Feature Distribution & Outlier Profiling")
                if report.numerical_stats:
                    selected_num = st.selectbox("Select Numerical Column to Inspect", options=list(report.numerical_stats.keys()))
                    if selected_num:
                        col_fig, col_out = st.columns([2, 1])
                        with col_fig:
                            fig_hist = px.histogram(eda_df, x=selected_num, marginal="box", title=f"Distribution of {selected_num}", template="plotly_white")
                            st.plotly_chart(fig_hist, use_container_width=True)
                        with col_out:
                            if selected_num in report.outlier_analysis:
                                ostat = report.outlier_analysis[selected_num]
                                st.markdown("##### Outlier Metrics")
                                st.metric("IQR Outliers", ostat.iqr_outliers)
                                st.metric("Z-Score Outliers", ostat.zscore_outliers)
                                st.write(f"**IQR Bounds:** `[{ostat.iqr_lower_bound}, {ostat.iqr_upper_bound}]`")
                                st.write(f"**Z-Score Bounds:** `[{ostat.zscore_lower_bound:.2f}, {ostat.zscore_upper_bound:.2f}]`")

            with tab_corr:
                st.subheader("🔥 Feature Correlation Analysis")
                if "corr_heatmap" in report.charts_plotly_json:
                    fig_corr = go.Figure(json.loads(report.charts_plotly_json["corr_heatmap"]))
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Correlation heatmap unavailable (requires at least 2 numerical features).")

                if report.top_correlation_pairs:
                    st.subheader("Top Correlated Feature Pairs")
                    st.dataframe(pd.DataFrame(report.top_correlation_pairs), use_container_width=True)

            with tab_cat:
                st.subheader("🏷️ Categorical Value Frequencies")
                if report.categorical_stats:
                    selected_cat = st.selectbox("Select Categorical Column to Inspect", options=list(report.categorical_stats.keys()))
                    if selected_cat and selected_cat in report.category_frequencies:
                        freqs = report.category_frequencies[selected_cat]
                        fig_cat = px.bar(
                            x=list(freqs.keys()),
                            y=list(freqs.values()),
                            labels={"x": selected_cat, "y": "Frequency"},
                            title=f"Top Categories for {selected_cat}",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)

            with tab_time:
                st.subheader("📅 Time-Series & Temporal Trends")
                if report.time_trends and "trends" in report.time_trends:
                    st.write(f"**Datetime Column Analyzed:** `{report.time_trends.get('datetime_column')}`")
                    if "time_series_line" in report.charts_plotly_json:
                        fig_time = go.Figure(json.loads(report.charts_plotly_json["time_series_line"]))
                        st.plotly_chart(fig_time, use_container_width=True)

                    st.markdown("##### Detected Feature Trends")
                    trend_data = []
                    for c_name, t_info in report.time_trends["trends"].items():
                        trend_data.append({
                            "Feature": c_name,
                            "Direction": t_info["direction"],
                            "Slope": t_info["slope"],
                            "R² Score": t_info["r_squared"],
                            "Min Date": t_info["min_date"],
                            "Max Date": t_info["max_date"],
                        })
                    st.dataframe(pd.DataFrame(trend_data), use_container_width=True)
                else:
                    st.info("No datetime column detected for temporal trend analysis.")

            with tab_target:
                st.subheader("🎯 Target Variable Relationships")
                if report.target_analysis:
                    st.write(f"**Selected Target Variable:** `{report.target_analysis.get('target_column')}` ({report.target_analysis.get('target_type')})")
                    if "target_relationship" in report.charts_plotly_json:
                        fig_target = go.Figure(json.loads(report.charts_plotly_json["target_relationship"]))
                        st.plotly_chart(fig_target, use_container_width=True)

                    if "feature_correlations" in report.target_analysis:
                        st.markdown("##### Feature Correlations to Target")
                        t_corrs = pd.DataFrame(
                            [{"Feature": k, "Correlation to Target": v} for k, v in report.target_analysis["feature_correlations"].items()]
                        ).sort_values("Correlation to Target", key=abs, ascending=False)
                        st.dataframe(t_corrs, use_container_width=True)
                else:
                    st.info("Select a target column in the dropdown above and click 'Run Complete EDA Analysis' to analyze target relationships.")


elif nav_option == "Predictive Modeling & ML":
    st.markdown('<div class="main-header">Predictive Modeling & ML Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Classification & Regression, Preprocessing Integration, Cross-Validation, Tuning, Evaluation Reports, & Real-Time Inference</div>', unsafe_allow_html=True)

    ml_df = st.session_state.get("active_df", None)

    if ml_df is None or ml_df.empty:
        st.warning("⚠️ No active dataset found. Please upload a dataset in 'Data Ingestion & Validation' or upload a file below.")
        uploaded_file_ml = st.file_uploader("Upload CSV dataset for ML Training", type=["csv", "xlsx"])
        if uploaded_file_ml is not None:
            try:
                file_bytes = uploaded_file_ml.read()
                ml_df = DataIngestionService.ingest_file(file_bytes=file_bytes, filename=uploaded_file_ml.name)
                st.session_state["active_df"] = ml_df
                st.success(f"Dataset successfully loaded: {ml_df.shape[0]} rows × {ml_df.shape[1]} columns.")
            except Exception as exc:
                st.error(f"Error loading uploaded dataset: {exc}")

    if ml_df is not None and not ml_df.empty:
        st.markdown("### 1. Training Configuration & Setup")
        cols = list(ml_df.columns)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            target_col = st.selectbox(
                "Select Target Column (y):",
                options=cols,
                index=len(cols) - 1,
                help="The column your model will learn to predict.",
            )

            task_option = st.selectbox(
                "Select ML Task Type:",
                options=["Auto-Detect (Recommended)", "Classification", "Regression"],
                help="Choose explicit task or let DataSense AI auto-detect based on target characteristics.",
            )

            # Auto-detection banner
            if target_col:
                try:
                    inferred_task = determine_task_type(ml_df, target_col)
                    st.info(f"💡 Target Column **'{target_col}'** auto-detected as **{inferred_task.value.upper()}**")
                except Exception as e:
                    st.warning(f"Task auto-detection notice: {e}")

        with col_right:
            if task_option == "Classification":
                resolved_task_enum = TaskType.CLASSIFICATION
            elif task_option == "Regression":
                resolved_task_enum = TaskType.REGRESSION
            else:
                resolved_task_enum = determine_task_type(ml_df, target_col) if target_col else TaskType.CLASSIFICATION

            # Model Algorithm Selection Checkboxes
            st.markdown(f"**Select Models to Train ({resolved_task_enum.value.capitalize()}):**")
            if resolved_task_enum == TaskType.CLASSIFICATION:
                avail_models = {
                    "Logistic Regression": ClassificationAlgorithm.LOGISTIC_REGRESSION.value,
                    "Random Forest Classifier": ClassificationAlgorithm.RANDOM_FOREST.value,
                    "XGBoost Classifier": ClassificationAlgorithm.XGBOOST.value,
                    "Gradient Boosting Classifier": ClassificationAlgorithm.GRADIENT_BOOSTING.value,
                }
            else:
                avail_models = {
                    "Linear Regression": RegressionAlgorithm.LINEAR_REGRESSION.value,
                    "Random Forest Regressor": RegressionAlgorithm.RANDOM_FOREST.value,
                    "XGBoost Regressor": RegressionAlgorithm.XGBOOST.value,
                    "Gradient Boosting Regressor": RegressionAlgorithm.GRADIENT_BOOSTING.value,
                }

            selected_model_labels = st.multiselect(
                "Algorithms:",
                options=list(avail_models.keys()),
                default=list(avail_models.keys()),
            )
            selected_model_keys = [avail_models[lbl] for lbl in selected_model_labels]

        with st.expander("⚙️ Advanced Hyperparameters & Data Split Settings", expanded=False):
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                test_size_val = st.slider("Holdout Test Split Ratio:", min_value=0.10, max_value=0.30, value=0.15, step=0.05)
            with exp_col2:
                cv_folds_val = st.slider("Cross-Validation Folds (K):", min_value=2, max_value=10, value=5, step=1)
            with exp_col3:
                enable_tuning_val = st.checkbox("Enable Randomized Hyperparameter Tuning", value=False)

        st.divider()

        # Training Trigger Button
        if st.button("🚀 Train & Compare Machine Learning Models", type="primary", use_container_width=True):
            if not selected_model_keys:
                st.error("Please select at least one algorithm to train.")
            else:
                with st.spinner("Training models with leakage prevention, cross-validation, and metrics evaluation..."):
                    config = TrainingConfig(
                        target_column=target_col,
                        task_type=resolved_task_enum,
                        selected_models=selected_model_keys,
                        test_size=test_size_val,
                        val_size=test_size_val,
                        cross_validation_folds=cv_folds_val,
                        enable_tuning=enable_tuning_val,
                    )

                    try:
                        trainer = ModelTrainer(registry=LocalModelRegistry())
                        report: ModelComparisonReport = trainer.train(ml_df, config)
                        st.session_state["ml_report"] = report
                        st.success("✅ Model training and comparison completed successfully!")
                    except Exception as train_exc:
                        st.error(f"Model training failed: {train_exc}")

        # Display Comparison Report & Metrics
        if "ml_report" in st.session_state:
            report: ModelComparisonReport = st.session_state["ml_report"]
            st.divider()
            st.markdown("### 2. Model Comparison Report & Evaluation Metrics")

            # Best Model Hero Header
            st.success(
                f"🏆 **BEST PERFORMING MODEL:** `{report.best_model_name}` (ID: `{report.best_model_id}`) | "
                f"Selection Metric: `{report.selection_metric.upper()}`"
            )

            # Metadata metrics summary
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Total Rows", f"{report.total_samples:,}")
            sm2.metric("Train Split", f"{report.train_samples:,}")
            sm3.metric("Validation Split", f"{report.val_samples:,}")
            sm4.metric("Test Split", f"{report.test_samples:,}")

            st.write("")

            # Comparison Table Data Preparation
            table_rows = []
            for res in report.results:
                row_dict = {
                    "Best": "⭐ BEST" if res.is_best else "",
                    "Model Name": res.model_name,
                    "Train Time (s)": res.training_time_seconds,
                    "CV Score (Mean ± Std)": f"{res.cv_scores_mean:.4f} ± {res.cv_scores_std:.4f}" if res.cv_scores_mean is not None else "N/A",
                }

                if report.task_type == TaskType.CLASSIFICATION:
                    val_m = res.validation_metrics
                    test_m = res.test_metrics
                    row_dict.update({
                        "Val Accuracy": round(val_m.get("accuracy", 0), 4),
                        "Val Precision": round(val_m.get("precision", 0), 4),
                        "Val Recall": round(val_m.get("recall", 0), 4),
                        "Val F1": round(val_m.get("f1", 0), 4),
                        "Val ROC-AUC": round(val_m.get("roc_auc", 0), 4) if val_m.get("roc_auc") is not None else "N/A",
                        "Test Accuracy": round(test_m.get("accuracy", 0), 4),
                        "Test F1": round(test_m.get("f1", 0), 4),
                    })
                else:
                    val_m = res.validation_metrics
                    test_m = res.test_metrics
                    row_dict.update({
                        "Val MAE": round(val_m.get("mae", 0), 4),
                        "Val MSE": round(val_m.get("mse", 0), 4),
                        "Val RMSE": round(val_m.get("rmse", 0), 4),
                        "Val R²": round(val_m.get("r2", 0), 4),
                        "Val MAPE (%)": round(val_m.get("mape", 0), 2) if val_m.get("mape") is not None else "N/A",
                        "Test RMSE": round(test_m.get("rmse", 0), 4),
                        "Test R²": round(test_m.get("r2", 0), 4),
                    })

                table_rows.append(row_dict)

            comp_df = pd.DataFrame(table_rows)
            st.dataframe(comp_df, use_container_width=True)

            # Performance Bar Chart Visualization
            st.markdown("#### 📊 Model Performance Comparison Chart")
            if report.task_type == TaskType.CLASSIFICATION:
                fig_comp = px.bar(
                    comp_df,
                    x="Model Name",
                    y=["Val F1", "Test F1", "Val Accuracy"],
                    barmode="group",
                    title="Classification Metrics (Val F1 vs Test F1 vs Val Accuracy)",
                    color_discrete_sequence=["#0284C7", "#10B981", "#F59E0B"],
                )
            else:
                fig_comp = px.bar(
                    comp_df,
                    x="Model Name",
                    y=["Val RMSE", "Test RMSE"],
                    barmode="group",
                    title="Regression Metrics (Val RMSE vs Test RMSE - Lower is Better)",
                    color_discrete_sequence=["#EF4444", "#F59E0B"],
                )
            fig_comp.update_layout(height=380)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Confusion Matrix for Classification Models
            if report.task_type == TaskType.CLASSIFICATION:
                st.markdown("#### 🌀 Confusion Matrix (Validation Set)")
                cm_cols = st.columns(len(report.results))
                for idx, res in enumerate(report.results):
                    with cm_cols[idx]:
                        cm = res.validation_metrics.get("confusion_matrix", [])
                        if cm:
                            fig_cm = px.imshow(
                                np.array(cm),
                                text_auto=True,
                                color_continuous_scale="Blues",
                                title=f"{res.model_name}",
                                labels=dict(x="Predicted Class", y="Actual Class"),
                            )
                            fig_cm.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(fig_cm, use_container_width=True)

            st.divider()

        # 3. Real-Time Model Prediction Interface
        st.markdown("### 3. Real-Time Model Inference & Predictions")
        registry = LocalModelRegistry()
        saved_models = registry.list_models()

        if not saved_models:
            st.info("Train a model above to enable real-time predictions.")
        else:
            model_options = {f"{m['model_name']} ({m['model_id']}) {'⭐ BEST' if m.get('is_best') else ''}": m["model_id"] for m in saved_models}
            selected_model_label = st.selectbox("Select Model for Prediction:", options=list(model_options.keys()))
            selected_model_id = model_options[selected_model_label]

            pred_tab1, pred_tab2 = st.tabs(["Single Record Prediction Form", "Batch CSV Prediction"])

            with pred_tab1:
                meta = registry.get_model_metadata(selected_model_id)
                feature_cols = meta.get("feature_columns", []) if meta else []
                target_name = meta.get("target_column", "target") if meta else "target"

                st.markdown(f"Fill in feature values to predict target `{target_name}`:")

                with st.form("single_prediction_form"):
                    input_data = {}
                    # Build input fields dynamically from feature list or sample row
                    sample_row = ml_df.drop(columns=[target_name], errors="ignore").iloc[0] if ml_df is not None else {}
                    
                    form_cols = st.columns(3)
                    col_idx = 0

                    features_to_input = feature_cols if feature_cols else [c for c in ml_df.columns if c != target_name]
                    for feat in features_to_input[:15]: # Limit form inputs for clean UI
                        with form_cols[col_idx % 3]:
                            default_val = sample_row.get(feat, 0)
                            if isinstance(default_val, (int, float, np.integer, np.floating)) and not isinstance(default_val, bool):
                                input_data[feat] = st.number_input(f"{feat}", value=float(default_val))
                            else:
                                input_data[feat] = st.text_input(f"{feat}", value=str(default_val))
                        col_idx += 1

                    submit_pred = st.form_submit_button("🔮 Make Prediction", type="primary")

                if submit_pred:
                    try:
                        trainer = ModelTrainer(registry=registry)
                        df_pred = pd.DataFrame([input_data])
                        res_pred = trainer.predict(selected_model_id, df_pred)

                        st.success(f"**Predicted Value / Class:** `{res_pred.predictions[0]}`")
                        if res_pred.probabilities:
                            st.write("**Class Probabilities:**", res_pred.probabilities[0])
                    except Exception as pred_err:
                        st.error(f"Prediction failed: {pred_err}")

            with pred_tab2:
                uploaded_batch = st.file_uploader("Upload CSV file for batch predictions", type=["csv"])
                if uploaded_batch is not None:
                    try:
                        batch_df = pd.read_csv(uploaded_batch)
                        st.write(f"Loaded {len(batch_df)} rows for prediction preview:")
                        st.dataframe(batch_df.head(5), use_container_width=True)

                        if st.button("🔮 Run Batch Predictions", type="primary"):
                            trainer = ModelTrainer(registry=registry)
                            res_pred = trainer.predict(selected_model_id, batch_df)

                            out_df = batch_df.copy()
                            out_df[f"Predicted_{target_name}"] = res_pred.predictions
                            st.dataframe(out_df, use_container_width=True)
                            
                            # CSV download button
                            csv_data = out_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv_data,
                                file_name="datasense_predictions.csv",
                                mime="text/csv",
                            )
                    except Exception as batch_err:
                        st.error(f"Batch prediction error: {batch_err}")


elif nav_option == "Time-Series Forecasting":
    st.markdown('<div class="main-header">Time-Series Forecasting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Chronological Validation, Missing Date Resampling, Feature Engineering, Multi-Model Forecasting, & 95% Confidence Intervals</div>', unsafe_allow_html=True)

    fc_df = st.session_state.get("active_df", None)

    if fc_df is None or fc_df.empty:
        st.warning("⚠️ No active dataset found. Please upload a dataset in 'Data Ingestion & Validation' or upload a file below.")
        uploaded_file_fc = st.file_uploader("Upload CSV dataset for Time-Series Forecasting", type=["csv", "xlsx"], key="fc_uploader")
        if uploaded_file_fc is not None:
            try:
                file_bytes = uploaded_file_fc.read()
                fc_df = DataIngestionService.ingest_file(file_bytes=file_bytes, filename=uploaded_file_fc.name)
                st.session_state["active_df"] = fc_df
                st.success(f"Dataset loaded: {fc_df.shape[0]} rows × {fc_df.shape[1]} columns.")
            except Exception as exc:
                st.error(f"Error loading dataset: {exc}")

    if fc_df is not None and not fc_df.empty:
        cols = list(fc_df.columns)
        dt_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(fc_df[c]) or "date" in c.lower() or "time" in c.lower()]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(fc_df[c])]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date_col = st.selectbox("Date / Time Column:", options=dt_cols if dt_cols else cols, index=0)
        with col2:
            target_col = st.selectbox("Numerical Target Column:", options=num_cols if num_cols else cols, index=0)
        with col3:
            horizon_val = st.slider("Forecast Horizon (Steps):", min_value=1, max_value=180, value=30, step=1)
        with col4:
            freq_val = st.selectbox("Frequency Grid:", options=["auto", "D", "W", "M", "H"])

        selected_models_fc = st.multiselect(
            "Forecasting Models:",
            options=["baseline", "prophet", "xgboost"],
            default=["baseline", "prophet", "xgboost"],
            format_func=lambda x: {"baseline": "Baseline Exponential Smoothing", "prophet": "Prophet / Seasonal Decomposition", "xgboost": "XGBoost Autoregressive"}[x],
        )

        st.divider()

        if st.button("📈 Run Time-Series Forecasting Pipeline", type="primary", use_container_width=True):
            if not selected_models_fc:
                st.error("Please select at least one forecasting model.")
            else:
                with st.spinner("Executing time-series preprocessing, feature engineering, and model training..."):
                    try:
                        config = ForecastingConfig(
                            date_column=date_col,
                            target_column=target_col,
                            forecast_horizon=horizon_val,
                            frequency=freq_val,
                            selected_models=selected_models_fc,
                        )
                        engine = ForecastingEngine()
                        report: ForecastingReport = engine.run_forecasting(fc_df, config)
                        st.session_state["fc_report"] = report
                        st.success("✅ Time-series forecasting completed successfully!")
                    except Exception as fc_err:
                        st.error(f"Forecasting failed: {fc_err}")

        if "fc_report" in st.session_state:
            report: ForecastingReport = st.session_state["fc_report"]
            st.divider()

            st.success(f"🏆 **BEST FORECAST MODEL:** `{report.best_model_name}` | Frequency: `{report.inferred_frequency}` | Missing Dates Imputed: `{report.missing_dates_detected}`")

            # Metrics Table
            st.markdown("#### 📊 Forecast Model Metrics Comparison")
            metrics_rows = []
            for res in report.results:
                m = res.test_metrics
                metrics_rows.append({
                    "Best": "⭐ BEST" if res.is_best else "",
                    "Model Name": res.model_name,
                    "MAE": round(m.mae, 4),
                    "MSE": round(m.mse, 4),
                    "RMSE": round(m.rmse, 4),
                    "MAPE (%)": round(m.mape, 2) if m.mape is not None else "N/A",
                    "SMAPE (%)": round(m.smape, 2) if m.smape is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

            # Interactive Plotly Chart
            if report.chart_plotly_json:
                st.markdown("#### 📈 Historical vs Forecast Interactive Plotly Chart")
                fig_fc = go.Figure(json.loads(report.chart_plotly_json))
                st.plotly_chart(fig_fc, use_container_width=True)

            # Forecast Points Table
            st.markdown(f"#### 🔮 Future Forecast Table ({report.forecast_horizon} Steps)")
            best_res = next((r for r in report.results if r.is_best), report.results[0])
            fc_items = [
                {
                    "Timestamp": item.timestamp,
                    "Predicted Forecast": item.predicted_value,
                    "Lower Bound (95%)": item.lower_bound,
                    "Upper Bound (95%)": item.upper_bound,
                }
                for item in best_res.future_forecast
            ]
            fc_table_df = pd.DataFrame(fc_items)
            st.dataframe(fc_table_df, use_container_width=True)

            csv_fc = fc_table_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_fc,
                file_name="datasense_time_series_forecast.csv",
                mime="text/csv",
            )


elif nav_option == "Anomaly Detection":
    st.markdown('<div class="main-header">Automated Anomaly Detection Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Isolation Forest, Z-Score, IQR, & Ensemble Detection with Severity Scoring and Top Feature Attribution</div>', unsafe_allow_html=True)

    anom_df = st.session_state.get("active_df", None)

    if anom_df is None or anom_df.empty:
        st.warning("⚠️ No active dataset found. Please upload a dataset in 'Data Ingestion & Validation' or upload a file below.")
        uploaded_file_anom = st.file_uploader("Upload CSV dataset for Anomaly Detection", type=["csv", "xlsx"], key="anom_uploader")
        if uploaded_file_anom is not None:
            try:
                file_bytes = uploaded_file_anom.read()
                anom_df = DataIngestionService.ingest_file(file_bytes=file_bytes, filename=uploaded_file_anom.name)
                st.session_state["active_df"] = anom_df
                st.success(f"Dataset loaded: {anom_df.shape[0]} rows × {anom_df.shape[1]} columns.")
            except Exception as exc:
                st.error(f"Error loading dataset: {exc}")

    if anom_df is not None and not anom_df.empty:
        num_cols = [c for c in anom_df.columns if pd.api.types.is_numeric_dtype(anom_df[c])]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_features = st.multiselect("Numerical Features to Evaluate:", options=num_cols, default=num_cols)
        with col2:
            method_opt = st.selectbox(
                "Detection Algorithm Method:",
                options=["ensemble", "isolation_forest", "zscore", "iqr"],
                format_func=lambda x: {"ensemble": "Ensemble Voting (Recommended)", "isolation_forest": "Isolation Forest", "zscore": "Z-Score Statistical", "iqr": "Interquartile Range (IQR)"}[x],
            )
        with col3:
            contamination_val = st.slider("Contamination Rate:", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
        with col4:
            z_thresh_val = st.slider("Z-Score Threshold:", min_value=1.5, max_value=5.0, value=3.0, step=0.5)

        st.divider()

        if st.button("🔍 Run Automated Anomaly Detection", type="primary", use_container_width=True):
            if not selected_features:
                st.error("Please select at least one numerical feature.")
            else:
                with st.spinner("Detecting statistical and machine learning anomalies..."):
                    try:
                        config = AnomalyConfig(
                            features=selected_features,
                            method=AnomalyMethod(method_opt),
                            contamination=contamination_val,
                            z_threshold=z_thresh_val,
                        )
                        detector = AnomalyDetector()
                        report: AnomalyReport = detector.detect(anom_df, config)
                        st.session_state["anom_report"] = report
                        st.success("✅ Anomaly detection analysis completed successfully!")
                    except Exception as anom_err:
                        st.error(f"Anomaly detection failed: {anom_err}")

        if "anom_report" in st.session_state:
            report: AnomalyReport = st.session_state["anom_report"]
            st.divider()

            # Metric Header Cards
            st.markdown("#### 🚨 Anomaly Summary & Severity Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows Analyzed", f"{report.total_rows:,}")
            c2.metric("Anomalous Rows Count", f"{report.affected_rows_count:,}")
            c3.metric("Anomaly Rate (%)", f"{report.anomaly_percentage}%")
            
            sev_val = report.max_severity.value if hasattr(report.max_severity, "value") else str(report.max_severity)
            c4.metric("Highest Severity Detected", sev_val)

            st.write("")

            # Plotly Scatter Chart
            if report.chart_plotly_json:
                st.markdown("#### 📊 Interactive Anomaly Visual Scatter Plot")
                fig_anom = go.Figure(json.loads(report.chart_plotly_json))
                st.plotly_chart(fig_anom, use_container_width=True)

            # Feature Attribution Ranking
            if report.feature_importance_ranking:
                st.markdown("#### 💡 Top Feature Deviations (Anomaly Importance)")
                feat_df = pd.DataFrame([{"Feature": k, "Avg Z-Deviation": v} for k, v in report.feature_importance_ranking.items()])
                fig_feat = px.bar(feat_df, x="Avg Z-Deviation", y="Feature", orientation="h", title="Top Features Contributing to Anomalies")
                fig_feat.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_feat, use_container_width=True)

            # Anomalous Records Table
            st.markdown(f"#### 📋 Anomalous Records Detail ({len(report.anomalous_records)} rows)")
            if report.anomalous_records:
                anom_rows = []
                for rec in report.anomalous_records:
                    contrib_str = ", ".join([f"{k} (|Z|={v})" for k, v in rec.contributing_features.items()])
                    anom_rows.append({
                        "Row Index": rec.row_index,
                        "Anomaly Score": rec.anomaly_score,
                        "Severity": rec.severity.value if hasattr(rec.severity, "value") else str(rec.severity),
                        "Top Contributing Features": contrib_str,
                        "Feature Snippet": str(rec.feature_values),
                    })
                anom_table_df = pd.DataFrame(anom_rows)
                st.dataframe(anom_table_df, use_container_width=True)

                csv_anom = anom_table_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Anomalous Records CSV",
                    data=csv_anom,
                    file_name="datasense_anomalies.csv",
                    mime="text/csv",
                )


elif nav_option == "Business Intelligence & Analytics":
    st.markdown('<div class="main-header">Business Intelligence & Customer Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive KPIs, RFM Persona Segmentation, Automated K-Means & Hierarchical Clustering Evaluation (Silhouette & Davies-Bouldin), Churn Risk, & Lifetime Value (CLV)</div>', unsafe_allow_html=True)

    bi_df = st.session_state.get("active_df", None)

    if bi_df is None or bi_df.empty:
        st.warning("⚠️ No active dataset found. Please upload a dataset in 'Data Ingestion & Validation' or upload a file below.")
        uploaded_file_bi = st.file_uploader("Upload CSV dataset for Business Intelligence Analysis", type=["csv", "xlsx"], key="bi_uploader")
        if uploaded_file_bi is not None:
            try:
                file_bytes = uploaded_file_bi.read()
                bi_df = DataIngestionService.ingest_file(file_bytes=file_bytes, filename=uploaded_file_bi.name)
                st.session_state["active_df"] = bi_df
                st.success(f"Dataset loaded: {bi_df.shape[0]} rows × {bi_df.shape[1]} columns.")
            except Exception as exc:
                st.error(f"Error loading dataset: {exc}")

    if bi_df is not None and not bi_df.empty:
        cols = ["<Auto-Detect>"] + list(bi_df.columns)

        with st.expander("⚙️ Configurable Column Mapping & Algorithm Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                c_id = st.selectbox("Customer ID Column:", options=cols, index=0)
                c_date = st.selectbox("Order Date Column:", options=cols, index=0)
            with col2:
                c_rev = st.selectbox("Revenue / Sales Column:", options=cols, index=0)
                c_prof = st.selectbox("Profit Column:", options=cols, index=0)
            with col3:
                c_qty = st.selectbox("Quantity Column:", options=cols, index=0)
                c_churn = st.selectbox("Churn Target Column:", options=cols, index=0)

            alg_opt = st.radio("Customer Clustering Algorithm:", ["kmeans", "hierarchical"], format_func=lambda x: "K-Means Clustering" if x == "kmeans" else "Hierarchical Agglomerative Clustering", horizontal=True)

        manual_map = ColumnMappingConfig(
            customer_id_col=None if c_id == "<Auto-Detect>" else c_id,
            order_date_col=None if c_date == "<Auto-Detect>" else c_date,
            revenue_col=None if c_rev == "<Auto-Detect>" else c_rev,
            profit_col=None if c_prof == "<Auto-Detect>" else c_prof,
            quantity_col=None if c_qty == "<Auto-Detect>" else c_qty,
            churn_col=None if c_churn == "<Auto-Detect>" else c_churn,
        )

        st.divider()

        if st.button("🚀 Run Business Intelligence Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing business KPIs, RFM customer segments, cluster metrics, churn risk, and CLV..."):
                try:
                    engine = BIEngine()
                    report: BIAnalysisReport = engine.analyze(
                        df=bi_df,
                        manual_mapping=manual_map,
                        clustering_algorithm=alg_opt,
                    )
                    st.session_state["bi_report"] = report
                    st.success("✅ Business Intelligence analysis completed successfully!")
                except Exception as bi_err:
                    st.error(f"BI Analysis failed: {bi_err}")

        if "bi_report" in st.session_state:
            report: BIAnalysisReport = st.session_state["bi_report"]
            kpis = report.business_kpis
            st.divider()

            # Resolved Mapping Badge
            resolved = report.resolved_mapping
            st.info(f"🔍 **Resolved Mapping**: Customer ID: `{resolved.customer_id_col or 'N/A'}` | Date: `{resolved.order_date_col or 'N/A'}` | Revenue: `{resolved.revenue_col or 'N/A'}` | Profit: `{resolved.profit_col or 'N/A'}`")

            # Executive KPI Cards
            st.markdown("#### 📈 Executive Business KPIs")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Revenue", f"${kpis.total_revenue:,.2f}")
            c2.metric("Total Profit", f"${kpis.total_profit:,.2f}" if kpis.total_profit is not None else "N/A", delta=f"{kpis.profit_margin_pct}% Margin" if kpis.profit_margin_pct else None)
            c3.metric("Average Order Value (AOV)", f"${kpis.average_order_value:,.2f}")
            c4.metric("Total Customers", f"{kpis.total_customers:,}")
            c5.metric("Repeat Purchase Rate", f"{kpis.repeat_purchase_rate}%")

            # Business Insights List
            if report.business_insights:
                with st.expander("💡 Actionable Executive Business Insights", expanded=True):
                    for ins in report.business_insights:
                        st.markdown(f"- {ins}")

            # Cluster Evaluation Card
            if report.cluster_evaluation:
                st.markdown("#### 🧪 Cluster Evaluation & Segmentation Quality")
                ce = report.cluster_evaluation
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Algorithm", ce.algorithm)
                m2.metric("Optimal Clusters (K)", ce.optimal_k)
                m3.metric("Silhouette Score (higher=better)", ce.silhouette_score)
                m4.metric("Davies-Bouldin Index (lower=better)", ce.davies_bouldin_index)

            # Analysis Tabs
            tab1, tab2, tab3 = st.tabs(["👥 RFM & Customer Segmentation", "📊 Revenue & Profit Analytics", "🚨 Churn & Lifetime Value (CLV)"])

            with tab1:
                if "customer_segments_scatter" in report.charts_plotly_json:
                    fig_sc = go.Figure(json.loads(report.charts_plotly_json["customer_segments_scatter"]))
                    st.plotly_chart(fig_sc, use_container_width=True)

                if report.rfm_segments:
                    st.markdown("##### RFM Persona Segment Summary Table")
                    rfm_rows = [
                        {
                            "Segment Persona": s.segment_name,
                            "Customer Count": s.customer_count,
                            "Avg Recency (Days)": s.avg_recency_days,
                            "Avg Frequency (Orders)": s.avg_frequency,
                            "Avg Spend ($)": f"${s.avg_monetary_value:,.2f}",
                            "Revenue Contribution (%)": f"{s.total_revenue_share_pct}%",
                        }
                        for s in report.rfm_segments
                    ]
                    st.dataframe(pd.DataFrame(rfm_rows), use_container_width=True)

                if "rfm_distribution_bar" in report.charts_plotly_json:
                    fig_dist = go.Figure(json.loads(report.charts_plotly_json["rfm_distribution_bar"]))
                    st.plotly_chart(fig_dist, use_container_width=True)

            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    if "revenue_by_segment_pie" in report.charts_plotly_json:
                        fig_rev = go.Figure(json.loads(report.charts_plotly_json["revenue_by_segment_pie"]))
                        st.plotly_chart(fig_rev, use_container_width=True)
                with col_b:
                    if "profit_by_segment_bar" in report.charts_plotly_json:
                        fig_prof = go.Figure(json.loads(report.charts_plotly_json["profit_by_segment_bar"]))
                        st.plotly_chart(fig_prof, use_container_width=True)

                if "customer_trends_line" in report.charts_plotly_json:
                    fig_trend = go.Figure(json.loads(report.charts_plotly_json["customer_trends_line"]))
                    st.plotly_chart(fig_trend, use_container_width=True)

            with tab3:
                if report.churn_summary:
                    st.markdown("##### 🚨 Customer Churn Risk Analysis")
                    ch = report.churn_summary
                    ch1, ch2 = st.columns(2)
                    ch1.metric("Overall Churn Rate", f"{ch.overall_churn_rate_pct}%")
                    ch2.metric("High Churn Risk Customers (>70%)", f"{ch.high_risk_customer_count:,}")

                    if ch.top_churn_drivers:
                        fig_ch = px.bar(
                            x=list(ch.top_churn_drivers.keys()),
                            y=list(ch.top_churn_drivers.values()),
                            labels={"x": "Feature Driver", "y": "Feature Importance"},
                            title="Top Feature Drivers of Customer Churn",
                            template="plotly_white",
                        )
                        fig_ch.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_ch, use_container_width=True)

                if report.clv_summary:
                    st.markdown("##### 💎 Customer Lifetime Value (CLV) Estimation")
                    cl = report.clv_summary
                    cl1, cl2, cl3 = st.columns(3)
                    cl1.metric("Average Historical CLV", f"${cl.average_historical_clv:,.2f}")
                    cl2.metric("Projected 12-Month CLV", f"${cl.average_projected_12m_clv:,.2f}")
                    cl3.metric("Top CLV Segment", cl.top_clv_segment)


elif nav_option == "Model Explainability & SHAP":
    st.markdown('<div class="main-header">Model Explainability & SHAP Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Global Feature Importance, Local Per-Instance Prediction Attributions, & Top Positive/Negative Drivers</div>', unsafe_allow_html=True)

    xai_df = st.session_state.get("active_df", None)

    if xai_df is None or xai_df.empty:
        st.warning("⚠️ No active dataset found. Please upload a dataset in 'Data Ingestion & Validation' or upload a file below.")
        uploaded_file_xai = st.file_uploader("Upload CSV dataset for Model Explainability", type=["csv", "xlsx"], key="xai_uploader")
        if uploaded_file_xai is not None:
            try:
                file_bytes = uploaded_file_xai.read()
                xai_df = DataIngestionService.ingest_file(file_bytes=file_bytes, filename=uploaded_file_xai.name)
                st.session_state["active_df"] = xai_df
                st.success(f"Dataset loaded: {xai_df.shape[0]} rows × {xai_df.shape[1]} columns.")
            except Exception as exc:
                st.error(f"Error loading dataset: {exc}")

    if xai_df is not None and not xai_df.empty:
        num_cols = [c for c in xai_df.columns if pd.api.types.is_numeric_dtype(xai_df[c])]

        col1, col2 = st.columns(2)
        with col1:
            target_col_xai = st.selectbox("Target Column for Prediction Explanation:", options=num_cols if num_cols else list(xai_df.columns), index=0)
        with col2:
            row_idx_xai = st.slider("Select Instance Row Index for Local Explanation:", min_value=0, max_value=max(0, len(xai_df) - 1), value=0)

        st.divider()

        if st.button("🧠 Compute SHAP Model Explanations", type="primary", use_container_width=True):
            with st.spinner("Fitting surrogate model and computing SHAP values..."):
                try:
                    num_feats = [c for c in num_cols if c != target_col_xai]
                    X_num = xai_df[num_feats].fillna(xai_df[num_feats].median()).to_numpy()
                    y_num = xai_df[target_col_xai].to_numpy()

                    task_type = determine_task_type(xai_df[target_col_xai])
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
                        instance_indices=[row_idx_xai, min(1, len(xai_df)-1)],
                    )
                    st.session_state["xai_report"] = report
                    st.success("✅ SHAP model explanations computed successfully!")
                except Exception as xai_err:
                    st.error(f"SHAP explanation failed: {xai_err}")

        if "xai_report" in st.session_state:
            report: XAIReport = st.session_state["xai_report"]
            st.divider()

            # Global Feature Importance Chart
            if report.summary_chart_plotly_json:
                st.markdown("#### 🌐 Global Feature Importance (Mean |SHAP Value|)")
                fig_glob = go.Figure(json.loads(report.summary_chart_plotly_json))
                st.plotly_chart(fig_glob, use_container_width=True)

            # Local Explanation for selected instance
            if report.sample_local_explanations:
                local_exp = report.sample_local_explanations[0]
                st.markdown(f"#### 🔍 Local Instance Explanation (Row #{local_exp.instance_index})")

                c1, c2 = st.columns(2)
                c1.metric("Base Value E[f(X)]", local_exp.base_value)
                c2.metric("Prediction Value f(X_i)", local_exp.prediction_value)

                st.write("")

                pos_col, neg_col = st.columns(2)
                with pos_col:
                    st.markdown("##### 🟢 Top Positive Drivers (+ Increases Output)")
                    if local_exp.top_positive_features:
                        pos_df = pd.DataFrame([f.model_dump() for f in local_exp.top_positive_features])
                        st.dataframe(pos_df[["feature_name", "feature_value", "shap_value"]], use_container_width=True)
                    else:
                        st.info("No positive contributing features for this instance.")

                with neg_col:
                    st.markdown("##### 🔴 Top Negative Drivers (- Decreases Output)")
                    if local_exp.top_negative_features:
                        neg_df = pd.DataFrame([f.model_dump() for f in local_exp.top_negative_features])
                        st.dataframe(neg_df[["feature_name", "feature_value", "shap_value"]], use_container_width=True)
                    else:
                        st.info("No negative contributing features for this instance.")

                st.markdown("##### 📋 Complete SHAP Feature Contributions")
                all_df = pd.DataFrame([f.model_dump() for f in local_exp.all_contributions])
                st.dataframe(all_df, use_container_width=True)


elif nav_option == "Business Recommendations Engine":
    st.markdown('<div class="main-header">Business Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Synthesizing EDA, Predictive ML Models, Time-Series Forecasts, Anomaly Detection, BI Analytics, & SHAP into Data-Grounded Executive Actions</div>', unsafe_allow_html=True)

    rec_df = st.session_state.get("active_df", None)

    if st.button("💡 Synthesize & Generate Business Recommendations", type="primary", use_container_width=True):
        with st.spinner("Synthesizing multi-module analytics and computing evidence-backed recommendations..."):
            try:
                engine = BusinessRecommendationEngine()
                eda_rep = st.session_state.get("eda_report", {}).model_dump() if hasattr(st.session_state.get("eda_report", None), "model_dump") else None
                ml_rep = st.session_state.get("ml_report", {}).model_dump() if hasattr(st.session_state.get("ml_report", None), "model_dump") else None
                fc_rep = st.session_state.get("fc_report", {}).model_dump() if hasattr(st.session_state.get("fc_report", None), "model_dump") else None
                anom_rep = st.session_state.get("anom_report", {}).model_dump() if hasattr(st.session_state.get("anom_report", None), "model_dump") else None
                bi_rep = st.session_state.get("bi_report", {}).model_dump() if hasattr(st.session_state.get("bi_report", None), "model_dump") else None
                xai_rep = st.session_state.get("xai_report", {}).model_dump() if hasattr(st.session_state.get("xai_report", None), "model_dump") else None

                report: RecommendationReport = engine.generate_recommendations(
                    df=rec_df,
                    eda_report=eda_rep,
                    ml_report=ml_rep,
                    forecast_report=fc_rep,
                    anomaly_report=anom_rep,
                    bi_report=bi_rep,
                    xai_report=xai_rep,
                )
                st.session_state["recommendation_report"] = report
                st.success("✅ Business recommendations generated successfully!")
            except Exception as rec_err:
                st.error(f"Recommendation generation failed: {rec_err}")

    if "recommendation_report" in st.session_state:
        report: RecommendationReport = st.session_state["recommendation_report"]
        st.divider()

        # Priority Metric Breakdown
        st.markdown("#### 🚨 Recommendation Priority Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Recommendations", report.total_recommendations)
        c2.metric("Critical Priority", report.critical_count)
        c3.metric("High Priority", report.high_count)
        c4.metric("Medium/Low Priority", report.total_recommendations - report.critical_count - report.high_count)

        st.divider()

        # Recommendation Feed Cards
        st.markdown("#### 📋 Actionable Executive Recommendations Feed")
        for item in report.items:
            priority_val = item.severity_priority.value if hasattr(item.severity_priority, "value") else str(item.severity_priority)
            badge_color = {
                "Critical": "#EF4444",
                "High": "#F97316",
                "Medium": "#F59E0B",
                "Low": "#3B82F6",
            }.get(priority_val, "#6B7280")

            with st.container():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 16px; background-color: #F8FAFC;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #1E293B;">{item.title}</h4>
                            <div>
                                <span style="background-color: {badge_color}; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 12px;">{priority_val} Priority</span>
                                <span style="background-color: #64748B; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; margin-left: 6px;">{item.source_module}</span>
                            </div>
                        </div>
                        <p style="margin-top: 8px; color: #475569;">{item.explanation}</p>
                        <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 10px; margin: 10px 0; border-radius: 4px;">
                            <strong>💡 Empirical Data Evidence:</strong> {item.evidence}
                        </div>
                        <div style="display: flex; gap: 16px; font-size: 13px; color: #334155;">
                            <div><strong>🎯 Affected Metric:</strong> <code>{item.affected_metric}</code></div>
                        </div>
                        <div style="margin-top: 10px; background-color: #F0FDF4; border-left: 4px solid #22C55E; padding: 10px; border-radius: 4px;">
                            <strong>✅ Suggested Action:</strong> {item.suggested_action}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Download CSV
        rec_rows = [item.model_dump() for item in report.items]
        rec_df_out = pd.DataFrame(rec_rows)
        csv_rec = rec_df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Recommendations CSV",
            data=csv_rec,
            file_name="datasense_business_recommendations.csv",
            mime="text/csv",
        )


elif nav_option == "System Diagnostics":
    st.header("System Diagnostics & Backend API Health")
    if health_data:
        st.json(health_data)
    else:
        st.warning("Backend API is currently offline. Running in local Streamlit mode.")



