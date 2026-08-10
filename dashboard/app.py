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
from datasense.data_processing.schemas import ValidationReport

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
        "Platform Overview",
        "Exploratory Data Analysis",
        "Predictive Modeling & ML",
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

    else:
        st.info("💡 Please upload a CSV or Excel file, or query a database table above to view data quality results.")

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
    st.header("Exploratory Data Analysis (EDA)")
    st.info("Automated EDA module ready. Switch to Data Ingestion tab to upload datasets for exploratory analysis.")

elif nav_option == "Predictive Modeling & ML":
    st.header("Machine Learning & Model Training")
    st.info("Machine Learning pipelines (XGBoost, Scikit-Learn, SHAP, MLflow) initialized.")

elif nav_option == "System Diagnostics":
    st.header("System Diagnostics & Backend API Health")
    if health_data:
        st.json(health_data)
    else:
        st.warning("Backend API is currently offline. Running in local Streamlit mode.")
