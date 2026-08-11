# 📊 DataSense AI: Intelligent Business Intelligence & Predictive Analytics Platform

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-v3.8-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**DataSense AI** is an enterprise-grade, end-to-end Automated Machine Learning (AutoML), Time-Series Forecasting, Anomaly Detection, Customer Segmentation, Explainable AI (SHAP), and Business Recommendation Platform built with **Python**, **FastAPI**, **Streamlit**, **PostgreSQL**, and **Docker**.

---

## 🎯 Problem Statement & Business Solution

Modern business organizations generate vast volumes of operational, financial, and customer data. However, extracting actionable insights usually requires combining fragmented tools: data cleaning scripts, statistical EDA dashboards, machine learning model trainers, time-series forecasters, anomaly auditors, and customer RFM analytics engines.

**DataSense AI** unifies these decoupled analytics workflows into a single, cohesive, data-grounded decision engine:
1. **Automates Data Cleaning & Quality Validation**: Ingests messy datasets, profiles missingness, removes duplicates, and caps statistical outliers.
2. **Trains & Compares Ensemble ML Models**: Automatically determines classification vs. regression tasks, performs train/validation/test splits, tunes hyperparameters, and evaluates accuracy, F1, RMSE, and R².
3. **Projects Future Demand Trends**: Fits Exponential Smoothing, Prophet, and XGBoost time-series models to predict future revenue and demand.
4. **Audits Risk & Operational Anomalies**: Uses Isolation Forest, Z-Score, IQR, and Ensemble voting to flag high-severity operational outliers.
5. **Segment Customers & Predict Churn**: Conducts RFM scoring, K-Means & Hierarchical Agglomerative clustering, evaluates Silhouette scores, and calculates 12-month Customer Lifetime Value (CLV).
6. **Explains Predictions via SHAP**: Computes global feature importances and local per-instance feature contributions.
7. **Generates Data-Grounded Executive Recommendations**: Synthesizes multi-engine analytics findings into actionable, evidence-backed business recommendations with zero hallucinations.

---

## 🏗️ System Architecture & Workflow

```
                               ┌────────────────────────────────────────┐
                               │   Streamlit Executive Dashboard (UI)   │
                               └───────────────────┬────────────────────┘
                                                   │ (REST API / Fallback)
                               ┌───────────────────▼────────────────────┐
                               │     FastAPI Backend Service (API)      │
                               └─────────┬────────────────────┬─────────┘
                                         │                    │
                ┌────────────────────────┴───────┐   ┌────────┴───────────────────────┐
                │   Core Analytics Engine Suite  │   │  Persistence & Model Tracking  │
                ├────────────────────────────────┤   ├────────────────────────────────┤
                │ 1. Data Validator & Cleaner    │   │ • PostgreSQL / SQLite Database │
                │ 2. Exploratory Data Analysis   │   │ • Local Model Registry         │
                │ 3. ML Trainer (RF/XGB/GB)      │   │ • MLflow Experiment Tracking   │
                │ 4. Time-Series Forecaster      │   └────────────────────────────────┘
                │ 5. Anomaly Detector            │
                │ 6. BI & Customer Segmentation  │
                │ 7. SHAP Explainable AI (XAI)   │
                │ 8. Recommendation Synthesizer  │
                └────────────────────────────────┘
```

---

## 🚀 Key Modules & Features

| # | Module Section | Key Capability & Business Output |
|---|---|---|
| **1** | **Home / Executive Overview** | Executive summary cards for Dataset Size, Quality Index (0–100%), Revenue, Anomaly Alerts, Best ML Model, Forecast Trends, and High-Priority Recommendations. |
| **2** | **Dataset Upload** | Multipart CSV/Excel file uploader, database SQL query loader, and pre-loaded synthetic e-commerce benchmark dataset picker. |
| **3** | **Data Quality** | Data Quality Index score gauge, warning flags, missingness profiling, duplicate row tracking, and column data type classification. |
| **4** | **Data Cleaning** | Automated Median/Mean/Mode imputation, IQR/Winsorization outlier capping, duplicate filtering, and constant column removal. |
| **5** | **Exploratory Data Analysis** | Summary stats (`describe`), Plotly correlation matrix heatmap, boxplot distributions, time trends, and automated insight generation. |
| **6** | **Machine Learning** | Auto task type detection (Classification/Regression), Logistic/Linear Regression, Random Forest, XGBoost, Gradient Boosting, hyperparameter tuning, holdout test metrics, and ML model leaderboard. |
| **7** | **Time-Series Forecasting** | Chronological train/test split, Exponential Smoothing, Prophet, XGBoost time-series modeling, forecast horizon projections ($H$), sMAPE/MAPE evaluation, and Plotly confidence intervals. |
| **8** | **Anomaly Detection** | Isolation Forest, Z-Score, IQR, and Ensemble voting anomaly detection with severity scoring and affected row audits. |
| **9** | **Customer Segmentation** | RFM scoring (Recency, Frequency, Monetary), K-Means & Hierarchical Agglomerative clustering, Silhouette & Davies-Bouldin evaluation, 12-Month CLV, and Churn Risk prediction. |
| **10** | **Explainable AI (SHAP)** | Global SHAP feature importances, local instance prediction explainer slider, top positive ($+$) vs. negative ($-$) feature contribution tables, and Plotly summary charts. |
| **11** | **Business Recommendations** | Actionable executive recommendations feed with priority badges (`Critical`, `High`, `Medium`, `Low`), empirical computed data evidence, affected business metrics, suggested action checklists, and CSV export. |
| **12** | **Model History** | Persistent SQLite/PostgreSQL `LocalModelRegistry` database table with metrics, features, algorithm names, best model badge, and real-time interactive model inference prediction sandbox. |

---

## 🛠️ Technology Stack

- **Frontend Interface**: Streamlit, Plotly Express, Plotly Graph Objects, HTML/CSS Glassmorphism UI.
- **Backend API Layer**: FastAPI, Pydantic v2, Starlette, Uvicorn, Gunicorn.
- **Data Engineering & Machine Learning**: Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels, Prophet, SHAP.
- **Database & Model Persistence**: PostgreSQL 16, SQLite, SQLAlchemy 2.0, Joblib, MLflow.
- **Diagnostics & Infrastructure**: Psutil, Docker, Docker Compose, Pytest.

---

## 💻 Installation & Quick Start

### Option A: Running with Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdulrahmanrifayath/DataSenseAI.git
   cd DataSenseAI
   ```

2. **Launch the multi-container stack**:
   ```bash
   docker-compose up --build
   ```

3. **Access Services**:
   - **Streamlit Executive Dashboard**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI REST API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **MLflow Tracking Server**: [http://localhost:5000](http://localhost:5000)

---

### Option B: Local Python Environment Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r pyproject.toml
   # Or install via pip:
   pip install fastapi uvicorn streamlit pandas numpy scikit-learn xgboost statsmodels prophet shap plotly psutil sqlalchemy joblib httpx requests pytest
   ```

3. **Set up Environment Variables**:
   ```bash
   cp .env.example .env
   ```

4. **Launch the FastAPI Backend API Server**:
   ```bash
   uvicorn datasense.api.main:app --reload --port 8000
   ```

5. **In a separate terminal, launch the Streamlit Dashboard**:
   ```bash
   streamlit run dashboard/app.py --server.port 8501
   ```

---

## 📡 REST API Documentation

FastAPI automatically generates interactive OpenAPI documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Primary Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` / `/api/v1/health` | System health check, CPU/RAM/Disk metrics, DB state, dataset/model counts. |
| `POST` | `/api/v1/datasets/upload` | Upload CSV/Excel dataset file. |
| `POST` | `/api/v1/preprocessing/clean` | Preprocess dataset with custom cleaning rules. |
| `POST` | `/api/v1/eda/analyze` | Generate full exploratory data analysis report. |
| `POST` | `/api/v1/ml/train` | Train and compare classification/regression ensemble models. |
| `POST` | `/api/v1/ml/predict` | Real-time model inference prediction sandbox. |
| `POST` | `/api/v1/forecasting/predict` | Generate time-series demand forecasts and confidence intervals. |
| `POST` | `/api/v1/anomaly/detect` | Execute anomaly detection algorithms (Isolation Forest, Z-Score, Ensemble). |
| `POST` | `/api/v1/bi/analyze` | Execute RFM customer analytics, clustering, CLV, and churn predictions. |
| `POST` | `/api/v1/xai/explain` | Compute global and local SHAP model feature attributions. |
| `POST` | `/api/v1/recommendations/generate` | Synthesize multi-engine findings into actionable business recommendations. |

---

## 🧪 Automated Testing Suite

The repository contains comprehensive unit, integration, API, and end-to-end test suites (`pytest`).

Run the full test suite:
```bash
pytest
```

**Test Coverage Summary**:
```bash
====================== 70 passed in 55.02s =======================
- Unit Tests (Preprocessors, EDA, ML, Forecasting, Anomaly, BI, SHAP): 35 passed
- FastAPI Endpoint Tests (Health, Datasets, Preprocessing, EDA, ML, Forecast, Anomaly, BI, XAI, Recs): 25 passed
- End-to-End Workflow & System Diagnostics Tests: 10 passed
```

---

## 📁 Repository & Project Structure

```
DataSenseAI/
├── .env.example                    # Environment variables template
├── Dockerfile.backend              # Docker container definition for FastAPI API
├── Dockerfile.frontend             # Docker container definition for Streamlit UI
├── docker-compose.yml              # Multi-container orchestration (API, UI, Postgres, MLflow)
├── pyproject.toml                  # Python package configuration & dependencies
├── README.md                       # Complete platform documentation
├── configuration/                  # Global application settings & configuration
│   └── settings.py
├── data/                           # Data storage & benchmark datasets
│   └── sample_ecommerce_data.csv
├── dashboard/                      # Streamlit Executive Dashboard UI
│   ├── api_client.py               # HTTP client connecting Streamlit to FastAPI
│   └── app.py                      # 12-Section Streamlit dashboard app
├── src/datasense/                  # Core DataSense AI Application Source
│   ├── anomaly_detection/          # Isolation Forest, Z-Score, IQR, Ensemble detector
│   ├── api/                        # FastAPI REST API main app & endpoint routers
│   ├── bi/                         # Customer RFM, K-Means, CLV, & Churn BI engine
│   ├── data_processing/            # Ingestion, DataValidator, & DataPreprocessor
│   ├── database/                   # SQLAlchemy ORM models & database connection
│   ├── eda/                        # Summary stats, correlation, & insight engine
│   ├── forecasting/                # Exponential Smoothing, Prophet, & XGBoost forecaster
│   ├── ml_models/                  # Trainer, ModelRegistry, MLflow tracking, & algorithms
│   ├── recommendations/            # Multi-module Recommendation Synthesizer engine
│   ├── utilities/                  # Structured logger & utility helpers
│   └── xai/                        # SHAP Explainable AI service & schemas
└── tests/                          # Automated pytest unit & integration test suite
```

---

## 🔮 Future Enhancements Roadmap

- [ ] **MLflow Model Registry Remote Server Integration**: Expand artifact registration to remote AWS S3 / Azure Blob Storage.
- [ ] **LLM Executive Summary Insights**: Integrate OpenAI/Claude API for enhanced natural language executive briefs.
- [ ] **Real-Time Data Stream Ingestion**: Support Apache Kafka / WebSocket data stream anomaly detection.
- [ ] **Automated SQL Analytics Query Generator**: Natural language to SQL query conversion for database extraction.

---

## 📜 License

This project is open-source software licensed under the **MIT License**.
