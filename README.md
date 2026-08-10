# DataSense AI – Intelligent Business Intelligence & Predictive Analytics Platform

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

DataSense AI is an end-to-end production-ready data science platform designed for enterprise predictive analytics, automated dataset validation, exploratory data analysis, machine learning forecasting, anomaly detection, cohort segmentation, explainable AI (SHAP), and actionable business recommendations.

---

## 🏗️ Architecture & Project Structure

```
DataSenseAI/
├── .env                    # Local environment variables
├── .env.example            # Template for environment settings
├── .gitignore              # Git exclusions file
├── Dockerfile              # Production multi-stage Docker build
├── docker-compose.yml      # Orchestration for PostgreSQL, FastAPI & Streamlit
├── pyproject.toml          # Package configuration and pytest setup
├── README.md               # Documentation and execution guide
├── requirements.txt        # Production dependency specifications
├── configuration/          # Configuration and settings management
│   ├── settings.py         # Pydantic BaseSettings class
│   └── logging_config.py   # Structured logging configuration
├── src/
│   └── datasense/          # Core package namespace
│       ├── api/            # FastAPI REST API services
│       │   ├── main.py     # FastAPI application entry point
│       │   └── routers/    # Modular API routers (Health, Diagnostics)
│       ├── database/       # PostgreSQL connection & ORM models
│       │   ├── connection.py
│       │   └── models.py
│       ├── data_processing/# Validation and preprocessing pipelines
│       ├── eda/            # Automated exploratory data analysis
│       ├── ml_models/      # Scikit-learn & XGBoost model training
│       ├── forecasting/    # Time-series trend and demand forecasting
│       ├── anomaly_detection/# Outlier & statistical anomaly detection
│       ├── segmentation/   # Customer cohort clustering (K-Means/RFM)
│       ├── explainability/ # Model explanation engine (SHAP)
│       ├── recommendations/# Business insight & recommendation generator
│       └── utilities/      # System loggers and shared utilities
├── dashboard/
│   └── app.py              # Streamlit interactive analytics UI
└── tests/                  # Automated pytest unit & integration test suite
```

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings |
| **Database** | PostgreSQL 16, SQLAlchemy 2.0, Psycopg2 |
| **Data Engine** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Explainable AI** | SHAP |
| **Visualization** | Plotly Express |
| **Dashboard UI** | Streamlit |
| **MLOps / Tracking** | MLflow |
| **Testing** | Pytest, Pytest-Asyncio, HTTpx |
| **Containerization**| Docker, Docker Compose |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12 installed
- PostgreSQL installed and running (or run via Docker Compose)

---

### 2. Creating Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Installing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

### 4. Configuration

Copy `.env.example` to create your local `.env` file:
```bash
cp .env.example .env
```

Ensure environment settings (database host, credentials, ports) match your local setup.

---

### 5. Running the Application Services

#### Option A: Running Local Services (FastAPI + Streamlit)

1. **Start FastAPI Backend:**
```bash
uvicorn datasense.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Interactive OpenAPI Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check Endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

2. **Start Streamlit Dashboard (In a separate terminal):**
```bash
streamlit run dashboard/app.py --server.port 8501
```
- Interactive Dashboard UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

#### Option B: Running with Docker Compose (Full Stack)

To launch PostgreSQL, FastAPI Backend, and Streamlit Dashboard simultaneously in containers:

```bash
docker-compose up --build -d
```

Check status:
```bash
docker-compose ps
```

Stop services:
```bash
docker-compose down
```

---

### 6. Running Tests

Run the pytest automated test suite:

```bash
pytest
```

Run with detailed verbose output and coverage:
```bash
pytest -v --tb=short
```

---

## 🧪 API Verification & Health Check Example

Invocations to `/health` return detailed platform and database status:

```json
{
  "status": "healthy",
  "app_name": "DataSense AI",
  "version": "v1",
  "environment": "development",
  "timestamp": "2026-08-10T22:00:00Z",
  "database": {
    "connected": true,
    "message": "Database connection healthy",
    "details": {
      "database_name": "datasense_db",
      "server": "localhost",
      "port": 5432,
      "status": "connected"
    }
  }
}
```

---

## 📜 License
MIT License.
