"""Application settings management using Pydantic Settings."""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DataSense AI Central Configuration Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General Application Settings
    APP_NAME: str = "DataSense AI"
    APP_ENV: str = Field(default="development", description="Application environment: development, staging, production")
    DEBUG: bool = Field(default=True, description="Debug mode enable/disable flag")
    API_VERSION: str = Field(default="v1", description="API route prefix version")
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", description="Secret key for signing")

    # Server Settings
    HOST: str = Field(default="127.0.0.1", description="FastAPI host binding")
    PORT: int = Field(default=8000, description="FastAPI port binding")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"],
        description="Allowed CORS origins list",
    )

    # PostgreSQL Database Settings
    POSTGRES_SERVER: str = Field(default="localhost", description="PostgreSQL database host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL database port")
    POSTGRES_USER: str = Field(default="datasense_user", description="PostgreSQL database user")
    POSTGRES_PASSWORD: str = Field(default="datasense_password", description="PostgreSQL database password")
    POSTGRES_DB: str = Field(default="datasense_db", description="PostgreSQL database name")
    DATABASE_URL: str = Field(
        default="postgresql://datasense_user:datasense_password@localhost:5432/datasense_db",
        description="Full SQLAlchemy database connection string",
    )
    DB_POOL_SIZE: int = Field(default=10, description="SQLAlchemy connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=20, description="SQLAlchemy max pool overflow")
    DB_ECHO: bool = Field(default=False, description="SQLAlchemy query logging flag")

    # Streamlit Settings
    STREAMLIT_SERVER_PORT: int = Field(default=8501, description="Streamlit UI server port")
    BACKEND_API_URL: str = Field(default="http://127.0.0.1:8000", description="FastAPI backend endpoint for dashboard")

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = Field(default="sqlite:///mlflow.db", description="MLflow tracking URI endpoint or DB")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="DataSense_AI_Default", description="Default MLflow experiment name")

    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO", description="Global log level (DEBUG, INFO, WARNING, ERROR)")
    LOG_FORMAT: str = Field(default="console", description="Log output format (console, json)")


# Single instance singleton for app-wide import
settings = Settings()
