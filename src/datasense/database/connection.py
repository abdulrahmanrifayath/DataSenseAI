"""PostgreSQL and SQLite Fallback Database Connection and Session Management module."""

from typing import Generator, Tuple, Dict, Any
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from sqlalchemy.exc import SQLAlchemyError

from configuration.settings import settings
from datasense.utilities.logger import get_logger

logger = get_logger("database.connection")


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


def _create_database_engine():
    """Attempts to create PostgreSQL engine, falling back to local SQLite engine if unreachable."""
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            echo=settings.DB_ECHO,
            pool_pre_ping=True,
        )
        # Test connection immediately
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Successfully connected to PostgreSQL at {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}")
        return engine, "postgresql"
    except Exception as err:
        logger.warning(f"PostgreSQL connection unavailable ({err}). Initializing SQLite local fallback database...")
        fallback_db_path = os.path.abspath("datasense.db")
        fallback_url = f"sqlite:///{fallback_db_path}"
        fallback_engine = create_engine(
            fallback_url,
            connect_args={"check_same_thread": False},
            echo=settings.DB_ECHO,
        )
        return fallback_engine, "sqlite"


engine, db_backend_type = _create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db_tables():
    """Create all declarative tables in the connected database engine."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")


# Initialize tables on import
init_db_tables()


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI Dependency for retrieving database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> Tuple[bool, str, Dict[str, Any]]:
    """Checks active database connection health status."""
    if engine is None:
        return False, "Database engine not initialized", {"status": "uninitialized"}

    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            val = result.scalar()
            if val == 1:
                return True, f"Database connection healthy ({db_backend_type})", {
                    "backend": db_backend_type,
                    "database_name": settings.POSTGRES_DB if db_backend_type == "postgresql" else "datasense.db",
                    "status": "connected",
                }
            return False, "Database ping returned unexpected response", {"status": "error"}
    except SQLAlchemyError as err:
        logger.warning(f"Database health check failed: {str(err)}")
        return False, f"Database connection failed: {str(err)}", {
            "backend": db_backend_type,
            "status": "disconnected",
        }
