"""FastAPI Application Entry Point for DataSense AI."""

from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from configuration.settings import settings
from datasense.api.routers import health, datasets, preprocessing, eda, ml, forecasting, anomaly, bi
from datasense.utilities.logger import get_logger

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} Backend API v{settings.API_VERSION} in [{settings.APP_ENV}] mode...")
    yield
    logger.info(f"Shutting down {settings.APP_NAME} Backend API gracefully.")


def create_app() -> FastAPI:
    """Factory function for creating and configuring the FastAPI instance."""
    app = FastAPI(
        title=f"{settings.APP_NAME} Backend API",
        description="Intelligent Business Intelligence & Predictive Analytics Platform REST APIs",
        version=settings.API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(health.router)
    app.include_router(datasets.router)
    app.include_router(preprocessing.router)
    app.include_router(eda.router)
    app.include_router(ml.router)
    app.include_router(forecasting.router)
    app.include_router(anomaly.router)
    app.include_router(bi.router)






    @app.get("/", tags=["Root"], summary="Platform Root Information")
    def root_info() -> Dict[str, Any]:
        """Root API endpoint returning platform description and available services."""
        return {
            "name": settings.APP_NAME,
            "version": settings.API_VERSION,
            "description": "DataSense AI – Intelligent Business Intelligence & Predictive Analytics Platform",
            "documentation": "/docs",
            "health_check": "/health",
            "status": "online",
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        logger.error(f"Unhandled exception occurred on path {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "details": str(exc) if settings.DEBUG else "An error occurred."},
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "datasense.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
