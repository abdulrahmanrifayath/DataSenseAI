"""Central application logger module."""

import logging
from configuration.settings import settings
from configuration.logging_config import setup_logging

# Initialize global logger
setup_logging(level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
logger = logging.getLogger("datasense")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger named under the datasense namespace."""
    return logging.getLogger(f"datasense.{name}")
