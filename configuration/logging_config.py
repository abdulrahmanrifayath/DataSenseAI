"""Logging setup and formatting configuration."""

import logging
import sys
from typing import Optional


class StructuredConsoleFormatter(logging.Formatter):
    """Custom clean console formatter with colored/structured output."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: blue + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.fmt)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(level: str = "INFO", log_format: str = "console") -> logging.Logger:
    """Configures root application logging system."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if log_format.lower() == "json":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s", "level":"%(levelname)s", "name":"%(name)s", "message":"%(message)s"}'
        )
    else:
        formatter = StructuredConsoleFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger
