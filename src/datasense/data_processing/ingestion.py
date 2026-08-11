"""Data Ingestion Service for CSV, Excel, and PostgreSQL sources."""

import io
from typing import Union, BinaryIO, Optional
import pandas as pd
from sqlalchemy import create_engine
from datasense.utilities.logger import get_logger

logger = get_logger("data_processing.ingestion")


class DataIngestionService:
    """Service for loading and parsing datasets from CSV, Excel, and PostgreSQL database sources."""

    @staticmethod
    def load_csv(source: Union[str, bytes, BinaryIO], filename: str = "dataset.csv") -> pd.DataFrame:
        """Parses CSV content with automatic encoding and delimiter sniffing."""
        logger.info(f"Ingesting CSV dataset: {filename}")
        encodings_to_try = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
        delimiters_to_try = [",", ";", "\t", "|"]

        if isinstance(source, str) and not ("\n" in source or ";" in source or "," in source):
            content_bytes = None
            file_path = source
        else:
            file_path = None
            if isinstance(source, str):
                content_bytes = source.encode("utf-8")
            elif isinstance(source, bytes):
                content_bytes = source
            elif hasattr(source, "read"):
                content_bytes = source.read()
            else:
                raise ValueError("Unsupported CSV source type.")

        if content_bytes is not None and len(content_bytes.strip()) == 0:
            raise ValueError("CSV source content is empty.")

        for encoding in encodings_to_try:
            # First attempt sep=None (python engine auto sniffing)
            try:
                buf = io.BytesIO(content_bytes) if content_bytes else file_path
                df = pd.read_csv(buf, encoding=encoding, sep=None, engine="python", on_bad_lines="skip")
                if not df.empty and len(df.columns) > 1:
                    return df
            except Exception:
                pass

            best_df = None
            max_cols = 0
            for sep in delimiters_to_try:
                try:
                    buf = io.BytesIO(content_bytes) if content_bytes else file_path
                    df = pd.read_csv(buf, encoding=encoding, sep=sep, engine="python", on_bad_lines="skip")
                    if not df.empty and len(df.columns) > max_cols:
                        max_cols = len(df.columns)
                        best_df = df
                except Exception:
                    continue

            if best_df is not None and not best_df.empty:
                return best_df

        raise ValueError(f"Failed to parse CSV file '{filename}'. Check delimiter or encoding.")


    @staticmethod
    def load_excel(source: Union[str, bytes, BinaryIO], filename: str = "dataset.xlsx") -> pd.DataFrame:
        """Parses Excel dataset (.xlsx, .xls)."""
        logger.info(f"Ingesting Excel dataset: {filename}")
        try:
            if isinstance(source, (bytes, bytearray)):
                buffer = io.BytesIO(source)
                return pd.read_excel(buffer)
            elif hasattr(source, "read"):
                content = source.read()
                buffer = io.BytesIO(content)
                return pd.read_excel(buffer)
            else:
                return pd.read_excel(source)
        except Exception as e:
            logger.error(f"Error parsing Excel file {filename}: {e}")
            raise ValueError(f"Unable to parse Excel file '{filename}': {str(e)}")

    @staticmethod
    def load_from_db(connection_url: str, query_or_table: str) -> pd.DataFrame:
        """Loads dataset directly from PostgreSQL or arbitrary database connection."""
        logger.info(f"Ingesting dataset from DB query/table: {query_or_table}")
        try:
            engine = create_engine(connection_url)
            clean_query = query_or_table.strip()
            if not clean_query.lower().startswith("select"):
                clean_query = f'SELECT * FROM "{clean_query}"'
            with engine.connect() as conn:
                df = pd.read_sql_query(clean_query, conn)
            return df
        except Exception as e:
            logger.error(f"Failed to execute DB load query: {e}")
            raise ValueError(f"Database extraction failed: {str(e)}")

    @classmethod
    def ingest_file(cls, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """Helper router to ingest file based on filename extension."""
        lower_name = filename.lower()
        if lower_name.endswith(".csv") or lower_name.endswith(".txt"):
            return cls.load_csv(file_bytes, filename=filename)
        elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
            return cls.load_excel(file_bytes, filename=filename)
        else:
            # Fallback attempt as CSV
            return cls.load_csv(file_bytes, filename=filename)
