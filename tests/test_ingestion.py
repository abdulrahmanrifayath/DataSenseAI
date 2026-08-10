"""Unit tests for DataIngestionService."""

import pytest
import io
import pandas as pd
from datasense.data_processing.ingestion import DataIngestionService


def test_load_csv_from_string_and_bytes():
    """Verify CSV ingestion parses standard comma-separated content."""
    csv_text = "id,name,score\n1,Alice,95\n2,Bob,88\n3,Charlie,92\n"
    df = DataIngestionService.load_csv(csv_text, filename="test.csv")

    assert len(df) == 3
    assert list(df.columns) == ["id", "name", "score"]
    assert df["score"].sum() == 275


def test_load_csv_semicolon_delimiter():
    """Verify CSV ingestion automatically detects semicolon delimiter."""
    csv_text = "id;region;sales\n100;East;500.5\n101;West;750.0\n"
    df = DataIngestionService.load_csv(csv_text.encode("utf-8"), filename="semicolon.csv")

    assert len(df) == 2
    assert "region" in df.columns
    assert "sales" in df.columns


def test_ingest_file_helper_csv():
    """Verify ingest_file router correctly ingests CSV bytes."""
    csv_bytes = b"col_a,col_b\n10,20\n30,40\n"
    df = DataIngestionService.ingest_file(csv_bytes, "sample.csv")

    assert len(df) == 2
    assert list(df.columns) == ["col_a", "col_b"]


def test_load_csv_malformed_input():
    """Verify ingestion handles malformed or empty content gracefully by raising ValueError."""
    with pytest.raises(ValueError):
        DataIngestionService.load_csv(b"", filename="empty.csv")
