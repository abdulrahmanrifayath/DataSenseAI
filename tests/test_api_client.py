"""Unit tests for DataSenseAPIClient wrapper."""

import pytest
from dashboard.api_client import DataSenseAPIClient


def test_api_client_instantiation():
    """Test DataSenseAPIClient initialization and URL formatting."""
    client = DataSenseAPIClient("http://127.0.0.1:8000/")
    assert client.base_url == "http://127.0.0.1:8000"


def test_api_client_offline_graceful_handling():
    """Test API client returns None when API server is unreachable."""
    client = DataSenseAPIClient("http://127.0.0.1:59999")
    assert client.check_health() is None
    assert client.upload_dataset(b"col_a\n1", "test.csv") is None
    assert client.run_eda(9999) is None
