"""Unit tests for configuration settings."""

from configuration.settings import Settings


def test_default_settings():
    """Verify settings loads default attributes correctly."""
    s = Settings()
    assert s.APP_NAME == "DataSense AI"
    assert s.API_VERSION == "v1"
    assert isinstance(s.PORT, int)
    assert isinstance(s.CORS_ORIGINS, list)
