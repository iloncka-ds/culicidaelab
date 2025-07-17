"""Tests for core utility functions."""

import pytest
import requests
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from culicidaelab.core.utils import download_file


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


def test_download_file_invalid_url():
    """Test download_file with invalid URL."""
    with pytest.raises(ValueError, match="Invalid URL"):
        download_file("not-a-url")


def test_download_file_success(temp_dir):
    """Test successful file download."""
    test_url = "https://example.com/test.txt"
    test_content = b"test content"

    mock_response = MagicMock()
    mock_response.headers = {"content-length": str(len(test_content))}
    mock_response.iter_content.return_value = [test_content]
    mock_response.raise_for_status.return_value = None
    mock_response.__enter__.return_value = mock_response

    with patch("requests.get", return_value=mock_response) as mock_get:
        result = download_file(test_url, downloads_dir=temp_dir)

        assert result.exists()
        assert result.name == "test.txt"
        assert result.read_bytes() == test_content

        mock_get.assert_called_once_with(
            test_url,
            stream=True,
            timeout=30,
        )


def test_download_file_request_exception():
    """Test download_file when request fails."""
    with patch("requests.get", side_effect=requests.RequestException("Network error")):
        with pytest.raises(RuntimeError, match="Failed to download file"):
            download_file("https://example.com/test.txt")


def test_download_file_with_custom_destination(temp_dir):
    """Test download_file with custom destination path."""
    test_url = "https://example.com/test.txt"
    custom_dest = temp_dir / "custom" / "path" / "myfile.txt"

    mock_response = MagicMock()
    mock_response.headers = {"content-length": "10"}
    mock_response.iter_content.return_value = [b"test data"]
    mock_response.raise_for_status.return_value = None
    mock_response.__enter__.return_value = mock_response

    with patch("requests.get", return_value=mock_response):
        result = download_file(test_url, destination=custom_dest)

        assert result == custom_dest
        assert result.exists()
        assert result.parent.exists()
