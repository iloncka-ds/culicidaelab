"""Tests for core settings module."""

import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from culicidaelab.core.settings import Settings, get_settings


@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict(os.environ, {"APP_ENV": "testing"}, clear=True):
        yield


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return OmegaConf.create(
        {
            "paths": {"config_dir": "conf"},
            "datasets": {
                "paths": {
                    "detection": "detection_data",
                    "segmentation": "segmentation_data",
                    "classification": "classification_data",
                },
            },
            "processing": {
                "batch_size": 32,
                "num_workers": 4,
            },
            "resource_mapping": {
                "data_dir": "data",
            },
        },
    )


@pytest.fixture
def mock_config_manager(mock_config):
    """Create a mock config manager."""
    mock = MagicMock()
    mock.get_config.return_value = mock_config

    mock.get_resource_dirs.return_value = {
        "cache_dir": Path("cache"),
        "weights_dir": Path("weights"),
        "datasets_dir": Path("datasets"),
        "dataset_dir": Path("datasets"),
    }
    return mock


@pytest.fixture
def settings(mock_env_vars, mock_config_manager):
    """Create a settings instance with mocked dependencies."""
    with patch("culicidaelab.core.settings.ConfigManager", return_value=mock_config_manager):
        Settings._instance = None
        settings = Settings()
        settings._get_abs_path = lambda x: Path(x)
        return settings


def test_singleton_pattern():
    """Test that Settings follows the singleton pattern."""
    with patch("culicidaelab.core.settings.ConfigManager"):
        Settings._instance = None
        settings1 = Settings()
        settings2 = Settings()
        assert settings1 is settings2


def test_get_settings_singleton():
    """Test get_settings function maintains singleton pattern."""
    with patch("culicidaelab.core.settings.ConfigManager"):
        global _settings_instance
        Settings._instance = None
        _settings_instance = None

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

        settings3 = get_settings("custom/config/dir")
        assert settings1 is not settings3


def test_environment_setup(settings):
    """Test environment setup."""
    assert settings.environment == "testing"


def test_get_resource_dir(settings):
    """Test get_resource_dir method."""
    cache_dir = settings.get_resource_dir("cache_dir")
    assert isinstance(cache_dir, Path)
    assert cache_dir.name == "cache"

    with pytest.raises(ValueError, match="Unknown resource type"):
        settings.get_resource_dir("invalid_dir")


def test_get_dataset_path(settings, tmp_path):
    """Test get_dataset_path method."""
    settings.get_resource_dir = lambda x: tmp_path if x == "dataset_dir" else Path(x)

    detection_path = settings.get_dataset_path("detection")
    assert isinstance(detection_path, Path)
    assert detection_path.name == "detection_data"

    with pytest.raises(ValueError, match="Unknown dataset type"):
        settings.get_dataset_path("invalid_type")


def test_get_processing_params(settings, mock_config):
    """Test get_processing_params method."""
    settings._config = mock_config
    params = settings.get_processing_params()
    assert isinstance(params, dict)
    assert params["batch_size"] == 32
    assert params["num_workers"] == 4


def test_property_getters(settings):
    """Test property getters."""
    assert isinstance(settings.weights_dir, Path)
    assert isinstance(settings.datasets_dir, Path)
    assert isinstance(settings.cache_dir, Path)
    assert isinstance(settings.config_dir, Path)


def test_initialize_with_custom_config():
    """Test initialization with custom config directory."""
    custom_config = "custom/config/path"
    Settings._instance = None

    with patch("culicidaelab.core.settings.ConfigManager") as mock_cm:
        mock_cm.return_value = MagicMock()
        settings = Settings(custom_config)
        mock_cm.assert_called_once()
        assert mock_cm.call_args[0][0] == custom_config


def test_directory_setup(settings, tmp_path):
    """Test directory setup with temporary path."""
    resource_dirs = {
        "cache_dir": tmp_path / "cache",
        "weights_dir": tmp_path / "weights",
        "datasets_dir": tmp_path / "datasets",
    }
    settings.config_manager.get_resource_dirs.return_value = resource_dirs

    settings._setup_directories()

    for dir_path in resource_dirs.values():
        assert dir_path.exists()
        assert dir_path.is_dir()


def test_config_reload(settings, mock_config):
    """Test configuration reloading."""
    settings._config = mock_config
    original_config = settings._config
    settings.__init__()
    assert settings._config is original_config
