"""Tests for core settings module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from culicidaelab.core.config_models import AppSettings, ProcessingConfig
from culicidaelab.core.settings import Settings, get_settings


@pytest.fixture
def mock_config():
    """Create a mock configuration with proper Pydantic models."""
    return {
        "app_settings": AppSettings(environment="testing"),
        "datasets": {
            "detection": MagicMock(path="detection_data"),
            "segmentation": MagicMock(path="segmentation_data"),
            "classification": MagicMock(path="classification_data"),
        },
        "processing": ProcessingConfig(batch_size=32, num_workers=4),
        "predictors": {
            "default": MagicMock(model_path="models/default.pt"),
        },
        "species": {},
    }


@pytest.fixture
def mock_config_manager(mock_config):
    """Create a mock config manager."""
    mock = MagicMock()
    mock.get_config.return_value = MagicMock(**mock_config)
    return mock


@pytest.fixture
def settings(mock_config_manager):
    """Create a settings instance with mocked dependencies."""
    with patch("culicidaelab.core.settings.ConfigManager", return_value=mock_config_manager):
        Settings._instance = None
        Settings._initialized = False
        return Settings()


def test_get_settings_singleton(monkeypatch):
    """Test that get_settings() maintains singleton pattern."""
    import importlib

    settings_module = importlib.import_module("culicidaelab.core.settings")

    # Create a mock for ConfigManager
    mock_config_manager = MagicMock()
    mock_config = MagicMock()
    mock_config_manager.get_config.return_value = mock_config
    mock_config_manager.user_config_dir = None

    # Save original instance to restore later
    original_instance = settings_module._SETTINGS_INSTANCE

    try:
        # Replace with our own lock for testing
        test_lock = type("FakeLock", (), {"__enter__": lambda self: None, "__exit__": lambda *args: None})()
        monkeypatch.setattr("culicidaelab.core.settings._SETTINGS_LOCK", test_lock)

        with patch("culicidaelab.core.settings.ConfigManager", return_value=mock_config_manager):
            # Clear the singleton
            monkeypatch.setattr("culicidaelab.core.settings._SETTINGS_INSTANCE", None)
            settings_module.Settings._initialized = False

            # First call should create a new instance
            settings1 = get_settings()
            assert settings_module._SETTINGS_INSTANCE is settings1

            # Second call should return the same instance
            settings2 = get_settings()
            assert settings1 is settings2, "get_settings() should return the same instance"

            # Different config dir should create new instance
            with patch("culicidaelab.core.settings.Path") as mock_path:
                mock_path.return_value.resolve.return_value = Path("different/config")
                settings3 = get_settings("different/config")
                assert settings1 is not settings3, "Different config dir should create new instance"

    finally:
        # Restore original instance using monkeypatch to ensure thread safety
        monkeypatch.setattr("culicidaelab.core.settings._SETTINGS_INSTANCE", original_instance)


# Removed duplicate test in favor of the more comprehensive one above


def test_environment_property(settings):
    """Test environment property."""
    assert settings.config.app_settings.environment == "testing"


def test_get_dataset_path(settings, tmp_path):
    """Test get_dataset_path method."""
    # Patch the resource manager's dataset_dir
    with patch.object(settings._resource_manager, "dataset_dir", tmp_path):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            detection_path = settings.get_dataset_path("detection")
            assert isinstance(detection_path, Path)
            assert str(detection_path).endswith("detection_data")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Test with invalid dataset type
    with pytest.raises(ValueError, match="not configured"):
        settings.get_dataset_path("invalid_type")


def test_property_getters(settings, monkeypatch, tmp_path):
    """Test that all property getters return the expected values."""

    # Create a mock ResourceManager with property methods
    class MockResourceManager:
        def __init__(self, base_dir):
            self._dataset_dir = base_dir / "datasets"
            self._model_dir = base_dir / "models"
            self._user_cache_dir = base_dir / "cache"
            # Create the directories
            self._dataset_dir.mkdir(parents=True, exist_ok=True)
            self._model_dir.mkdir(parents=True, exist_ok=True)
            self._user_cache_dir.mkdir(parents=True, exist_ok=True)

        @property
        def dataset_dir(self) -> Path:
            return self._dataset_dir

        @property
        def model_dir(self) -> Path:
            return self._model_dir

        @property
        def user_cache_dir(self) -> Path:
            return self._user_cache_dir

    # Create a unique subdirectory for this test
    test_dir = tmp_path / "test_property_getters"
    test_dir.mkdir()

    # Replace the resource manager with our mock
    mock_rm = MockResourceManager(test_dir)
    settings._resource_manager = mock_rm

    # Ensure config_dir uses a subdirectory called 'config' for meaningful path
    config_subdir = test_dir / "config"
    config_subdir.mkdir(exist_ok=True)
    settings._config_manager.user_config_dir = config_subdir

    # Test properties
    assert settings.dataset_dir == mock_rm.dataset_dir
    assert settings.model_dir == mock_rm.model_dir
    assert settings.weights_dir == mock_rm.model_dir  # Alias for model_dir
    assert settings.cache_dir == mock_rm.user_cache_dir

    # Verify paths exist
    assert mock_rm.dataset_dir.exists()
    assert mock_rm.model_dir.exists()
    assert mock_rm.user_cache_dir.exists()

    # Verify string representations
    assert "dataset" in str(settings.dataset_dir).lower()
    assert "model" in str(settings.model_dir).lower()
    assert "cache" in str(settings.cache_dir).lower()

    # Verify types and config dir
    assert isinstance(settings.cache_dir, Path)
    assert isinstance(settings.config_dir, Path)
    assert "config" in str(settings.config_dir).lower()


def test_initialize_with_custom_config():
    """Test initialization with custom config directory."""
    custom_config = "custom/config/path"
    Settings._instance = None

    with patch("culicidaelab.core.settings.ConfigManager") as mock_cm:
        mock_cm.return_value = MagicMock()
        # settings = Settings(custom_config)
        mock_cm.assert_called_once()
        assert mock_cm.call_args[1]["user_config_dir"] == custom_config


def test_list_datasets(settings):
    """Test listing available datasets."""
    datasets = settings.list_datasets()
    assert set(datasets) == {"detection", "segmentation", "classification"}


def test_get_model_weights(settings, tmp_path):
    """Test getting model weights path."""
    # Setup mock config with model path
    settings.config.predictors = {"default": MagicMock(model_path="models/default.pt")}

    # Patch the resource manager's model_dir
    settings._resource_manager.model_dir = tmp_path

    weights_path = settings.get_model_weights("default")
    assert isinstance(weights_path, Path)
    assert str(weights_path) == str(tmp_path / "models/default.pt")

    # Test with invalid model type
    with pytest.raises(ValueError, match="not configured in 'predictors'"):
        settings.get_model_weights("invalid_model")
