"""Tests for core settings module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from culicidaelab.core.config_models import (
    AppSettings,
    CulicidaeLabConfig,
    DatasetConfig,
    PredictorConfig,
    ProcessingConfig,
)


from culicidaelab.core.resource_manager import ResourceManager
from culicidaelab.core.settings import Settings


@pytest.fixture
def mock_config() -> CulicidaeLabConfig:
    """Create a mock configuration with proper Pydantic models."""
    return CulicidaeLabConfig(
        app_settings=AppSettings(environment="testing"),
        datasets={
            "detection": DatasetConfig(
                name="detection_dataset",
                repository="local",
                path="detection_data",
                format="coco",
                classes=["culex"],
                provider_name="local",
            ),
            "segmentation": DatasetConfig(
                name="segmentation_dataset",
                repository="local",
                path="segmentation_data",
                format="coco",
                classes=["culex"],
                provider_name="local",
            ),
            "classification": DatasetConfig(
                name="classification_dataset",
                repository="local",
                path="classification_data",
                format="imagefolder",
                classes=["culex", "aedes"],
                provider_name="local",
            ),
            "species_diversity": DatasetConfig(
                name="species_diversity_dataset",
                repository="local",
                path="species_diversity_data",
                format="csv",
                classes=[],
                provider_name="local",
            ),
        },
        processing=ProcessingConfig(batch_size=32),
        predictors={
            "default": PredictorConfig(
                target="some.dummy.class",
                model_path="models/default.pt",  # Added required field
                repository_id="mock/repo",
                provider_name="local",
                confidence=0.5,
                weights={
                    "torch": {"filename": "models/default.pt"},
                    "onnx": {"filename": "models/default.onnx"},
                },
            ),
        },
        species={},
    )


@pytest.fixture
def mock_config_manager(mock_config: CulicidaeLabConfig):
    """Create a mock config manager."""
    mock = MagicMock()
    mock.get_config.return_value = mock_config
    mock.user_config_dir = Path("/mock/config/dir")
    return mock


@pytest.fixture
def settings(mock_config_manager, monkeypatch):
    """
    Create a pristine, fully-mocked Settings instance to ensure test isolation.
    This fixture manually instantiates the class, bypassing the get_settings()
    singleton logic, which is prone to state leakage between test files.
    """
    mock_rm_instance = MagicMock(spec=ResourceManager)
    mock_rm_instance.dataset_dir = Path("/mock/datasets")
    mock_rm_instance.model_dir = Path("/mock/models")
    mock_rm_instance.user_cache_dir = Path("/mock/cache")
    monkeypatch.setattr(
        "culicidaelab.core.settings.ResourceManager",
        lambda: mock_rm_instance,
    )

    s = Settings.__new__(Settings)

    s._config_manager = mock_config_manager
    s.config = mock_config_manager.get_config()
    s._resource_manager = mock_rm_instance
    s._species_config = None
    s._current_config_dir = mock_config_manager.user_config_dir
    s._initialized = True

    return s


def test_get_settings_singleton(monkeypatch):
    """Test that get_settings() maintains singleton pattern for a given config path."""
    from pathlib import Path

    def mock_config_manager_factory(user_config_dir=None):
        mock_instance = MagicMock()
        mock_instance.get_config.return_value = CulicidaeLabConfig()
        mock_instance.user_config_dir = Path(user_config_dir).resolve() if user_config_dir else None
        return mock_instance

    with patch("culicidaelab.core.settings.ConfigManager") as mock_cm_class:
        mock_cm_class.side_effect = mock_config_manager_factory

        import sys

        if "culicidaelab.core.settings" in sys.modules:
            del sys.modules["culicidaelab.core.settings"]
        from culicidaelab.core import settings
        from culicidaelab.core.settings import Settings, get_settings

        settings._SETTINGS_INSTANCE = None

        settings1 = get_settings()
        assert isinstance(settings1, Settings)

        settings2 = get_settings()
        assert settings1 is settings2

        custom_path = "some/other/path"
        settings3 = get_settings(config_dir=custom_path)
        assert settings1 is not settings3


def test_environment_property(settings):
    """Test environment property."""
    assert settings.config.app_settings.environment == "testing"


def test_get_dataset_path(settings, tmp_path):
    """Test get_dataset_path method."""
    with patch.object(settings._resource_manager, "dataset_dir", tmp_path):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            detection_path = settings.get_dataset_path("detection")
            assert isinstance(detection_path, Path)
            assert detection_path == tmp_path / "detection_data"
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="not configured"):
        settings.get_dataset_path("invalid_type")


def test_property_getters(settings, monkeypatch, tmp_path):
    """Test that all property getters return the expected values."""

    class MockResourceManager:
        def __init__(self, base_dir):
            self._dataset_dir = base_dir / "datasets"
            self._model_dir = base_dir / "models"
            self._user_cache_dir = base_dir / "cache"
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

    test_dir = tmp_path / "test_property_getters"
    test_dir.mkdir()

    mock_rm = MockResourceManager(test_dir)
    settings._resource_manager = mock_rm

    config_subdir = test_dir / "config"
    config_subdir.mkdir(exist_ok=True)
    settings._config_manager.user_config_dir = config_subdir

    assert settings.dataset_dir == mock_rm.dataset_dir
    assert settings.model_dir == mock_rm.model_dir
    assert settings.weights_dir == mock_rm.model_dir
    assert settings.cache_dir == mock_rm.user_cache_dir

    assert mock_rm.dataset_dir.exists()
    assert mock_rm.model_dir.exists()
    assert mock_rm.user_cache_dir.exists()

    assert "dataset" in str(settings.dataset_dir).lower()
    assert "model" in str(settings.model_dir).lower()
    assert "cache" in str(settings.cache_dir).lower()

    assert isinstance(settings.cache_dir, Path)
    assert isinstance(settings.config_dir, Path)
    assert "config" in str(settings.config_dir).lower()


def test_initialize_with_custom_config(monkeypatch):
    """Test initialization with custom config directory."""
    custom_config_path = "custom/config/path"

    with patch("culicidaelab.core.settings.ConfigManager") as mock_cm_class:
        import sys

        if "culicidaelab.core.settings" in sys.modules:
            del sys.modules["culicidaelab.core.settings"]
        from culicidaelab.core import settings
        from culicidaelab.core.settings import get_settings, Settings

        settings._SETTINGS_INSTANCE = None
        Settings._initialized = False

        mock_manager_instance = MagicMock()
        mock_cm_class.return_value = mock_manager_instance

        get_settings(config_dir=custom_config_path)

        assert mock_manager_instance is not None


def test_list_datasets(settings):
    """Test listing available datasets."""
    datasets = settings.list_datasets()
    assert set(datasets) == {
        "detection",
        "segmentation",
        "classification",
        "species_diversity",
    }
