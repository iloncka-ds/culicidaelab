import pytest
from unittest.mock import Mock
from pathlib import Path

from culicidaelab.datasets.datasets_manager import DatasetsManager
from culicidaelab.core.settings import Settings
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.config_models import DatasetConfig


@pytest.fixture
def mock_settings():
    """Fixture for a mocked Settings object."""
    settings = Mock(spec=Settings)
    # Mock config data
    mock_dataset_config = Mock(spec=DatasetConfig)
    mock_dataset_config.provider_name = "mock_provider"
    mock_dataset_config.name = "classification"
    mock_dataset_config.path = "some/path/to/classification"

    # Configure get_config to return the mock config for a specific dataset
    settings.get_config.side_effect = lambda path: (mock_dataset_config if path == "datasets.classification" else None)
    settings.list_datasets.return_value = ["classification"]
    return settings


@pytest.fixture
def mock_provider():
    """Fixture for a mocked BaseProvider object."""
    provider = Mock()
    provider.download_dataset.return_value = Path("/fake/path/classification_dataset")
    provider.load_dataset.return_value = {"data": "mock_dataset_content"}
    return provider


@pytest.fixture
def mock_provider_service(mock_provider):
    """Fixture for a mocked ProviderService."""
    provider_service = Mock(spec=ProviderService)
    provider_service.get_provider.return_value = mock_provider
    return provider_service


@pytest.fixture
def datasets_manager(mock_settings, mock_provider_service):
    """Fixture to create a DatasetsManager instance with mocked dependencies."""
    return DatasetsManager(settings=mock_settings, provider_service=mock_provider_service)


def test_get_dataset_info(datasets_manager, mock_settings):
    """Test that getting dataset info retrieves the correct config."""
    dataset_name = "classification"
    config = datasets_manager.get_dataset_info(dataset_name)
    mock_settings.get_config.assert_called_once_with(f"datasets.{dataset_name}")
    assert config is not None
    assert config.name == dataset_name


def test_get_dataset_info_not_found(datasets_manager, mock_settings):
    """Test that getting info for a non-existent dataset raises KeyError."""
    mock_settings.get_config.return_value = None
    with pytest.raises(KeyError, match="Dataset 'nonexistent' not found in configuration."):
        datasets_manager.get_dataset_info("nonexistent")


def test_list_datasets(datasets_manager, mock_settings):
    """Test that list_datasets returns the list from settings."""
    datasets_list = datasets_manager.list_datasets()
    mock_settings.list_datasets.assert_called_once()
    assert datasets_list == ["classification"]


def test_load_dataset_first_time(datasets_manager, mock_provider_service, mock_provider):
    """Test loading a dataset that is not yet cached."""
    dataset_name = "classification"
    # Call the method to test
    dataset = datasets_manager.load_dataset(dataset_name, split="train")

    # Assert that the provider was retrieved
    mock_provider_service.get_provider.assert_called_once_with("mock_provider")

    # Assert that the dataset was downloaded and loaded
    mock_provider.download_dataset.assert_called_once_with(dataset_name, split="train")
    mock_provider.load_dataset.assert_called_once_with(Path("/fake/path/classification_dataset"), split="train")

    # Assert that the dataset content is correct and it's now cached
    assert dataset == {"data": "mock_dataset_content"}
    assert dataset_name in datasets_manager.loaded_datasets
    assert datasets_manager.loaded_datasets[dataset_name] == Path("/fake/path/classification_dataset")


def test_load_dataset_cached(datasets_manager, mock_provider):
    """Test that loading a cached dataset does not trigger a new download."""
    dataset_name = "classification"
    cached_path = Path("/cached/path/to/dataset")

    # Pre-populate the cache
    datasets_manager.loaded_datasets[dataset_name] = cached_path

    # Reset mocks to check for new calls
    mock_provider.download_dataset.reset_mock()
    mock_provider.load_dataset.reset_mock()

    # Call the method to test
    datasets_manager.load_dataset(dataset_name, split="test")

    # Assert that download was NOT called
    mock_provider.download_dataset.assert_not_called()

    # Assert that load_dataset was called with the cached path
    mock_provider.load_dataset.assert_called_once_with(cached_path, split="test")


def test_load_dataset_not_configured(datasets_manager, mock_settings):
    """Test that loading a non-configured dataset raises KeyError."""
    dataset_name = "nonexistent"
    mock_settings.get_config.return_value = None
    with pytest.raises(KeyError, match=f"Dataset '{dataset_name}' not found in configuration."):
        datasets_manager.load_dataset(dataset_name)


def test_list_loaded_datasets_empty(datasets_manager):
    """Test that listing loaded datasets is empty initially."""
    assert datasets_manager.list_loaded_datasets() == []


def test_list_loaded_datasets_after_loading(datasets_manager):
    """Test that listing loaded datasets works after loading a dataset."""
    dataset_name = "classification"
    # Pre-populate the cache as if a dataset was loaded
    datasets_manager.loaded_datasets[dataset_name] = Path("/fake/path")
    assert datasets_manager.list_loaded_datasets() == [dataset_name]
