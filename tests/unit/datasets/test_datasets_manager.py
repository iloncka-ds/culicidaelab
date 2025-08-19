import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from culicidaelab.datasets.datasets_manager import DatasetsManager
from culicidaelab.core.settings import Settings
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.config_models import DatasetConfig


@pytest.fixture
def mock_settings():
    """Fixture for a mocked Settings object."""
    settings = Mock(spec=Settings)
    settings.cache_dir = Path("/fake/cache")
    settings.dataset_dir = Path("/fake/datasets")

    # Mock dataset config
    mock_dataset_config = Mock(spec=DatasetConfig)
    mock_dataset_config.provider_name = "mock_provider"
    mock_dataset_config.name = "classification"
    mock_dataset_config.path = "some/path/to/classification"

    # Mock provider config
    mock_provider_config = Mock()
    mock_provider_config.name = "mock_provider"

    # Configure get_config to return different mocks based on the path
    def get_config_side_effect(path):
        if path == "datasets.classification":
            return mock_dataset_config
        elif path == "providers.mock_provider":
            return mock_provider_config
        return None

    settings.get_config.side_effect = get_config_side_effect
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
    manager = DatasetsManager(settings=mock_settings)
    # Replace the provider_service with our mock
    manager.provider_service = mock_provider_service
    return manager


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
    with pytest.raises(
        KeyError,
        match="Dataset 'nonexistent' not found in configuration.",
    ):
        datasets_manager.get_dataset_info("nonexistent")


def test_list_datasets(datasets_manager, mock_settings):
    """Test that list_datasets returns the list from settings."""
    datasets_list = datasets_manager.list_datasets()
    mock_settings.list_datasets.assert_called_once()
    assert datasets_list == ["classification"]


@patch("pathlib.Path.exists", return_value=False)
def test_load_dataset_first_time(
    mock_path_exists,
    datasets_manager,
    mock_provider_service,
    mock_provider,
):
    """Test loading a dataset that is not yet cached."""
    dataset_name = "classification"
    dataset = datasets_manager.load_dataset(dataset_name, split="train")

    mock_provider_service.get_provider.assert_called_once_with("mock_provider")

    # Use str(Path()) to normalize path separators for the platform
    expected_path = Path("/fake/datasets/some/path/to/classification/80d5b9c7c81ecfa4")
    mock_provider.download_dataset.assert_called_once_with(
        dataset_name="classification",
        config_name="default",
        save_dir=expected_path,
        split="train",
    )
    # Provider is expected to be asked to load from the cache path created by manager
    expected_load_path = Path(expected_path)
    # Allow either the provider's own returned path or the manager-created split path
    assert mock_provider.load_dataset.call_count == 1
    actual_call = mock_provider.load_dataset.call_args
    assert actual_call is not None
    actual_arg = actual_call[0][0] if actual_call[0] else None
    actual_path = Path(actual_arg) if actual_arg is not None else None
    provider_expected = Path("/fake/path/classification_dataset")
    assert actual_path in (provider_expected, expected_load_path)

    assert dataset == {"data": "mock_dataset_content"}
    assert dataset_name in datasets_manager.loaded_datasets
    assert datasets_manager.loaded_datasets[dataset_name] == Path(
        "/fake/path/classification_dataset",
    )


def test_load_dataset_cached(datasets_manager, mock_provider):
    """Test that loading a cached dataset does not trigger a new download."""
    dataset_name = "classification"
    # Set up the same paths that would be created in load_dataset
    split_path = Path("/fake/datasets/some/path/to/classification/4d967a30111bf29f")

    # Mock that the cache directory exists
    from unittest.mock import patch

    with patch("pathlib.Path.exists") as mock_exists, patch("pathlib.Path.mkdir"):
        mock_exists.return_value = True

        mock_provider.download_dataset.reset_mock()
        mock_provider.load_dataset.reset_mock()

        datasets_manager.load_dataset(dataset_name, split="test")

    mock_provider.download_dataset.assert_not_called()
    mock_provider.load_dataset.assert_called_once_with(split_path)


def test_load_dataset_not_configured(datasets_manager, mock_settings):
    """Test that loading a non-configured dataset raises KeyError."""
    dataset_name = "nonexistent"
    mock_settings.get_config.return_value = None
    with pytest.raises(
        KeyError,
        match=f"Dataset '{dataset_name}' not found in configuration.",
    ):
        datasets_manager.load_dataset(dataset_name)


def test_list_loaded_datasets_empty(datasets_manager):
    """Test that listing loaded datasets is empty initially."""
    assert datasets_manager.list_loaded_datasets() == []


def test_list_loaded_datasets_after_loading(datasets_manager):
    """Test that listing loaded datasets works after loading a dataset."""
    dataset_name = "classification"
    datasets_manager.loaded_datasets[dataset_name] = Path("/fake/path")
    assert datasets_manager.list_loaded_datasets() == [dataset_name]
