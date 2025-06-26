import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import requests

from culicidaelab.providers.huggingface_provider import HuggingFaceProvider
from culicidaelab.core.config_models import DatasetConfig


@pytest.fixture
def mock_settings():
    """Provides a mock Settings object for testing."""
    settings = Mock()
    settings.get_dataset_path.return_value = Path("/fake/datasets/my_dataset")
    settings.get_model_weights_path.return_value = Path("/fake/models/classifier.pt")
    settings.get_api_key.return_value = "fake_api_key_from_env"

    dataset_config = DatasetConfig(
        name="my_dataset",
        path="culicidae/my_dataset",
        repository="culicidae/my_dataset_repo",
        format="hf",
        classes=[],
        provider_name="huggingface",
    )
    settings.get_config.return_value = dataset_config

    return settings


@pytest.fixture
def hf_provider(mock_settings):
    """Provides a HuggingFaceProvider instance."""
    return HuggingFaceProvider(
        settings=mock_settings,
        dataset_url="http://fake-hf.com/api/{dataset_name}",
    )


def test_initialization(mock_settings):
    # Test with api_key from kwargs
    provider_with_key = HuggingFaceProvider(settings=mock_settings, dataset_url="url", api_key="kwarg_key")
    assert provider_with_key.api_key == "kwarg_key"

    # Test with api_key from settings (env var)
    provider_from_env = HuggingFaceProvider(settings=mock_settings, dataset_url="url")
    assert provider_from_env.api_key == "fake_api_key_from_env"
    mock_settings.get_api_key.assert_called_once_with("huggingface")


@patch("requests.get")
def test_get_dataset_metadata_success(mock_get, hf_provider):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"id": "my_dataset", "files": []}
    mock_get.return_value = mock_response

    metadata = hf_provider.get_dataset_metadata("my_dataset")

    expected_url = "http://fake-hf.com/api/my_dataset"
    mock_get.assert_called_once_with(
        expected_url,
        headers={"Authorization": "Bearer fake_api_key_from_env"},
        timeout=10.0,
    )
    assert metadata["id"] == "my_dataset"


@patch("requests.get")
def test_get_dataset_metadata_failure(mock_get, hf_provider):
    mock_get.side_effect = requests.RequestException("Connection error")

    with pytest.raises(requests.RequestException, match="Failed to fetch dataset metadata"):
        hf_provider.get_dataset_metadata("my_dataset")


@patch("culicidaelab.providers.huggingface_provider.load_dataset")
def test_download_dataset(mock_load_dataset, hf_provider, mock_settings):
    mock_hf_dataset = MagicMock()
    mock_load_dataset.return_value = mock_hf_dataset
    save_path = Path("/fake/datasets/my_dataset")

    result_path = hf_provider.download_dataset("my_dataset", split="train")

    mock_load_dataset.assert_called_once_with("culicidae/my_dataset_repo", split="train", token="fake_api_key_from_env")
    expected_save_path = save_path / "train"
    mock_hf_dataset.save_to_disk.assert_called_once_with(str(expected_save_path))
    assert result_path == expected_save_path


@patch("culicidaelab.providers.huggingface_provider.load_from_disk")
def test_load_dataset(mock_load_from_disk, hf_provider):
    path_to_load = "/local/disk/path"
    hf_provider.load_dataset(path_to_load)
    mock_load_from_disk.assert_called_once_with(path_to_load)


@patch("culicidaelab.providers.huggingface_provider.hf_hub_download")
@patch("pathlib.Path.exists")
def test_download_model_weights_not_found(mock_exists, mock_hf_download, hf_provider, mock_settings):
    mock_exists.return_value = False  # File does not exist

    # Mock predictor config
    predictor_config = Mock()
    predictor_config.repository_id = "org/model_repo"
    predictor_config.filename = "weights.pt"
    mock_settings.get_config.return_value = predictor_config
    mock_settings.cache_dir = Path("/fake/cache")

    local_path = Path("/fake/models/classifier.pt")
    mock_hf_download.return_value = str(local_path)

    result_path = hf_provider.download_model_weights("classifier")

    mock_hf_download.assert_called_once()
    assert result_path == local_path


@patch("pathlib.Path.exists")
def test_download_model_weights_already_exists(mock_exists, hf_provider):
    mock_exists.return_value = True  # File exists

    result_path = hf_provider.download_model_weights("classifier")

    assert result_path == Path("/fake/models/classifier.pt")
