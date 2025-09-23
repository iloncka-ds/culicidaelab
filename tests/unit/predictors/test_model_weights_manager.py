import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from culicidaelab.predictors.model_weights_manager import ModelWeightsManager
from culicidaelab.core.settings import Settings


@pytest.fixture
def mock_settings():
    """Fixture for mocking the Settings object."""
    settings = MagicMock(spec=Settings)
    settings.construct_weights_path.return_value = Path("/fake/path/model.pt")
    return settings


@pytest.fixture
def manager(mock_settings):
    """Provides a ModelWeightsManager instance with mocked settings."""
    with patch("culicidaelab.predictors.model_weights_manager.ProviderService"):
        return ModelWeightsManager(settings=mock_settings)


def test_ensure_weights_file_exists_locally(manager, mock_settings):
    """Test that if the weights file exists, its path is returned."""
    with patch("pathlib.Path.exists", return_value=True) as mock_exists:
        result = manager.ensure_weights("detector", "yolo")
        assert result == Path("/fake/path/model.pt")
        mock_settings.construct_weights_path.assert_called_once_with(
            predictor_type="detector",
            backend="yolo",
        )
        mock_exists.assert_called_once()


def test_ensure_weights_downloads_if_not_local(manager, mock_settings):
    """Test that weights are downloaded if they don't exist locally."""
    mock_settings.construct_weights_path.return_value = Path("/fake/weights/model.pt")
    mock_settings.get_config.side_effect = [
        MagicMock(repository_id="user/repo", provider_name="huggingface"),
        MagicMock(filename="model.pt"),
    ]

    with patch("pathlib.Path.exists", return_value=False):
        mock_provider = manager.provider_service.get_provider.return_value
        mock_provider.download_model_weights.return_value = Path("/fake/weights/model.pt")

        result = manager.ensure_weights("detector", "yolo")

        assert result == Path("/fake/weights/model.pt")
        mock_settings.get_config.assert_any_call("predictors.detector")
        mock_settings.get_config.assert_any_call("predictors.detector.weights.yolo")
        mock_provider.download_model_weights.assert_called_once_with(
            repo_id="user/repo",
            filename="model.pt",
            local_dir=Path("/fake/weights"),
        )


def test_ensure_weights_missing_repo_id_raises_error(manager, mock_settings):
    """Test that a RuntimeError is raised if repo_id is missing."""
    mock_settings.get_config.side_effect = [
        MagicMock(repository_id=None, provider_name="huggingface"),
        MagicMock(filename="model.pt"),
    ]
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(RuntimeError, match="Missing 'repository_id' or 'filename'"):
            manager.ensure_weights("detector", "yolo")


def test_ensure_weights_download_fails_raises_error(manager, mock_settings):
    """Test that a RuntimeError is raised if the download fails."""
    mock_settings.construct_weights_path.return_value = Path("/fake/weights/model.pt")
    mock_settings.get_config.side_effect = [
        MagicMock(repository_id="user/repo", provider_name="huggingface"),
        MagicMock(filename="model.pt"),
    ]

    with patch("pathlib.Path.exists", return_value=False):
        mock_provider = manager.provider_service.get_provider.return_value
        mock_provider.download_model_weights.side_effect = Exception("Download failed")

        with pytest.raises(RuntimeError, match="Failed to resolve weights"):
            manager.ensure_weights("detector", "yolo")
