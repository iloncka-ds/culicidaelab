import pytest
from unittest.mock import MagicMock, patch

from culicidaelab.providers.roboflow_provider import RoboflowProvider
from culicidaelab.core.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings()


def test_roboflow_provider_init(settings):
    provider = RoboflowProvider(
        settings,
        api_key="test_key",
        rf_workspace="ws",
        rf_dataset="ds",
        project_version=1,
        data_fornat="yolo",
    )
    assert provider.provider_name == "roboflow"
    assert provider.api_key == "test_key"
    assert provider.workspace == "ws"
    assert provider.project == "ds"
    assert provider.version == 1
    assert provider.model_format == "yolo"


@patch("roboflow.Roboflow")
def test_download_dataset(mock_roboflow, settings):
    mock_ws = MagicMock()
    mock_proj = MagicMock()
    mock_ver = MagicMock()
    mock_roboflow.return_value.workspace.return_value = mock_ws
    mock_ws.project.return_value = mock_proj
    mock_proj.version.return_value = mock_ver

    provider = RoboflowProvider(
        settings,
        api_key="test_key",
        rf_workspace="ws",
        rf_dataset="ds",
        project_version=1,
        data_fornat="yolo",
    )

    save_path = provider.download_dataset("test_dataset")

    mock_roboflow.assert_called_with(api_key="test_key")
    mock_ws.project.assert_called_with("ds")
    mock_proj.version.assert_called_with(1)
    mock_ver.download.assert_called_with(model_format="yolo", location=str(save_path))
    assert save_path == settings.get_dataset_path("test_dataset")
