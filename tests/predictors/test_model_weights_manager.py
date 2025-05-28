import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager


@pytest.fixture
def mock_config():
    config = Mock()
    config.models.weights = {
        "detection": Mock(
            local_path="models/detection.pt",
            remote_repo="iloncka/culico-net-det-v1",
            remote_file="culico-net-det-v1-nano.pt",
        ),
    }
    # Use absolute path for root dir
    config.paths.root_dir = Path(__file__).parent.parent.parent.absolute()
    return config


@pytest.fixture
def weights_manager(mock_config):
    manager = ModelWeightsManager(Mock())
    manager._config = mock_config
    return manager


def test_init(weights_manager):
    assert isinstance(weights_manager, ModelWeightsManager)


def test_get_weights_local_exists(weights_manager):
    project_root = weights_manager._config.paths.root_dir
    expected_path = project_root / "models/detection.pt"

    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        path = weights_manager.get_weights("detection")
        assert Path(path).resolve() == expected_path.resolve()


def test_get_weights_invalid_type(weights_manager):
    with pytest.raises(ValueError, match="Unknown model type"):
        weights_manager.get_weights("invalid_type")


@patch("builtins.input", return_value="y")
@patch("culicidaelab.predictors.model_weights_manager.hf_hub_download")
@patch("urllib3.connectionpool.HTTPSConnectionPool")
def test_get_weights_download(
    mock_pool,
    mock_download,
    mock_input,
    weights_manager,
    tmp_path,
):
    # Setup paths
    project_root = weights_manager._config.paths.root_dir
    models_dir = project_root / "models"

    # Create and setup mock download file
    download_file = (
        tmp_path / "culico-net-det-v1-nano.pt.5e55930b01a0b014201b579ed00b198e941d09716d9be1437e64e062fbd29494"
    )
    download_file.touch()
    mock_download.return_value = str(download_file)

    # Mock network operations
    mock_pool.return_value.request.return_value.status = 200
    mock_pool.return_value.request.return_value.data = b"mock_data"

    # Track whether download has been called to simulate file creation
    download_called = False

    def exists_side_effect(p):
        nonlocal download_called
        p = str(Path(p).resolve())

        # Always return False for .netrc files
        if ".netrc" in p:
            return False

        # For the detection.pt file, return False initially (file doesn't exist)
        # but True after download is called (file exists after download)
        if p == str((models_dir / "detection.pt").resolve()):
            return download_called

        return False

    # Mock the download to set the flag when called
    def mock_download_side_effect(*args, **kwargs):
        nonlocal download_called
        download_called = True
        return str(download_file)

    mock_download.side_effect = mock_download_side_effect

    with (
        patch("os.path.exists", side_effect=exists_side_effect),
        patch("shutil.move") as mock_move,
    ):
        path = weights_manager.get_weights("detection")
        expected_path = models_dir / "detection.pt"

        assert Path(path).resolve() == expected_path.resolve()
        mock_download.assert_called_once()

        # Verify the download was called with correct parameters
        mock_download.assert_called_with(
            repo_id=weights_manager._config.models.weights["detection"].remote_repo,
            filename=weights_manager._config.models.weights["detection"].remote_file,
            local_dir=str(models_dir),
        )

        # Verify shutil.move was called to move the file to the correct location
        mock_move.assert_called_once_with(str(download_file), expected_path)


@patch("builtins.input", return_value="n")
def test_get_weights_download_declined(mock_input, weights_manager):
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            weights_manager.get_weights("detection")


def test_get_abs_path(weights_manager):
    relative_path = "models/weights.pt"
    abs_path = weights_manager._get_abs_path(relative_path)
    expected_path = weights_manager._config.paths.root_dir / "models/weights.pt"
    assert abs_path == expected_path.resolve()

    # Test absolute path handling
    absolute_path = Path("/weights.pt")
    abs_path = weights_manager._get_abs_path(absolute_path)
    assert abs_path == absolute_path.resolve()
