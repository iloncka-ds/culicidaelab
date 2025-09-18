import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from culicidaelab.predictors.model_weights_manager import ModelWeightsManager
from culicidaelab.core.config_models import PredictorConfig, WeightDetails


@pytest.fixture
def mock_settings(tmp_path) -> Mock:
    """Fixture providing a mocked Settings instance with a valid model_dir."""
    settings = Mock()
    settings.model_dir = tmp_path  # Use a real Path object from pytest's tmp_path
    return settings


def test_init(mock_settings: Mock):
    """ModelWeightsManager stores the provided settings instance."""
    wm = ModelWeightsManager(settings=mock_settings)
    assert wm.settings is mock_settings


@patch("culicidaelab.predictors.model_weights_manager.construct_weights_path")
@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_successful_download(
    mock_provider_cls: Mock,
    mock_construct_path: Mock,
    mock_settings: Mock,
):
    """ensure_weights orchestrates a successful download."""
    predictor_type = "classifier"
    backend_type = "torch"

    # Define the mock base directory
    mock_base_dir = Path("/mock/models")
    mock_settings.model_dir = mock_base_dir

    # This is the full path the file will have
    final_file_path = mock_base_dir / predictor_type / backend_type / "model.pkl"
    # This is just the directory part
    expected_local_dir = final_file_path.parent

    # --- Mocks Setup ---
    # Mock the path construction to return our expected final path
    mock_construct_path.return_value = final_file_path

    with patch("pathlib.Path.exists", return_value=False):
        # Mock configs
        predictor_config = PredictorConfig(
            target="some.class.path",
            repository_id="repo/cls",
            provider_name="huggingface",
            weights={"torch": WeightDetails(filename="model.pkl")},
        )
        weights_config = predictor_config.weights[backend_type]

        def get_config_side_effect(key):
            if key == f"predictors.{predictor_type}":
                return predictor_config
            if key == f"predictors.{predictor_type}.weights.{backend_type}":
                return weights_config
            return None

        mock_settings.get_config.side_effect = get_config_side_effect

        # ProviderService mock setup
        mock_provider = mock_provider_cls.return_value.get_provider.return_value
        mock_provider.download_model_weights.return_value = final_file_path

        # --- Act ---
        wm = ModelWeightsManager(settings=mock_settings)
        result = wm.ensure_weights(predictor_type, backend_type)

        # --- Assertions ---
        assert result == final_file_path

        mock_construct_path.assert_called_once_with(
            model_dir=mock_base_dir,
            predictor_type=predictor_type,
            predictor_config=predictor_config,
            backend=backend_type,
        )

        mock_provider.download_model_weights.assert_called_once_with(
            repo_id="repo/cls",
            filename="model.pkl",
            local_dir=expected_local_dir,
        )


@patch("culicidaelab.predictors.model_weights_manager.construct_weights_path")
@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_uses_correct_repo(
    mock_provider_cls: Mock,
    mock_construct_path: Mock,
    mock_settings: Mock,
):
    """ensure_weights uses the repository_id from the main predictor config."""
    predictor_type = "detector"
    backend_type = "onnx"

    mock_base_dir = Path("/mock/models")
    mock_settings.model_dir = mock_base_dir

    final_file_path = mock_base_dir / predictor_type / backend_type / "model.onnx"
    expected_local_dir = final_file_path.parent

    # --- Mocks Setup ---
    mock_construct_path.return_value = final_file_path

    with patch("pathlib.Path.exists", return_value=False):
        predictor_config = PredictorConfig(
            target="some.class.path",
            repository_id="repo/main-detector",
            provider_name="huggingface",
            weights={"onnx": WeightDetails(filename="model.onnx")},
        )
        weights_config = predictor_config.weights[backend_type]

        def get_config_side_effect(key):
            if key == f"predictors.{predictor_type}":
                return predictor_config
            if key == f"predictors.{predictor_type}.weights.{backend_type}":
                return weights_config
            return None

        mock_settings.get_config.side_effect = get_config_side_effect

        mock_provider = mock_provider_cls.return_value.get_provider.return_value

        # --- Act ---
        wm = ModelWeightsManager(settings=mock_settings)
        wm.ensure_weights(predictor_type, backend_type)

        # --- Assertions ---
        mock_provider.download_model_weights.assert_called_once_with(
            repo_id="repo/main-detector",
            filename="model.onnx",
            local_dir=expected_local_dir,
        )


@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_raises_runtime_error_on_provider_failure(mock_provider_cls: Mock, mock_settings: Mock):
    """ensure_weights wraps provider failures in RuntimeError."""
    predictor_type = "segmenter"
    backend_type = "torch"

    predictor_config = PredictorConfig(
        target="some.class.path",
        repository_id="repo/seg",
        provider_name="huggingface",
        weights={"torch": WeightDetails(filename="model.pt")},
    )
    weights_config = predictor_config.weights[backend_type]

    def get_config_side_effect(key):
        if key == f"predictors.{predictor_type}":
            return predictor_config
        if key == f"predictors.{predictor_type}.weights.{backend_type}":
            return weights_config
        return None

    mock_settings.get_config.side_effect = get_config_side_effect

    mock_provider = mock_provider_cls.return_value.get_provider.return_value
    mock_provider.download_model_weights.side_effect = ConnectionError("network down")

    wm = ModelWeightsManager(settings=mock_settings)
    with pytest.raises(RuntimeError, match="Failed to resolve weights"):
        wm.ensure_weights(predictor_type, backend_type)
