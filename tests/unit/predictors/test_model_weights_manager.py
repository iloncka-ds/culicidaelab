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


@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_successful_download(mock_provider_cls: Mock, mock_settings: Mock):
    """ensure_weights orchestrates a successful download."""
    predictor_type = "classifier"
    backend_type = "torch"
    expected_path = Path("/predictors/model.pkl")

    # Setup mock configs
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
    mock_provider_instance = mock_provider_cls.return_value
    mock_provider = mock_provider_instance.get_provider.return_value
    mock_provider.download_model_weights.return_value = expected_path

    wm = ModelWeightsManager(settings=mock_settings)
    result = wm.ensure_weights(predictor_type, backend_type)

    # Assertions
    assert result == expected_path
    mock_provider.download_model_weights.assert_called_once_with(
        repo_id="repo/cls",
        filename="model.pkl",
        local_dir=mock_settings.model_dir,
    )


@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_override_repo(mock_provider_cls: Mock, mock_settings: Mock):
    """ensure_weights uses overridden repository_id from weights config."""
    predictor_type = "detector"
    backend_type = "onnx"

    predictor_config = PredictorConfig(
        target="some.class.path",
        repository_id="repo/main",
        provider_name="huggingface",
        weights={"onnx": WeightDetails(filename="model.onnx")},
    )
    # This is the key change: the weights config can override the repo
    predictor_config.repository_id = "repo/onnx"

    weights_config = predictor_config.weights[backend_type]

    def get_config_side_effect(key):
        if key == f"predictors.{predictor_type}":
            return predictor_config
        if key == f"predictors.{predictor_type}.weights.{backend_type}":
            return weights_config
        return None

    mock_settings.get_config.side_effect = get_config_side_effect

    mock_provider = mock_provider_cls.return_value.get_provider.return_value

    wm = ModelWeightsManager(settings=mock_settings)
    wm.ensure_weights(predictor_type, backend_type)

    mock_provider.download_model_weights.assert_called_once_with(
        repo_id="repo/onnx",
        filename="model.onnx",
        local_dir=mock_settings.model_dir,
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
