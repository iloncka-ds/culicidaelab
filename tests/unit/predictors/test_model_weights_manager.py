import pytest
from pathlib import Path
from unittest.mock import Mock

from culicidaelab.predictors.model_weights_manager import ModelWeightsManager


@pytest.fixture
def mock_settings() -> Mock:
    """Fixture for a mocked Settings object."""
    settings = Mock()
    mock_predictor_config = Mock()
    mock_predictor_config.provider_name = "mock_huggingface_provider"
    settings.get_config.return_value = mock_predictor_config
    return settings


@pytest.fixture
def mock_provider_service() -> Mock:
    """Fixture for a mocked ProviderService object."""
    provider_service = Mock()
    mock_provider = Mock()
    mock_provider.download_model_weights.return_value = Path("/mock/path/to/model.pt")
    provider_service.get_provider.return_value = mock_provider
    return provider_service


@pytest.fixture
def weights_manager(
    mock_settings: Mock,
    mock_provider_service: Mock,
) -> ModelWeightsManager:
    """Fixture to create a ModelWeightsManager with mocked dependencies."""
    return ModelWeightsManager(
        settings=mock_settings,
        provider_service=mock_provider_service,
    )


def test_init(
    weights_manager: ModelWeightsManager,
    mock_settings: Mock,
    mock_provider_service: Mock,
):
    """Test the initialization of the ModelWeightsManager."""
    assert isinstance(weights_manager, ModelWeightsManager)
    assert weights_manager.settings is mock_settings
    assert weights_manager.provider_service is mock_provider_service


def test_ensure_weights_successful_download():
    """
    Test that ensure_weights correctly orchestrates the process
    of getting model weights when everything is successful.
    """
    model_type = "classifier"
    expected_path = Path("/mock/path/to/model.pt")

    mock_settings = Mock()
    mock_predictor_config = Mock()
    mock_predictor_config.provider_name = "mock_huggingface_provider"
    mock_settings.get_config.return_value = mock_predictor_config

    mock_provider = Mock()
    mock_provider.download_model_weights.return_value = expected_path

    mock_provider_service = Mock()
    mock_provider_service.get_provider.return_value = mock_provider

    weights_manager = ModelWeightsManager(
        settings=mock_settings,
        provider_service=mock_provider_service,
    )

    result_path = weights_manager.ensure_weights(model_type)

    mock_settings.get_config.assert_called_once_with(f"predictors.{model_type}")
    mock_provider_service.get_provider.assert_called_once_with(
        mock_predictor_config.provider_name,
    )
    mock_provider.download_model_weights.assert_called_once_with(model_type)
    assert result_path == expected_path


def test_ensure_weights_raises_runtime_error_on_provider_failure(
    weights_manager: ModelWeightsManager,
    mock_provider_service: Mock,
):
    """
    Test that ensure_weights catches exceptions from its provider
    and raises a single, informative RuntimeError.
    """
    model_type = "detector"
    original_exception = ConnectionError("Could not connect to Hugging Face Hub")

    mock_provider = mock_provider_service.get_provider.return_value
    mock_provider.download_model_weights.side_effect = original_exception

    with pytest.raises(RuntimeError) as excinfo:
        weights_manager.ensure_weights(model_type)

    assert f"Failed to download weights for '{model_type}'" in str(excinfo.value)
    assert str(original_exception) in str(excinfo.value)
    assert excinfo.value.__cause__ is original_exception


def test_ensure_weights_raises_runtime_error_on_config_failure(
    weights_manager: ModelWeightsManager,
    mock_settings: Mock,
):
    """
    Test that ensure_weights raises a RuntimeError if the config lookup fails.
    """
    model_type = "invalid_type"
    original_exception = KeyError(f"Config not found for predictors.{model_type}")

    mock_settings.get_config.side_effect = original_exception

    with pytest.raises(RuntimeError) as excinfo:
        weights_manager.ensure_weights(model_type)

    assert f"Failed to download weights for '{model_type}'" in str(excinfo.value)
    assert str(original_exception) in str(excinfo.value)
    assert excinfo.value.__cause__ is original_exception
