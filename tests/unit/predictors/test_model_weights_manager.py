import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call

from culicidaelab.predictors.model_weights_manager import ModelWeightsManager


@pytest.fixture
def mock_settings() -> Mock:
    """Fixture providing a mocked Settings instance."""
    return Mock()


def test_init(mock_settings: Mock):
    """ModelWeightsManager stores the provided settings instance."""
    wm = ModelWeightsManager(settings=mock_settings)
    assert wm.settings is mock_settings


@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_successful_download(mock_provider_cls: Mock, mock_settings: Mock):
    """ensure_weights orchestrates a successful download."""
    model_type = "classifier"
    expected_path = Path("/mock/path/to/model.pt")

    # Settings returns predictor config with provider_name
    predictor_cfg = Mock()
    predictor_cfg.provider_name = "mock_provider"
    mock_settings.get_config.return_value = predictor_cfg

    # ProviderService mock setup
    mock_provider_instance = mock_provider_cls.return_value
    mock_provider = mock_provider_instance.get_provider.return_value
    mock_provider.download_model_weights.return_value = expected_path

    # Patch instantiate_from_config to return our mock_provider
    mock_settings.instantiate_from_config.return_value = mock_provider

    wm = ModelWeightsManager(settings=mock_settings)
    result = wm.ensure_weights(model_type)

    # Assertions – confirm configuration lookup happened and the expected path is returned.
    assert call(f"predictors.{model_type}") in mock_settings.get_config.call_args_list
    assert result == expected_path


@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_raises_runtime_error_on_provider_failure(mock_provider_cls: Mock, mock_settings: Mock):
    """ensure_weights wraps provider failures in RuntimeError."""
    model_type = "detector"
    predictor_cfg = Mock()
    predictor_cfg.provider_name = "failing_provider"
    mock_settings.get_config.return_value = predictor_cfg

    # Simulate provider download failure which should be wrapped by ModelWeightsManager
    mock_provider = mock_provider_cls.return_value.get_provider.return_value
    mock_provider.download_model_weights.side_effect = ConnectionError("network down")
    mock_settings.instantiate_from_config.return_value = mock_provider

    wm = ModelWeightsManager(settings=mock_settings)
    with pytest.raises(RuntimeError):
        wm.ensure_weights(model_type)


@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_raises_runtime_error_on_missing_provider_name(mock_provider_cls: Mock, mock_settings: Mock):
    """Missing provider_name in config results in RuntimeError."""
    model_type = "invalid_model"

    cfg = Mock()
    if hasattr(cfg, "provider_name"):
        del cfg.provider_name
    mock_settings.get_config.return_value = cfg

    wm = ModelWeightsManager(settings=mock_settings)
    with pytest.raises(RuntimeError):
        wm.ensure_weights(model_type)


@patch("culicidaelab.predictors.model_weights_manager.ProviderService")
def test_ensure_weights_raises_runtime_error_on_none_config(mock_provider_cls: Mock, mock_settings: Mock):
    """None config should raise RuntimeError."""
    model_type = "missing_model"
    mock_settings.get_config.return_value = None

    wm = ModelWeightsManager(settings=mock_settings)
    with pytest.raises(RuntimeError):
        wm.ensure_weights(model_type)
