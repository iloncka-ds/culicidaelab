import pytest
from unittest.mock import Mock
from typing import Any

from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.base_provider import BaseProvider


class MockProvider(BaseProvider):
    def download_dataset(self, *args: Any, **kwargs: Any):
        pass

    def download_model_weights(self, *args: Any, **kwargs: Any):
        pass

    def get_provider_name(self) -> str:
        return "mock_provider"

    def load_dataset(self, *args: Any, **kwargs: Any):
        pass


@pytest.fixture
def mock_settings():
    """Creates a mock Settings object."""
    settings = Mock()
    settings.get_config = Mock()
    settings.instantiate_from_config = Mock()
    return settings


def test_get_provider_instantiates_on_first_call(mock_settings):
    provider_name = "huggingface"
    mock_provider_instance = MockProvider()

    mock_settings.get_config.return_value = {"target": "some.path.Provider"}
    mock_settings.instantiate_from_config.return_value = mock_provider_instance

    service = ProviderService(mock_settings)
    provider = service.get_provider(provider_name)

    mock_settings.get_config.assert_called_once_with(f"providers.{provider_name}")
    mock_settings.instantiate_from_config.assert_called_once_with(
        f"providers.{provider_name}",
        settings=mock_settings,
    )
    assert provider == mock_provider_instance


def test_get_provider_returns_cached_instance(mock_settings):
    provider_name = "huggingface"
    mock_provider_instance = MockProvider()

    mock_settings.get_config.return_value = {"target": "some.path.Provider"}
    mock_settings.instantiate_from_config.return_value = mock_provider_instance

    service = ProviderService(mock_settings)

    provider1 = service.get_provider(provider_name)

    provider2 = service.get_provider(provider_name)

    mock_settings.instantiate_from_config.assert_called_once()
    assert provider1 is provider2


def test_get_provider_raises_error_if_not_configured(mock_settings):
    provider_name = "non_existent_provider"

    mock_settings.get_config.return_value = None

    service = ProviderService(mock_settings)

    with pytest.raises(
        ValueError,
        match=f"Provider '{provider_name}' not found in configuration.",
    ):
        service.get_provider(provider_name)

    mock_settings.get_config.assert_called_once_with(f"providers.{provider_name}")
    mock_settings.instantiate_from_config.assert_not_called()
