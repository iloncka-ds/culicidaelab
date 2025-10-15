"""Service for managing data provider instances.

This module provides `ProviderService`, which is responsible for instantiating,
caching, and providing access to provider instances like `HuggingFaceProvider`.

Example:
    >>> from culicidaelab.core.settings import Settings
    >>> from culicidaelab.core.provider_service import ProviderService
    >>> settings = Settings()
    >>> provider_service = ProviderService(settings)
    >>> hf_provider = provider_service.get_provider("huggingface")
"""

from culicidaelab.core.base_provider import BaseProvider
from culicidaelab.core.settings import Settings


class ProviderService:
    """Manages the instantiation and lifecycle of data providers.

    This service acts as a factory and cache for provider instances, ensuring that
    each provider is a singleton within the application context.

    Attributes:
        _settings (Settings): The settings instance.
        _providers (dict[str, BaseProvider]): A cache of instantiated providers,
            keyed by provider name.
    """

    def __init__(self, settings: Settings):
        """Initializes the ProviderService.

        Args:
            settings (Settings): The main `Settings` object for the library.
        """
        self._settings = settings
        self._providers: dict[str, BaseProvider] = {}

    def get_provider(self, provider_name: str) -> BaseProvider:
        """Retrieves an instantiated provider by its name.

        It looks up the provider's configuration, instantiates it if it hasn't
        been already, and caches it for future calls.

        Args:
            provider_name (str): The name of the provider (e.g., 'huggingface').

        Returns:
            BaseProvider: An instance of a class that inherits from `BaseProvider`.

        Raises:
            ValueError: If the provider is not found in the configuration.
        """
        if provider_name not in self._providers:
            provider_path = f"providers.{provider_name}"

            provider_config = self._settings.get_config(provider_path)
            if not provider_config:
                raise ValueError(
                    f"Provider '{provider_name}' not found in configuration.",
                )

            # Use `instantiate_from_config` from `Settings`
            provider_instance = self._settings.instantiate_from_config(
                provider_path,
            )
            if not isinstance(provider_instance, BaseProvider):
                raise TypeError(
                    f"Instantiated provider '{provider_name}' is not a valid BaseProvider",
                )

            self._providers[provider_name] = provider_instance

        return self._providers[provider_name]
