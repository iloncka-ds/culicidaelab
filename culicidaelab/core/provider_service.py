"""Service for managing data provider instances.

This module provides `ProviderService`, which is responsible for instantiating,
caching, and providing access to provider instances like `HuggingFaceProvider`.
"""

from culicidaelab.core.base_provider import BaseProvider
from culicidaelab.core.settings import Settings


class ProviderService:
    """Manages the instantiation and lifecycle of data providers.

    Args:
        settings (Settings): The main `Settings` object for the library.

    Attributes:
        _settings (Settings): The settings instance.
        _providers (dict[str, BaseProvider]): A cache of instantiated providers.
    """

    def __init__(self, settings: Settings):
        """Initializes the ProviderService."""
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

            if not self._settings.get_config(provider_path):
                raise ValueError(
                    f"Provider '{provider_name}' not found in configuration.",
                )

            # Use `instantiate_from_config` from `Settings`
            provider_instance = self._settings.instantiate_from_config(
                provider_path,
                settings=self._settings,
            )
            self._providers[provider_name] = provider_instance

        return self._providers[provider_name]
