from .settings import Settings
from .base_provider import BaseProvider


class ProviderService:
    """Manages the instantiation and lifecycle of data providers."""

    def __init__(self, settings: Settings):
        """Initializes the ProviderService.

        Args:
            settings: The main Settings object for the library.
        """
        self._settings = settings
        self._providers: dict[str, BaseProvider] = {}  # Cache instantiated providers

    def get_provider(self, provider_name: str) -> BaseProvider:
        """
        Retrieves an instantiated provider by its name.

        It looks up the provider's configuration, instantiates it if it hasn't
        been already, and caches it for future calls.

        Args:
            provider_name: The name of the provider (e.g., 'huggingface').

        Returns:
            An instance of a class that inherits from BaseProvider.

        Raises:
            ValueError: If the provider is not found in the configuration.
        """
        if provider_name not in self._providers:
            # Define the string path to the provider's configuration.
            provider_path = f"providers.{provider_name}"

            # First, check if the config exists to provide a clearer error message.
            if not self._settings.get_config(provider_path):
                raise ValueError(f"Provider '{provider_name}' not found in configuration.")

            self._providers[provider_name] = self._settings.instantiate_from_config(
                provider_path,
                settings=self._settings,
            )

        return self._providers[provider_name]
