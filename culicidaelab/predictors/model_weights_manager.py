"""Model weights management for CulicidaeLab predictors.

This module defines the ModelWeightsManager, a class responsible for ensuring
that the necessary model weight files are available locally. If the weights
are not present, it coordinates with a provider service (e.g., HuggingFace)
to download them.

Attributes:
    settings (Settings): The main settings object for the library.
    provider_service (ProviderService): The service used to access and
        download model weights from various providers.
"""

from __future__ import annotations

from pathlib import Path

from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.settings import Settings
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol


class ModelWeightsManager(WeightsManagerProtocol):
    """Manages the download and local availability of model weights.

    This class implements the WeightsManagerProtocol and serves as the bridge
    between a predictor and the provider service that can download model files.

    Args:
        settings (Settings): The application's global settings object.
    """

    def __init__(self, settings: Settings):
        """Initializes the ModelWeightsManager."""
        self.settings = settings
        self.provider_service = ProviderService(settings)

    def resolve_weights_path(self, predictor_type: str, backend_type: str) -> Path:
        """
        Ensures weights for a given predictor and backend type are available locally,
        downloading them if necessary, and returns the absolute path.
        """
        try:
            # Construct the config key to get the specific weights info
            weights_config_key = f"predictors.{predictor_type}.weights.{backend_type}"
            weights_config = self.settings.get_config(weights_config_key)

            # Get the main predictor config to find the repository_id and provider
            predictor_config = self.settings.get_config(f"predictors.{predictor_type}")

            # The repository can be overridden at the weights level
            repo_id = weights_config.get("repository_id", predictor_config.get("repository_id"))
            provider_name = predictor_config.get("provider_name", "huggingface")  # Default provider
            filename = weights_config.get("filename")

            if not all([repo_id, filename]):
                raise ValueError(f"Missing 'repository_id' or 'filename' for {weights_config_key}")

            provider = self.provider_service.get_provider(provider_name)

            # Assuming provider has a method to download a specific file
            # This change will require updating the provider's interface
            return provider.download_model_weights(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.settings.model_dir,
            )
        except Exception as e:
            error_msg = f"Failed to resolve weights for '{predictor_type}' with backend '{backend_type}': {e}"
            raise RuntimeError(error_msg) from e
