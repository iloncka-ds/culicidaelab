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
        provider_service (ProviderService): The service for downloading artifacts.
    """

    def __init__(self, settings: Settings, provider_service: ProviderService):
        """Initializes the ModelWeightsManager."""
        self.settings = settings
        self.provider_service = provider_service

    def ensure_weights(self, model_type: str) -> Path:
        """Ensures model weights exist locally, downloading them if needed.

        This method checks for the local existence of a model's weights. If they
        are not found, it uses the provider service to download them based on
        the configuration associated with the given model type.

        Args:
            model_type (str): The type of the model for which to ensure weights,
                e.g., 'classifier', 'detector'.

        Returns:
            Path: The absolute, canonical path to the local model weights file.

        Raises:
            RuntimeError: If the weights cannot be downloaded or if the
                configuration for the provider is missing or invalid.
        """
        predictor_config = None
        try:
            predictor_config = self.settings.get_config(f"predictors.{model_type}")
            provider = self.provider_service.get_provider(predictor_config.provider_name)
            return provider.download_model_weights(model_type)
        except Exception as e:
            error_msg = f"Failed to download weights for '{model_type}': {str(e)}"
            if predictor_config:
                error_msg += f" with predictor config {predictor_config}"
            raise RuntimeError(error_msg) from e
