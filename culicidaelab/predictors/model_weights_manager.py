"""
Model weights management module for CulicidaeLab.
"""

from __future__ import annotations
from pathlib import Path


from culicidaelab.core.settings import Settings
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol


class ModelWeightsManager(WeightsManagerProtocol):
    """Manages ensuring model weights are available locally, downloading if necessary."""

    def __init__(self, settings: Settings, provider_service: ProviderService):
        """Initialize the model weights manager."""
        self.settings = settings
        self.provider_service = provider_service

    def ensure_weights(self, model_type: str) -> Path:
        """
        Ensures weights for a model type exist locally, downloading if not.
        This method correctly handles and resolves symbolic links.

        Args:
            model_type: The key for the predictor (e.g., 'classifier').

        Returns:
            The absolute, canonical Path to the validated, existing model file.
        """
        # Initialize predictor_config to None before the try block
        predictor_config = None
        try:
            predictor_config = self.settings.get_config(f"predictors.{model_type}")
            provider = self.provider_service.get_provider(predictor_config.provider)
            return provider.download_model_weights(model_type)
        except Exception as e:
            # Construct a detailed error message without causing a new error
            error_msg = f"Failed to download weights for '{model_type}': {str(e)}"
            if predictor_config:
                error_msg += f" with predictor config {predictor_config}"
            raise RuntimeError(error_msg) from e
