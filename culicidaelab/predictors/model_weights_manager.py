"""Model weights management for CulicidaeLab predictors.

This module defines the ModelWeightsManager, a class responsible for ensuring
that the necessary model weight files are available locally. If the weights
are not present, it coordinates with a provider service (e.g., HuggingFace)
to download them.
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

    Attributes:
        settings (Settings): The application's global settings object.
        provider_service (ProviderService): The service used to access and
            download model weights from various providers.
    """

    def __init__(self, settings: Settings):
        """Initializes the ModelWeightsManager.

        Args:
            settings: The application's global settings object.
        """
        self.settings = settings
        self.provider_service = ProviderService(settings)

    def ensure_weights(self, predictor_type: str, backend_type: str) -> Path:
        """Ensures weights for a given predictor and backend are available.

        This method checks if the model weights for the specified predictor and
        backend type exist locally. If they don't, it downloads them using the
        provider service.

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.predictors import ModelWeightsManager
            >>> # This example assumes you have a configured settings object
            >>> settings = Settings()
            >>> manager = ModelWeightsManager(settings)
            >>> weights_path = manager.ensure_weights("classifier", "torch")
            >>> print(weights_path.exists())
            True

        Args:
            predictor_type: The type of predictor (e.g., 'classifier').
            backend_type: The type of backend (e.g., 'torch', 'onnx').

        Returns:
            The absolute path to the local model weights file.

        Raises:
            RuntimeError: If the weights cannot be resolved or downloaded.
            ValueError: If the configuration for the weights is missing
                'repository_id' or 'filename'.
        """

        try:
            local_path = self.settings.construct_weights_path(
                predictor_type=predictor_type,
                backend=backend_type,
            )

            if local_path.exists():
                return local_path

            predictor_config = self.settings.get_config(f"predictors.{predictor_type}")
            # Construct the config key to get the specific weights info
            weights_config_key = f"predictors.{predictor_type}.weights.{backend_type}"
            weights_config = self.settings.get_config(weights_config_key)

            # The repository can be overridden at the weights level
            repo_id = predictor_config.repository_id
            filename = weights_config.filename

            if not all([repo_id, filename]):
                raise ValueError(f"Missing 'repository_id' or 'filename' for {weights_config_key}")

            provider_name = predictor_config.provider_name or "huggingface"  # Default provider
            provider = self.provider_service.get_provider(provider_name)

            # Assuming provider has a method to download a specific file
            return provider.download_model_weights(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_path.parent,
            )

        except Exception as e:
            error_msg = f"Failed to resolve weights for '{predictor_type}' with backend '{backend_type}': {e}"
            raise RuntimeError(error_msg) from e
