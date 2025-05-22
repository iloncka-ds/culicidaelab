"""
Model weights management module for CulicidaeLab.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

from culicidaelab.core.config_manager import ConfigurableComponent


class ModelWeightsManager(ConfigurableComponent):
    """Manages model weights downloading and access."""

    def __init__(self, config_manager):
        """Initialize the model weights manager."""
        super().__init__(config_manager)
        self._config = self.config_manager.get_config()

    def get_weights(self, model_type: str) -> str:
        """
        Get model weights, checking local path first and handling remote download.

        Args:
            model_type: Type of model ('detection', 'segmentation', or 'classification')

        Returns:
            str: Path to the model weights file
        """
        if model_type not in self._config.models.weights:
            raise ValueError(f"Unknown model type: {model_type}")

        config = self._config.models.weights[model_type]
        local_path = self._get_abs_path(config.local_path)

        # Check if weights exist locally
        if os.path.exists(local_path):
            return local_path

        # Ask permission to download
        print(f"\nModel weights for {model_type} not found at: {local_path}")
        response = input(f"Would you like to download them from {config.remote_repo}? (y/n): ")

        if response.lower() != "y":
            raise FileNotFoundError(
                f"Model weights not found and download was declined. "
                f"Please place the weights file at: {local_path}",
            )

        return self._download_weights(config, local_path)

    def _download_weights(self, config, local_path: str | Path) -> str:
        """Download weights from remote repository."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download weights
            downloaded_path = hf_hub_download(
                repo_id=config.remote_repo,
                filename=config.remote_file,
                local_dir=os.path.dirname(local_path),
            )

            # Move to the correct location if necessary
            if downloaded_path != local_path:
                shutil.move(downloaded_path, local_path)

            print(f"Successfully downloaded weights to: {local_path}")
            return local_path
        except Exception as e:
            raise Exception(f"Failed to download weights: {str(e)}")

    def _get_abs_path(self, path: str | Path) -> Path:
        """Convert relative path to absolute path."""
        path = Path(path)
        if not path.is_absolute():
            path = self._config.paths.root_dir / path
        return path
