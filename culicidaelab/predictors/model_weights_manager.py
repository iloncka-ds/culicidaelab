"""
Model weights management module for CulicidaeLab.
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Optional

from culicidaelab.core.settings import Settings


class ModelWeightsManager:
    """Manages model weights downloading and access in a non-interactive way."""

    def __init__(self, settings: Settings):
        """Initialize the model weights manager."""
        self.settings = settings

    def get_weights_path(self, model_type: str) -> Path:
        """
        Get the absolute local path to the model weights file.

        This method resolves the path but does not check for existence or download.

        Args:
            model_type: The key for the predictor (e.g., 'classifier').

        Returns:
            The absolute Path to where the model file should be.

        Raises:
            ValueError: If the model_type or its configuration is not found.
        """
        predictor_config = self.settings.get_config(f"predictors.{model_type}")
        if not predictor_config:
            raise ValueError(f"Configuration for predictor '{model_type}' not found.")

        # Use the settings.weights_dir for robust, centralized path management
        return self.settings.weights_dir / predictor_config.model_path

    def download_weights(self, model_type: str, force: bool = False) -> Path:
        """
        Download weights from a Hugging Face repository if they don't exist locally.

        This is an explicit, user-initiated action.

        Args:
            model_type: The key for the predictor (e.g., 'classifier').
            force: If True, re-download the weights even if a local file exists.

        Returns:
            The absolute Path to the downloaded model file.

        Raises:
            ValueError: If remote repository information is missing in the config.
            Exception: If the download fails for any reason.
        """
        local_path = self.get_weights_path(model_type)

        if local_path.exists() and not force:
            print(f"Weights for '{model_type}' already exist at: {local_path}")
            return local_path

        predictor_config = self.settings.get_config(f"predictors.{model_type}")
        repo_id = predictor_config.repository_id
        filename = predictor_config.filename

        if not repo_id or not filename:
            raise ValueError(
                f"Cannot download weights for '{model_type}'. "
                f"Missing 'repository_id' or 'filename' in configuration."
            )

        print(f"Downloading weights for '{model_type}' from repo '{repo_id}'...")
        try:
            # Create the parent directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # hf_hub_download handles caching and temporary files gracefully
            downloaded_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.settings.cache_dir / "huggingface",
                force_download=force,
            )

            # Copy the file from the cache to the desired final location
            shutil.copy(downloaded_path_str, local_path)

            print(f"Successfully downloaded weights to: {local_path}")
            return local_path
        except Exception as e:
            # Clean up a potentially partial file on failure
            if local_path.exists():
                local_path.unlink()
            raise Exception(f"Failed to download weights for '{model_type}': {e}")