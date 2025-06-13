"""HuggingFace Dataset Provider implementation."""

from typing import Any
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset

from ..core.base_provider import BaseProvider
from ..core.config_manager import ConfigManager


class HuggingFaceProvider(BaseProvider):
    """Provider for downloading and managing HuggingFace datasets."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize HuggingFace provider with configuration.

        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.provider_name = "huggingface"
        self.config_manager = config_manager

        provider_config = self.config_manager.get_provider_config("huggingface")
        self.api_key = provider_config.get("api_key")
        self.provider_url = provider_config.get("provider_url")

        if not self.api_key:
            raise ValueError("HuggingFace API key not found in environment variables")

    def get_dataset_metadata(self, dataset_name: str) -> dict[str, Any]:
        """Get metadata for a specific dataset from HuggingFace.

        Args:
            dataset_name (str): Name of the dataset to get metadata for

        Returns:
            Dict[str, Any]: Dataset metadata
        """
        url = self.provider_url.format(dataset_name=dataset_name)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def download_dataset(
        self,
        dataset_name: str,
        save_dir: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Download a dataset from HuggingFace.

        Args:
            dataset_name (str): Name of the dataset to download
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to None.
            **kwargs: Additional arguments to pass to the download method

        Returns:
            Path: Path to the downloaded dataset
        """

        if save_dir is None:
            save_dir = Path.cwd() / "data" / dataset_name
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(
            dataset_name,
            use_auth_token=self.api_key,
            **kwargs,
        )

        dataset.save_to_disk(save_dir)

        return Path(save_dir)

    def download_model_weights(self, model_type: str) -> Path:
        """
        Get model weights path.

        Args:
            model_type: Type of model ('detection', 'segmentation', or 'classification')

        Returns:
            Path: Path to the model weights file
        """
        config = self.config_manager.get_config()

        if not hasattr(config, "models") or model_type not in config.models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_config = config.models[model_type]
        local_path = Path(model_config.local_path).resolve()

        if local_path.exists():
            return Path(local_path)

        print(f"\nModel weights for {model_type} not found at: {local_path}")
        response = input(f"Would you like to download them from {model_config.remote_repo}? (y/n): ")

        if response.lower() != "y":
            raise FileNotFoundError(
                f"Model weights not found and download was declined. "
                f"Please place the weights file at: {local_path}",
            )

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            hf_hub_download(
                repo_id=model_config.remote_repo,
                filename=model_config.remote_file,
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False,
                token=self.api_key,
            )
            return Path(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download model weights: {str(e)}")

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name
