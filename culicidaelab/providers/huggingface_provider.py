"""HuggingFace Dataset Provider implementation."""

from __future__ import annotations

from typing import Any, cast
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download
from datasets import load_dataset  # type: ignore[import-untyped]

from ..core.base_provider import BaseProvider
from ..core.config_manager import ConfigManager
from ..core.config_models import ProviderConfig


class HuggingFaceProvider(BaseProvider):
    """Provider for downloading and managing HuggingFace datasets."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize HuggingFace provider with configuration.

        Args:
            config_manager: Configuration manager instance
        """
        super().__init__()
        self.provider_name = "huggingface"
        self.config_manager = config_manager
        self.api_key: str | None = None
        self.provider_url: str = "https://huggingface.co/api/datasets/{dataset_name}"

        # Get provider config with type hint
        provider_config: ProviderConfig | dict[str, Any] = getattr(
            self.config_manager.config,
            "providers",
            {},
        ).get("huggingface", {})

        # Handle both dict and ProviderConfig types
        if isinstance(provider_config, ProviderConfig):
            self.api_key = provider_config.api_key
            if provider_config.api_key:
                self.api_key = provider_config.api_key
            if hasattr(provider_config, "provider_url"):
                self.provider_url = provider_config.provider_url
        else:  # dict case
            self.api_key = provider_config.get("api_key")
            if "provider_url" in provider_config:
                self.provider_url = provider_config["provider_url"]

        if not self.api_key:
            raise ValueError(
                "HuggingFace API key not found. " "Please set it in the configuration or environment variables.",
            )

    def get_dataset_metadata(self, dataset_name: str) -> dict[str, Any]:
        """Get metadata for a specific dataset from HuggingFace.

        Args:
            dataset_name: Name of the dataset to get metadata for

        Returns:
            Dataset metadata as a dictionary

        Raises:
            ValueError: If API key is not set
            requests.RequestException: If the request fails
        """
        if not self.api_key:
            raise ValueError("HuggingFace API key is not set")

        url = self.provider_url.format(dataset_name=dataset_name)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to fetch dataset metadata for {dataset_name}: {str(e)}",
            ) from e

    def download_dataset(
        self,
        dataset_name: str,
        save_dir: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """Download a dataset from HuggingFace.

        Args:
            dataset_name: Name of the dataset to download
            save_dir: Directory to save the dataset. Defaults to ./data/{dataset_name}
            **kwargs: Additional arguments to pass to the download method

        Returns:
            Path to the downloaded dataset

        Raises:
            RuntimeError: If download fails
        """
        if not self.api_key:
            raise ValueError("HuggingFace API key is not set")

        save_path = Path(save_dir) if save_dir else Path.cwd() / "data" / dataset_name
        save_path = save_path.resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            dataset = load_dataset(
                dataset_name,
                token=self.api_key,  # Updated parameter name for newer versions
                **kwargs,
            )
            dataset.save_to_disk(str(save_path))  # type: ignore[union-attr]
            return save_path
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {dataset_name}: {str(e)}") from e

    def download_model_weights(self, model_type: str, *args: Any, **kwargs: Any) -> Path:
        """
        Get model weights path.

        Args:
            model_type: Type of model ('detection', 'segmentation', or 'classification')

        Returns:
            Path: Path to the model weights file

        Raises:
            ValueError: If model type is not found in config
            RuntimeError: If download fails
        """
        try:
            config = self.config_manager.config
            if not hasattr(config, "models") or model_type not in config.models:
                raise ValueError(f"Unknown model type: {model_type}")

            # Use getattr with type ignore since we've checked models exists
            model_config = config.models[model_type]  # type: ignore[attr-defined]
            local_path = Path(str(model_config.local_path)).resolve()

            if local_path.exists():
                return local_path

            # If file doesn't exist, download it
            print(f"Model weights not found at {local_path}. Would you like to download them? (y/n)")
            if input().lower() != "y":
                raise FileNotFoundError(
                    f"Model weights not found and download was declined. "
                    f"Please place the weights file at: {local_path}",
                )

            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                hf_hub_download(
                    repo_id=model_config.remote_repo,  # type: ignore[attr-defined]
                    filename=model_config.remote_file,  # type: ignore[attr-defined]
                    local_dir=str(local_path.parent),
                    local_dir_use_symlinks=False,
                    token=self.api_key,
                )
                return local_path
            except Exception as e:
                raise RuntimeError(f"Failed to download model weights: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Error in download_model_weights: {str(e)}") from e

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name
