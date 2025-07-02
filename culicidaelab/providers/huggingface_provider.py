"""HuggingFace Dataset Provider implementation."""

from __future__ import annotations

from typing import Any, cast
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk  # type: ignore[import-untyped]

from ..core.base_provider import BaseProvider
from ..core.settings import Settings


class HuggingFaceProvider(BaseProvider):
    """Provider for downloading and managing HuggingFace datasets."""

    def __init__(self, settings: Settings, dataset_url: str, **kwargs: Any) -> None:
        """
        Initialize HuggingFace provider.

        This constructor is called by the `ProviderService` which injects the
        global `settings` object and unpacks the specific provider's configuration
        (e.g., `dataset_url`) as keyword arguments.

        Args:
            settings: The main Settings object for the library.
            dataset_url: The base URL for fetching Hugging Face dataset metadata.
            **kwargs: Catches other config parameters (e.g., `api_key`).
        """
        super().__init__()
        self.provider_name = "huggingface"
        self.settings = settings
        self.dataset_url = dataset_url

        # The API key can be defined in the config file (passed in kwargs)
        # or loaded from environment variables as a fallback.
        self.api_key: str | None = kwargs.get("api_key") or self.settings.get_api_key(self.provider_name)

    def get_dataset_metadata(self, dataset_name: str) -> dict[str, Any]:
        """Get metadata for a specific dataset from HuggingFace.

        Args:
            dataset_name: Name of the dataset to get metadata for
            split: Optional split to get metadata for
        Returns:
            Dataset metadata as a dictionary

        Raises:
            requests.RequestException: If the request fails
        """

        url = self.dataset_url.format(dataset_name=dataset_name)
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        else:
            headers = {}

        try:
            response = requests.get(url, headers=headers, timeout=10.0)
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
        split: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Download a dataset from HuggingFace.

        Args:
            dataset_name: Type of the dataset to download ("segmentation", "classification", "detection")
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to None.
             **kwargs: Additional arguments to pass to the download method

        Returns:
            Path to the downloaded dataset

        Raises:
            RuntimeError: If download fails
        """

        save_path = self.settings.get_dataset_path(dataset_name)
        if save_dir:
            save_path = Path(save_dir)
        dataset_config = self.settings.get_config(f"datasets.{dataset_name}")

        # The config model uses 'path' for the repository ID.
        repo_id = dataset_config.repository
        if not repo_id:
            raise ValueError(f"Configuration for dataset '{dataset_name}' is missing the 'path' (repository ID).")

        try:
            if self.api_key:
                dataset = load_dataset(
                    repo_id,
                    split=split,
                    token=self.api_key,
                    **kwargs,
                )
            else:
                dataset = load_dataset(
                    repo_id,
                    split=split,
                    **kwargs,
                )
            if split:
                save_path = save_path / split
            dataset.save_to_disk(str(save_path))  # type: ignore[union-attr]
            dataset.cleanup_cache_files()
            return save_path
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {repo_id}: {str(e)}") from e

    def load_dataset(self, dataset_path: str | Path, split: str | None = None, **kwargs) -> Any:
        # if split:
        #     path = Path(path) / split
        return load_from_disk(str(dataset_path), **kwargs)

    def download_model_weights(self, model_type: str, *args: Any, **kwargs: Any) -> Path:
        """
        Get model weights path.

        Args:
            model_type: Type of model ('detector', 'segmenter', or 'classifier')

        Returns:
            Path: Path to the model weights file

        Raises:
            ValueError: If model type is not found in config
            RuntimeError: If download fails
        """

        local_path = self.settings.get_model_weights_path(model_type).resolve()

        if local_path.exists():
            if local_path.is_symlink():
                try:
                    real_path = local_path.resolve(strict=True)
                    print(f"Symlink found at {local_path}, resolved to real file: {real_path}")
                    return real_path
                except FileNotFoundError:
                    print(f"Warning: Broken symlink found at {local_path}. It will be removed.")
                    local_path.unlink()
            else:
                print(f"Weights file found at: {local_path}")
                return local_path

        # If we get here, the path either does not exist or was a broken symlink.
        # We must now download the file.
        print(f"Model weights for '{model_type}' not found. Attempting to download...")

        predictor_config = self.settings.get_config(f"predictors.{model_type}")
        repo_id = predictor_config.repository_id
        filename = predictor_config.filename

        if not repo_id or not filename:
            raise ValueError(
                f"Cannot download weights for '{model_type}'. "
                f"Configuration is missing 'repository_id' or 'filename'. "
                f"Please place the file manually at: {local_path}",
            )

        try:
            dest_dir = local_path.parent.resolve()
            print(f"Ensuring destination directory exists: {dest_dir}")

            # Robust directory creation with validation
            dest_dir.mkdir(parents=True, exist_ok=True)
            if not dest_dir.is_dir():
                raise NotADirectoryError(f"Failed to create directory: {dest_dir}")

            # Download the file to the Hugging Face cache
            downloaded_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.settings.cache_dir / "huggingface",
                local_dir=str(local_path.parent),
            )
            print(f"Downloaded weights to: {downloaded_path_str}")

            return local_path

        except Exception as e:
            # Clean up a potentially partial file on failure
            if local_path.exists():
                local_path.unlink()
            # Include directory info in error message
            dir_status = "exists" if dest_dir.exists() else "missing"
            dir_type = "directory" if dest_dir.is_dir() else "not-a-directory"
            raise RuntimeError(
                f"Failed to download weights for '{model_type}' to {local_path}. "
                f"Directory status: {dir_status} ({dir_type}). Error: {e}",
            ) from e

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name
