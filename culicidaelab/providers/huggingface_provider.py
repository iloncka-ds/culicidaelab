"""HuggingFace Provider for managing datasets and models.

This module provides the `HuggingFaceProvider` class, which is a concrete
implementation of `BaseProvider`. It handles downloading datasets and
model weights from the Hugging Face Hub, as well as loading them
from a local disk cache.
"""

from __future__ import annotations

# Standard library
import shutil
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import requests
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

# Internal imports
from culicidaelab.core.base_provider import BaseProvider
from culicidaelab.core.settings import Settings


class HuggingFaceProvider(BaseProvider):
    """Provider for downloading and managing HuggingFace datasets and models.

    This class interfaces with the Hugging Face Hub to fetch dataset metadata,
    download full datasets or specific splits, and download model weights. It uses
    the core settings object for path resolution and API key access.

    Attributes:
        provider_name (str): The name of the provider, "huggingface".
        settings (Settings): The main Settings object for the library.
        dataset_url (str): The base URL for fetching Hugging Face dataset metadata.
        api_key (str | None): The Hugging Face API key, if provided.
    """

    def __init__(self, settings: Settings, dataset_url: str, **kwargs: Any) -> None:
        """Initializes the HuggingFace provider.

        This constructor is called by the `ProviderService`, which injects the
        global `settings` object and unpacks the specific provider's configuration
        (e.g., `dataset_url`) as keyword arguments.

        Args:
            settings (Settings): The main Settings object for the library.
            dataset_url (str): The base URL for fetching Hugging Face dataset metadata.
            **kwargs (Any): Catches other config parameters (e.g., `api_key`).
        """
        super().__init__()
        self.provider_name = "huggingface"
        self.settings = settings
        self.dataset_url = dataset_url
        self.api_key: str | None = kwargs.get("api_key") or self.settings.get_api_key(
            self.provider_name,
        )

    def download_dataset(
        self,
        dataset_name: str,
        save_dir: str | None = None,
        split: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """Downloads a dataset from HuggingFace.

        Args:
            dataset_name (str): Name of the dataset to download (e.g., "segmentation").
            save_dir (str | None, optional): Directory to save the dataset.
                Defaults to None, using the path from settings.
            split (str | None, optional): Dataset split to download (e.g., "train").
                Defaults to None.
            *args (Any): Additional positional arguments to pass to `load_dataset`.
            **kwargs (Any): Additional keyword arguments to pass to `load_dataset`.

        Returns:
            Path: The path to the downloaded dataset.

        Raises:
            ValueError: If the configuration is missing the `repository` ID.
            RuntimeError: If the download fails.
        """
        save_path = self.settings.get_dataset_path(dataset_name)
        if save_dir:
            save_path = Path(save_dir)
        dataset_config = self.settings.get_config(f"datasets.{dataset_name}")

        repo_id = dataset_config.repository
        if not repo_id:
            raise ValueError(
                f"Configuration for dataset '{dataset_name}' is missing the 'repository' (repository ID).",
            )

        try:
            if self.api_key:
                dataset = load_dataset(repo_id, split=split, token=self.api_key, **kwargs)
            else:
                dataset = load_dataset(repo_id, split=split, **kwargs)

            if split:
                save_path = save_path / split

            if save_path.exists() and save_path.is_dir():
                print(f"Existing dataset found at '{save_path}'. Removing it for a clean download.")
                shutil.rmtree(save_path)

            dataset.save_to_disk(str(save_path))
            if isinstance(dataset, Dataset):
                dataset.cleanup_cache_files()

            return save_path
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {repo_id}: {str(e)}") from e

    def download_model_weights(self, model_type: str, *args: Any, **kwargs: Any) -> Path:
        """Downloads and caches model weights from the HuggingFace Hub.

        Checks if the weights exist locally. If not, it downloads them
        from the repository specified in the configuration and saves them
        to the appropriate directory.

        Args:
            model_type (str): The type of model ('detector', 'segmenter', or 'classifier').
            *args (Any): Additional positional arguments (unused).
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            Path: The path to the model weights file.

        Raises:
            ValueError: If the model type is not found in config or if `repository_id`
                or `filename` are missing.
            RuntimeError: If the download fails for any reason.
            NotADirectoryError: If the destination directory could not be created.
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

            dest_dir.mkdir(parents=True, exist_ok=True)
            if not dest_dir.is_dir():
                raise NotADirectoryError(f"Failed to create directory: {dest_dir}")

            downloaded_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.settings.cache_dir / "huggingface",
                local_dir=str(local_path.parent),
            )
            print(f"Downloaded weights to: {downloaded_path_str}")

            return local_path

        except Exception as e:
            if local_path.exists():
                local_path.unlink()

            dir_status = "exists" if dest_dir.exists() else "missing"
            dir_type = "directory" if dest_dir.is_dir() else "not-a-directory"
            raise RuntimeError(
                f"Failed to download weights for '{model_type}' to {local_path}. "
                f"Directory status: {dir_status} ({dir_type}). Error: {e}",
            ) from e

    def get_dataset_metadata(self, dataset_name: str) -> dict[str, Any]:
        """Gets metadata for a specific dataset from HuggingFace.

        Args:
            dataset_name (str): The name of the dataset to get metadata for.

        Returns:
            dict[str, Any]: The dataset metadata as a dictionary.

        Raises:
            requests.RequestException: If the HTTP request fails.
        """
        url = self.dataset_url.format(dataset_name=dataset_name)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            response = requests.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to fetch dataset metadata for {dataset_name}: {str(e)}",
            ) from e

    def get_provider_name(self) -> str:
        """Returns the provider's name.

        Returns:
            str: The name of the provider, "huggingface".
        """
        return self.provider_name

    def load_dataset(self, dataset_path: str | Path, split: str | None = None, **kwargs) -> Any:
        """Loads a dataset from disk.

        This method attempts to load a dataset from the specified path. If a `split`
        name is provided and a corresponding subdirectory exists, it will load
        the split from that subdirectory. Otherwise, it loads the entire dataset
        from the base path.

        Args:
            dataset_path (str | Path): The local path to the dataset,
                typically returned by `download_dataset`.
            split (str | None, optional): The specific split to load (e.g., "train",
                "validation", "test"). If a subdirectory with this name
                exists, it will be loaded. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the
                `datasets.load_from_disk` function.

        Returns:
            Any: The loaded dataset, typically a `datasets.Dataset` or
                `datasets.DatasetDict` object.
        """
        path_to_load = Path(dataset_path)
        if split and (path_to_load / split).exists():
            return load_from_disk(str(path_to_load / split), **kwargs)
        return load_from_disk(str(dataset_path), **kwargs)
