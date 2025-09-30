"""HuggingFace Provider for managing datasets and models.

This module provides the `HuggingFaceProvider` class, which is a concrete
implementation of `BaseProvider`. It handles downloading datasets and
model weights from the HuggingFace Hub, as well as loading them
from a local disk cache.
"""

from __future__ import annotations

# Standard library
import shutil
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import requests
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)
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
        save_dir: Path | None = None,
        config_name: str | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Downloads a dataset from HuggingFace.

        Args:
            dataset_name (str): Name of the dataset to download (e.g., "segmentation").
            save_dir (Path | None, optional): Directory to save the dataset.
                Defaults to None, using the path from settings.
            config_name (str | None, optional): Name of the dataset configuration.
                Defaults to None.
            split (str | None, optional): Dataset split to download (e.g., "train").
                Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to `load_dataset`.

        Returns:
            Path: The path to the downloaded dataset.

        Raises:
            ValueError: If the configuration is missing the `repository` ID.
            RuntimeError: If the download fails.
        """
        save_path = self.settings.get_dataset_path(dataset_name)
        cache_path = str(self.settings.cache_dir / dataset_name)
        if save_dir:
            save_path = save_dir
        dataset_config = self.settings.get_config(f"datasets.{dataset_name}")

        repo_id = dataset_config.repository
        if not repo_id:
            raise ValueError(
                f"Configuration for dataset '{dataset_name}' is missing the 'repository' (repository ID).",
            )

        try:
            if self.api_key:
                downloaded_object = load_dataset(
                    repo_id,
                    name=config_name,
                    split=split,
                    token=self.api_key,
                    cache_dir=cache_path,
                    **kwargs,
                )
            else:
                downloaded_object = load_dataset(
                    repo_id,
                    name=config_name,
                    split=split,
                    cache_dir=cache_path,
                    **kwargs,
                )

            saveable_dataset = None
            if isinstance(downloaded_object, (IterableDataset, IterableDatasetDict)):
                if isinstance(downloaded_object, IterableDataset):
                    saveable_dataset = Dataset.from_list(list(downloaded_object))
                else:
                    materialized_splits = {s_name: list(s_data) for s_name, s_data in downloaded_object.items()}
                    saveable_dataset = DatasetDict(
                        {s_name: Dataset.from_list(data) for s_name, data in materialized_splits.items()},
                    )
            else:
                saveable_dataset = downloaded_object

            if Path(save_path).exists() and Path(save_path).is_dir():
                shutil.rmtree(save_path)

            save_path.mkdir(parents=True, exist_ok=True)

            saveable_dataset.save_to_disk(str(save_path))

            shutil.rmtree(cache_path, ignore_errors=True)

            return save_path

        except Exception as e:
            if Path(save_path).exists() and Path(save_path).is_dir():
                shutil.rmtree(save_path)
            raise RuntimeError(f"Failed to download dataset {repo_id}: {str(e)}") from e

    def download_model_weights(self, repo_id: str, filename: str, local_dir: Path) -> Path:
        """Downloads and caches a specific model weights file from the HuggingFace Hub.

        Checks if the file exists locally. If not, it downloads it from the specified
        repository and saves it to the given local directory.

        Args:
            repo_id (str): The Hugging Face repository ID (e.g., 'user/repo-name').
            filename (str): The specific weights file to download (e.g., 'model.onnx').
            local_dir (Path): The local directory to save the file in.

        Returns:
            Path: The absolute path to the downloaded model weights file.

        Raises:
            ValueError: If `repo_id` or `filename` are not provided.
            RuntimeError: If the download fails for any reason.
        """
        if not repo_id or not filename:
            raise ValueError("'repo_id' and 'filename' must be provided.")

        local_path = (local_dir / filename).resolve()
        cache_dir = self.settings.cache_dir / f"{repo_id.replace('/', '_')}_{filename}"

        if local_path.exists():
            print(f"Weights file found at: {local_path}")
            return local_path

        print(f"Model weights '{filename}' not found locally. Attempting to download from {repo_id}...")

        try:
            local_dir.mkdir(parents=True, exist_ok=True)

            downloaded_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir),
                local_dir=str(local_dir),
                # local_dir_use_symlinks=False,  # Use direct file placement
            )

            downloaded_path = Path(downloaded_path_str)

            # hf_hub_download might place it in a subfolder structure, so we ensure it's moved to the final destination
            if downloaded_path.resolve() != local_path.resolve():
                shutil.move(str(downloaded_path), str(local_path))
                print(f"Moved weights to final destination: {local_path}")
            else:
                print(f"Downloaded weights directly to: {local_path}")

            # Clean up the cache directory if it was used
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)

            return local_path

        except Exception as e:
            # Clean up any partial download
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(
                f"Failed to download weights file '{filename}' from repo '{repo_id}'. Error: {e}",
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

    def load_dataset(self, dataset_path: str | Path, **kwargs: Any) -> Any:
        """Loads a dataset from disk.

        This method attempts to load a dataset from the specified path. If a `split`
        name is provided and a corresponding subdirectory exists, it will load
        the split from that subdirectory. Otherwise, it loads the entire dataset
        from the base path.

        Args:
            dataset_path (str | Path): The local path to the dataset,
                typically returned by `download_dataset`.
            **kwargs: Additional keyword arguments to pass to the
                `datasets.load_from_disk` function.

        Returns:
            Any: The loaded dataset, typically a `datasets.Dataset` or
                `datasets.DatasetDict` object.
        """
        return load_from_disk(str(dataset_path), **kwargs)
