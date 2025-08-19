"""Manages the loading and caching of datasets.

This module provides the DatasetsManager class, which acts as a centralized
system for handling datasets defined in the configuration files. It interacts
with the settings and provider services to download, cache, and load data
for use in the application.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from culicidaelab.core.config_models import DatasetConfig
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.settings import Settings


class DatasetsManager:
    """Manages access, loading, and caching of configured datasets.

    This manager provides a high-level interface that uses the global settings
    for configuration and a dedicated provider service for the actual data
    loading. This decouples the logic of what datasets are available from how
    they are loaded and sourced.

    Attributes:
        settings: The main settings object for the library.
        provider_service: The service for resolving and using data providers.
        loaded_datasets: A cache for storing the paths of downloaded datasets.
    """

    def __init__(self, settings: Settings):
        """Initializes the DatasetsManager with its dependencies.

        Args:
            settings (Settings): The main Settings object for the library.
        """
        self.settings = settings
        self.provider_service = ProviderService(settings)
        self.loaded_datasets: dict[str, str | Path] = {}

    def get_dataset_info(self, dataset_name: str) -> DatasetConfig:
        """Retrieves the configuration for a specific dataset.

        Args:
            dataset_name (str): The name of the dataset (e.g., 'classification').

        Returns:
            DatasetConfig: A Pydantic model instance containing the dataset's
                validated configuration.

        Raises:
            KeyError: If the specified dataset is not found in the configuration.

        Example:
            >>> manager = DatasetsManager(settings)
            >>> try:
            ...     info = manager.get_dataset_info('classification')
            ...     print(info.provider_name)
            ... except KeyError as e:
            ...     print(e)
        """
        dataset_config = self.settings.get_config(f"datasets.{dataset_name}")
        if not dataset_config:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration.")
        return dataset_config

    def list_datasets(self) -> list[str]:
        """Lists all available dataset names from the configuration.

        Returns:
            list[str]: A list of configured dataset names.

        Example:
            >>> manager = DatasetsManager(settings)
            >>> available_datasets = manager.list_datasets()
            >>> print(available_datasets)
        """
        return self.settings.list_datasets()

    def list_loaded_datasets(self) -> list[str]:
        """Lists all datasets that have been loaded during the session.

        Returns:
            list[str]: A list of names for datasets that are currently cached.

        Example:
            >>> manager = DatasetsManager(settings)
            >>> _ = manager.load_dataset('classification', split='train')
            >>> loaded = manager.list_loaded_datasets()
            >>> print(loaded)
            ['classification']
        """
        return list(self.loaded_datasets.keys())

    def load_dataset(
        self,
        name: str,
        split: str | list[str] | None = None,
        config_name: str | None = "default",
    ) -> Any:
        """
        Loads a dataset, handling complex splits and caching automatically.

        Args:
            name (str): The name of the dataset to load.
            split (str | list[str] | None, optional): The split(s) to load.
                - str: A single split name (e.g., "train", "test").
                - None: Loads ALL available splits into a `DatasetDict`.
                - Advanced: Can be a slice ("train[:100]") or a list for
                  cross-validation.

        Returns:
            The loaded dataset object.
        """
        # 1. Get config and provider
        config = self.get_dataset_info(name)
        provider = self.provider_service.get_provider(config.provider_name)

        # If we've already loaded this dataset in this session, prefer the cached path
        if name in self.loaded_datasets:
            cached = self.loaded_datasets[name]
            # Ensure a Path object when calling the provider
            cached_path = Path(cached) if not isinstance(cached, Path) else cached
            return provider.load_dataset(cached_path)

        # 2. Determine paths using the hashed cache key (Manager's responsibility)
        dataset_base_path = self.settings.dataset_dir / config.path
        split_key = self._get_cache_key_for_split(split)
        split_path = dataset_base_path / split_key

        # 3. Check cache, otherwise download
        downloaded_path = None
        if not split_path.exists():
            # Instruct the provider to download and save to the precise cache path
            # Some providers may return the actual saved path; prefer that when present.
            downloaded_path = provider.download_dataset(
                dataset_name=name,
                config_name=config_name,
                save_dir=split_path,
                split=split,
            )
        else:
            print(f"Cache hit for split config: {split} (key: {split_key})")

        # 4. Instruct the provider to load from the appropriate path
        load_from = None
        if downloaded_path:
            load_from = Path(downloaded_path) if not isinstance(downloaded_path, Path) else downloaded_path
        else:
            load_from = split_path

        dataset = provider.load_dataset(load_from)

        # 5. Update the session cache
        self.loaded_datasets[name] = load_from

        return dataset

    @staticmethod
    def _get_cache_key_for_split(split: str | list[str] | None) -> str:
        """
        Generates a unique, deterministic hash for any valid split configuration.
        Handles None, strings, and lists of strings.
        """
        if isinstance(split, list):
            split.sort()

        # json.dumps correctly handles None, converting it to the string "null"
        split_str = json.dumps(split, sort_keys=True)

        hasher = hashlib.sha256(split_str.encode("utf-8"))
        return hasher.hexdigest()[:16]
