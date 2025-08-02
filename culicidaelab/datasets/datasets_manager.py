"""Manages the loading and caching of datasets.

This module provides the DatasetsManager class, which acts as a centralized
system for handling datasets defined in the configuration files. It interacts
with the settings and provider services to download, cache, and load data
for use in the application.
"""

from __future__ import annotations

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

    def load_dataset(self, dataset_name: str, split: str | None = None, **kwargs: Any) -> Any:
        """Loads a specific dataset, downloading it if not already cached.

        This method first checks a local cache for the dataset path. If the
        dataset is not cached, it resolves the path using the settings,
        instructs the appropriate provider to download it, and caches the path.
        Finally, it uses the provider to load the dataset into memory.

        Args:
            dataset_name (str): The name of the dataset to load.
            split (str, optional): The specific dataset split to load (e.g.,
                'train', 'test'). Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the provider's
                dataset loading function.

        Returns:
            Any: The loaded dataset object, with its type depending on the provider.

        Raises:
            KeyError: If the dataset configuration does not exist.
        """
        dataset_config = self.get_dataset_info(dataset_name)
        provider = self.provider_service.get_provider(dataset_config.provider_name)

        if dataset_name in self.loaded_datasets:
            dataset_path = self.loaded_datasets[dataset_name]
        else:
            print(f"Dataset '{dataset_name}' not in cache. Downloading...")
            dataset_path = provider.download_dataset(dataset_name, split=split, **kwargs)
            self.loaded_datasets[dataset_name] = dataset_path
            print(f"Dataset '{dataset_name}' downloaded and path cached.")

        print(f"Loading '{dataset_name}' from path: {dataset_path}")
        dataset = provider.load_dataset(dataset_path, split=split, **kwargs)
        print(f"Dataset '{dataset_name}' loaded successfully.")

        return dataset
