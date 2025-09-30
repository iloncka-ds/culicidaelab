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
            settings: The main Settings object for the library.
        """
        self.settings = settings
        self.provider_service = ProviderService(settings)
        self.loaded_datasets: dict[str, str | Path] = {}

    def get_dataset_info(self, dataset_name: str) -> DatasetConfig:
        """Retrieves the configuration for a specific dataset.

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.datasets import DatasetsManager
            >>> settings = Settings()
            >>> manager = DatasetsManager(settings)
            >>> try:
            ...     info = manager.get_dataset_info('classification')
            ...     print(info.provider_name)
            ... except KeyError as e:
            ...     print(e)

        Args:
            dataset_name: The name of the dataset (e.g., 'classification').

        Returns:
            A Pydantic model instance containing the dataset's
            validated configuration.

        Raises:
            KeyError: If the specified dataset is not found in the configuration.
        """
        dataset_config = self.settings.get_config(f"datasets.{dataset_name}")
        if not dataset_config:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration.")
        return dataset_config

    def list_datasets(self) -> list[str]:
        """Lists all available dataset names from the configuration.

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.datasets import DatasetsManager
            >>> settings = Settings()
            >>> manager = DatasetsManager(settings)
            >>> available_datasets = manager.list_datasets()
            >>> print(available_datasets)

        Returns:
            A list of configured dataset names.
        """
        return self.settings.list_datasets()

    def list_loaded_datasets(self) -> list[str]:
        """Lists all datasets that have been loaded during the session.

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.datasets import DatasetsManager
            >>> settings = Settings()
            >>> manager = DatasetsManager(settings)
            >>> _ = manager.load_dataset('classification', split='train')
            >>> loaded = manager.list_loaded_datasets()
            >>> print(loaded)
            ['classification']

        Returns:
            A list of names for datasets that are currently cached.
        """
        return list(self.loaded_datasets.keys())

    def load_dataset(
        self,
        name: str,
        split: str | list[str] | None = None,
        config_name: str | None = "default",
    ) -> Any:
        """Loads a dataset, handling complex splits and caching automatically.

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.datasets import DatasetsManager
            >>> # This example assumes you have a configured settings object
            >>> settings = Settings()
            >>> manager = DatasetsManager(settings)
            >>> # Load the training split of the classification dataset
            >>> train_dataset = manager.load_dataset('classification', split='train')
            >>> # Load all splits
            >>> all_splits = manager.load_dataset('classification')

        Args:
            name: The name of the dataset to load.
            split: The split(s) to load.
                - str: A single split name (e.g., "train", "test").
                - None: Loads ALL available splits into a `DatasetDict`.
                - Advanced: Can be a slice ("train[:100]") or a list for
                  cross-validation.
            config_name: The name of the dataset configuration to use.
                Defaults to "default".

        Returns:
            The loaded dataset object, which could be a `Dataset` or `DatasetDict`
            depending on the provider and splits requested.
        """
        # 1. Get config and provider
        config = self.get_dataset_info(name)
        provider = self.provider_service.get_provider(config.provider_name)

        split_path = self.settings.get_dataset_path(
            dataset_type=name,
            split=split,
        )

        # Check cache, otherwise download
        downloaded_path = None
        if not split_path.exists():
            # Instruct the provider to download and save to the precise cache path
            downloaded_path = provider.download_dataset(
                dataset_name=name,
                config_name=config_name,
                save_dir=split_path,
                split=split,
            )
        else:
            print(f"Cache hit for split config: {split} {split_path}")

        # Instruct the provider to load from the appropriate path
        load_from = downloaded_path or split_path

        dataset = provider.load_dataset(load_from)

        # Update the session cache
        self.loaded_datasets[name] = load_from

        return dataset
