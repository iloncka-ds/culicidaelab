from __future__ import annotations

from typing import Any

from pathlib import Path

from culicidaelab.core.settings import Settings
from culicidaelab.core.config_models import DatasetConfig
from culicidaelab.core.provider_service import ProviderService


class DatasetsManager:
    """
    Manages access, loading, and caching of datasets defined in the configuration.

    This manager acts as a high-level interface that uses the global Settings
    for configuration and a dedicated loader for the actual data loading,
    decoupling the logic of what datasets are available from how they are loaded.
    """

    def __init__(self, settings: Settings, provider_service: ProviderService):
        """
        Initialize the DatasetsManager with its dependencies.

        Args:
            settings: The main Settings object for the library.
            dataset_loader: An object that conforms to the DatasetLoader protocol.
        """
        self.settings = settings
        self.provider_service = provider_service
        self.loaded_datasets: dict[str, str | Path] = {}

    def get_dataset_info(self, dataset_name: str) -> DatasetConfig:
        """
        Get the validated configuration object for a specific dataset.

        Args:
            dataset_name: The name of the dataset (e.g., 'classification').

        Returns:
            A DatasetConfig Pydantic model instance.

        Raises:
            KeyError: If the dataset is not found in the configuration.
        """
        dataset_config = self.settings.get_config(f"datasets.{dataset_name}")
        if not dataset_config:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration.")
        return dataset_config

    def list_datasets(self) -> list[str]:
        """
        List all available dataset names from the configuration.

        Returns:
            A list of configured dataset names.
        """

        return self.settings.list_datasets()

    def load_dataset(self, dataset_name: str, split: str | None = None, **kwargs) -> Any:
        """
        Load a specific dataset using the injected loader.

        Checks a local cache first. If the dataset is not cached, it resolves
        the dataset's path using the settings and instructs the loader to load it.

        Args:
            dataset_name: The name of the dataset to load.
            split: Optional dataset split to load (e.g., 'train', 'test').
            **kwargs: Additional keyword arguments to pass to the dataset loader.

        Returns:
            The loaded dataset object.

        Raises:
            KeyError: If the dataset configuration doesn't exist.
        """
        dataset_config = self.settings.get_config(f"datasets.{dataset_name}")
        if not dataset_config:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration.")

        provider = self.provider_service.get_provider(dataset_config.provider_name)

        # Determine the path to the dataset first
        if dataset_name in self.loaded_datasets:
            dataset_path = self.loaded_datasets[dataset_name]
        else:
            # Download if not found in our cache
            print(f"Dataset '{dataset_name}' not in cache. Downloading...")
            dataset_path = provider.download_dataset(dataset_name, split=split, **kwargs)
            self.loaded_datasets[dataset_name] = dataset_path
            print(f"Dataset '{dataset_name}' downloaded and path cached.")

        # Load the dataset from the determined path
        print(f"Loading '{dataset_name}' from path: {dataset_path}")
        dataset = provider.load_dataset(dataset_path, split=split, **kwargs)
        print(f"Dataset '{dataset_name}' loaded successfully.")

        return dataset

    def list_loaded_datasets(self) -> list[str]:
        """
        List all loaded datasets.

        Returns:
            List of loaded dataset names.
        """
        return list(self.loaded_datasets.keys())
