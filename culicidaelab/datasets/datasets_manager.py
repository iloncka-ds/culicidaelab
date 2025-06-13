from __future__ import annotations

from typing import Any, Dict

from ..core.settings import Settings
from ..core.config_models import DatasetConfig
from ..core.loader_protocol import DatasetLoader


class DatasetsManager:
    """
    Manages access, loading, and caching of datasets defined in the configuration.

    This manager acts as a high-level interface that uses the global Settings
    for configuration and a dedicated loader for the actual data loading,
    decoupling the logic of what datasets are available from how they are loaded.
    """

    def __init__(self, settings: Settings, dataset_loader: DatasetLoader):
        """
        Initialize the DatasetsManager with its dependencies.

        Args:
            settings: The main Settings object for the library.
            dataset_loader: An object that conforms to the DatasetLoader protocol.
        """
        self.settings = settings
        self.dataset_loader = dataset_loader
        self.loaded_datasets: Dict[str, Any] = {}

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
        # Delegate directly to the Settings object's helper method
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
        # First, check if the dataset is already loaded and cached
        if dataset_name in self.loaded_datasets:
            # Note: This simple cache doesn't handle different splits/kwargs.
            # A more complex key would be needed for that, e.g., f"{dataset_name}_{split}"
            return self.get_loaded_dataset(dataset_name)

        # Get the absolute path from the settings object, which handles resolution.
        dataset_path = self.settings.get_dataset_path(dataset_name)

        print(f"Loading dataset '{dataset_name}' from: {dataset_path}...")
        dataset = self.dataset_loader.load_dataset(str(dataset_path), split=split, **kwargs)

        # Cache the loaded dataset
        self.loaded_datasets[dataset_name] = dataset
        print(f"Dataset '{dataset_name}' loaded and cached.")
        return dataset

    def get_loaded_dataset(self, dataset_name: str) -> Any:
        """
        Get a dataset from the cache if it has been previously loaded.

        Args:
            dataset_name: The name of the loaded dataset.

        Returns:
            The cached dataset object.

        Raises:
            KeyError: If the dataset has not been loaded yet.
        """
        if dataset_name not in self.loaded_datasets:
            raise KeyError(
                f"Dataset '{dataset_name}' has not been loaded. " f"Call `load_dataset('{dataset_name}')` first."
            )
        return self.loaded_datasets[dataset_name]
