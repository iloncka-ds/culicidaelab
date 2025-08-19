"""Base provider class that all data providers inherit from.

This module defines the `BaseProvider` abstract base class, which establishes an
interface for classes responsible for downloading datasets and model weights
from various sources (e.g., Hugging Face).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseProvider(ABC):
    """Abstract base class for all data and model providers."""

    @abstractmethod
    def download_dataset(
        self,
        dataset_name: str,
        save_dir: Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """Downloads a dataset from a source.

        Args:
            dataset_name (str): The name of the dataset to download.
            save_dir (Path | None, optional): The directory to save the dataset.
                Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments to pass to the download method.

        Returns:
            Path: The path to the downloaded dataset.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def download_model_weights(
        self,
        model_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """Downloads model weights and returns the path to them.

        Args:
            model_type (str): The type of model (e.g., 'detection', 'classification').
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path to the model weights file.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_provider_name(self) -> str:
        """Gets the unique name of the provider.

        Returns:
            str: A string representing the provider's name (e.g., 'huggingface').
        """
        pass

    @abstractmethod
    def load_dataset(
        self,
        dataset_path: str | Path,
        **kwargs: Any,
    ) -> Any:
        """Loads a dataset from a local path.

        Args:
            dataset_path (str | Path): The local path to the dataset, typically
                returned by `download_dataset`.
            **kwargs: Additional keyword arguments for loading.

        Returns:
            Any: The loaded dataset object (e.g., a Hugging Face Dataset, a
            PyTorch Dataset, or a Pandas DataFrame).
        """
        raise NotImplementedError("Subclasses must implement this method")
