"""Base provider class that all data providers inherit from.

This module defines the `BaseProvider` abstract base class, which establishes an
interface for classes responsible for downloading datasets and model weights
from various sources (e.g., Hugging Face).

Example:
    >>> from culicidaelab.providers.huggingface_provider import HuggingFaceProvider
    >>> provider = HuggingFaceProvider()
    >>> dataset_path = provider.download_dataset("user/my-dataset")
    >>> print(dataset_path)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseProvider(ABC):
    """Abstract base class for all data and model providers.

    This class defines the contract for providers that fetch resources like
    datasets and model weights.
    """

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
            dataset_name (str): The name or identifier of the dataset to download.
            save_dir (Path | None, optional): The directory to save the dataset.
                If None, a default directory may be used. Defaults to None.
            *args: Additional positional arguments for the provider's implementation.
            **kwargs: Additional keyword arguments for the provider's implementation.

        Returns:
            Path: The path to the downloaded dataset directory or file.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def download_model_weights(
        self,
        repo_id: str,
        filename: str,
        local_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """Downloads model weights and returns the path to them.

        Args:
            repo_id (str): The repository ID from which to download the model
                (e.g., 'culicidae/mosquito-detector').
            filename (str): The name of the weights file in the repository.
            local_dir (Path): The local directory to save the weights file.
            *args: Additional positional arguments for the provider's implementation.
            **kwargs: Additional keyword arguments for the provider's implementation.

        Returns:
            Path: The local path to the downloaded model weights file.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
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

        This method is responsible for loading a dataset that has already been
        downloaded to the local filesystem.

        Args:
            dataset_path (str | Path): The local path to the dataset, typically
                a path returned by `download_dataset`.
            **kwargs: Additional keyword arguments for loading the dataset, which
                may vary by provider and dataset format.

        Returns:
            Any: The loaded dataset object, which could be a Hugging Face Dataset,
            a PyTorch Dataset, a Pandas DataFrame, or another format.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
