# src/culicidaelab/core/base_provider.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseProvider(ABC):
    """Abstract base class for all data and model providers."""

    @abstractmethod
    def download_dataset(
        self,
        dataset_name: str,
        save_dir: str | None = None,
        *args,
        **kwargs: Any,
    ) -> Path:
        """Download a dataset from HuggingFace.

        Args:
            dataset_name (str): Name of the dataset to download
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to None.
            args: Additional arguments
            kwargs: Additional keyword arguments to pass to the download method

        Returns:
            Path: Path to the downloaded dataset
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def load_dataset(
        self,
        dataset_path: str | Path,
        **kwargs: Any,
    ) -> Any:
        """Load a dataset from a local path.

        Args:
            dataset_path (str | Path): The local path to the dataset, typically returned by download_dataset.
            kwargs: Additional keyword arguments for loading.

        Returns:
            Any: The loaded dataset object (e.g., a Hugging Face Dataset, a PyTorch Dataset, a Pandas DataFrame).
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def download_model_weights(
        self,
        model_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """
        Download model and get weights path.

        Args:
            model_type: Type of model ('detection', 'segmentation', or 'classification')
            args: Additional arguments
            kwargs: Additional keyword arguments
        Returns:
            Path: Path to the model weights file
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_provider_name(self) -> str:
        """Gets the unique name of the provider.

        Returns:
            A string representing the provider's name (e.g., 'huggingface').
        """
        pass
