# src/culicidaelab/core/base_provider.py
from abc import ABC, abstractmethod
from pathlib import Path


class BaseProvider(ABC):
    @abstractmethod
    def download_dataset(
        self,
        dataset_name: str,
        save_dir: str | None = None,
        *args,
        **kwargs,
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
    def download_model_weights(self, model_type: str, *args, **kwargs) -> Path:
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
        """Abstract method for getting provider name"""
        pass
