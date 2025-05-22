"""Roboflow Dataset Provider implementation."""

from typing import Any
import requests
from pathlib import Path
import json

from ..core.base_provider import BaseProvider
from ..core.config_manager import ConfigManager


class RoboflowProvider(BaseProvider):
    """Provider for downloading and managing Roboflow datasets."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize Roboflow provider with configuration.

        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.provider_name = "roboflow"
        self.config_manager = config_manager

        # Get provider configuration with API key
        provider_config = self.config_manager.get_provider_config("roboflow")
        self.api_key = provider_config.get("api_key")
        self.provider_url = provider_config.get("provider_url")
        self.workspace = provider_config.get("rf_workspace")
        self.dataset = provider_config.get("rf_dataset")
        self.version = provider_config.get("project_version")
        self.format = provider_config.get("data_format", "yolov8")

        if not self.api_key:
            raise ValueError("Roboflow API key not found in environment variables")

    def get_metadata(self, dataset_id: str | None = None) -> dict[str, Any]:
        """Get metadata for a specific dataset from Roboflow.

        Args:
            dataset_id (Optional[str]): ID of the dataset. If None, uses configured dataset.

        Returns:
            Dict[str, Any]: Dataset metadata
        """
        if dataset_id is None:
            dataset_id = f"{self.workspace}/{self.dataset}"

        url = f"https://api.roboflow.com/dataset/{dataset_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def download(
        self,
        dataset_id: str | None = None,
        save_dir: str | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> Path:
        """Download a dataset from Roboflow.

        Args:
            dataset_id (Optional[str]): ID of the dataset. If None, uses configured dataset.
            save_dir (Optional[str]): Directory to save the dataset. Defaults to None.
            split (str): Dataset split to download ('train', 'valid', or 'test')
            **kwargs: Additional arguments to pass to the download method

        Returns:
            Path: Path to the downloaded dataset
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "Roboflow package not installed. Install it with: pip install roboflow",
            )

        # Get dataset ID
        if dataset_id is None:
            dataset_id = self.dataset

        # Get save directory
        if save_dir is None:
            save_dir = Path.cwd() / "data" / dataset_id
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Roboflow
        rf = Roboflow(api_key=self.api_key)

        # Get project
        project = rf.workspace(self.workspace).project(dataset_id)
        version = project.version(self.version)

        # Download dataset
        dataset = version.download(
            model_format=self.format,
            location=str(save_dir),
            split=split,
            **kwargs,
        )

        # Save metadata
        metadata = self.get_metadata(f"{self.workspace}/{dataset_id}")
        metadata_file = save_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return save_dir

    def list_versions(self, dataset_id: str | None = None) -> dict[str, Any]:
        """List all versions of a dataset.

        Args:
            dataset_id (Optional[str]): ID of the dataset. If None, uses configured dataset.

        Returns:
            Dict[str, Any]: Dictionary containing version information
        """
        if dataset_id is None:
            dataset_id = f"{self.workspace}/{self.dataset}"

        url = f"https://api.roboflow.com/dataset/{dataset_id}/versions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()
