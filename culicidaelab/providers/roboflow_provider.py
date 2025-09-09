"""Roboflow Provider for managing datasets.

This module provides the `RoboflowProvider` class, which is a concrete
implementation of `BaseProvider`. It handles downloading datasets from
Roboflow.
"""

from __future__ import annotations

# Standard library
from pathlib import Path
from typing import Any

# Internal imports
from culicidaelab.core.base_provider import BaseProvider
from culicidaelab.core.settings import Settings

try:
    import roboflow
except ImportError:
    raise ImportError(
        "Roboflow is not installed. Please install it with 'pip install roboflow'",
    )


class RoboflowProvider(BaseProvider):
    """Provider for downloading and managing Roboflow datasets.

    This class interfaces with Roboflow to fetch datasets. It uses
    the core settings object for path resolution and API key access.

    Attributes:
        provider_name (str): The name of the provider, "roboflow".
        settings (Settings): The main Settings object for the library.
        api_key (str | None): The Roboflow API key, if provided.
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initializes the Roboflow provider.

        Args:
            settings (Settings): The main Settings object for the library.
            **kwargs (Any): Catches other config parameters from the provider's config.
        """
        super().__init__()
        self.provider_name = "roboflow"
        self.settings = settings.get_config(f"providers.{self.provider_name}")
        self.api_key: str | None = kwargs.get("api_key") or self.settings.get_api_key(
            self.provider_name,
        )
        self.workspace = kwargs.get("rf_workspace") or self.settings.get_config("rf_workspace")
        self.project = kwargs.get("rf_dataset") or self.settings.get_config("rf_dataset")
        self.version = kwargs.get("project_version") or self.settings.get_config("project_version")
        self.model_format = kwargs.get("data_fornat") or self.settings.get_config(
            "data_fornat",
            "yolov11",
        )

    def download_dataset(
        self,
        dataset_name: str,
        save_dir: Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """Downloads a dataset from Roboflow.

        Args:
            dataset_name (str): Name of the dataset to download.
            save_dir (Path | None, optional): Directory to save the dataset.
                Defaults to None, using the path from settings.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Path: The path to the downloaded dataset.

        Raises:
            ValueError: If the configuration is missing required fields.
            RuntimeError: If the download fails.
        """
        if not self.api_key:
            raise ValueError("Roboflow API key is required to download datasets.")

        save_path = self.settings.get_dataset_path(dataset_name)
        if save_dir:
            save_path = save_dir

        if not all([self.workspace, self.project, self.version]):
            raise ValueError(
                f"Configuration for provider '{self.provider_name}' is missing:",
                "Please provide values for the following fields:",
                f" 'rf_workspace' : {self.workspace}",
                f" 'rf_dataset': {self.project}",
                f" or 'project_version': {self.version}.",
            )

        try:
            rf = roboflow.Roboflow(api_key=self.api_key)
            ws = rf.workspace(self.workspace)
            proj = ws.project(self.project)
            ver = proj.version(self.version)

            ver.download(model_format=self.model_format, location=str(save_path))

            return save_path

        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {dataset_name} from Roboflow: {e}") from e

    def download_model_weights(self, model_type: str, *args: Any, **kwargs: Any) -> Path:
        """Not implemented for Roboflow provider."""
        raise NotImplementedError("Roboflow provider does not support downloading model weights separately.")

    def get_dataset_metadata(self, dataset_name: str) -> dict[str, Any]:
        """Not implemented for Roboflow provider."""
        raise NotImplementedError("Roboflow provider does not support fetching dataset metadata.")

    def get_provider_name(self) -> str:
        """Returns the provider's name."""
        return self.provider_name

    def load_dataset(self, dataset_path: str | Path, **kwargs: Any) -> Any:
        """Not implemented for Roboflow provider."""
        raise NotImplementedError("Loading datasets from disk is not yet implemented for Roboflow provider.")
