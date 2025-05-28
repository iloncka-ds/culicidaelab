"""
Module for managing datasets configuration and loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from omegaconf import OmegaConf

from ..core.config_manager import ConfigurableComponent, ConfigManager

from ..core.loader_protocol import DatasetLoader


class DatasetsManager(ConfigurableComponent):
    """Manages datasets configuration and loading."""

    def __init__(
        self,
        config_manager: ConfigManager,
        dataset_loader: DatasetLoader,
        datasets_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ):
        """Initialize with explicit dependencies."""
        super().__init__(config_manager)
        self.dataset_loader = dataset_loader
        self.datasets_dir = Path(datasets_dir) if datasets_dir else None
        self.config_path = Path(config_path) if config_path else None
        self.datasets_config: dict[str, Any] = {}
        self.loaded_datasets: dict[str, Any] = {}

        if self.config_path:
            self.load_config()

    def load_config(self) -> dict[str, Any]:
        """Load datasets configuration."""
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        config = OmegaConf.load(self.config_path)
        self.datasets_config = config.get("datasets", {})
        return self.datasets_config

    def save_config(self, config_path: Path | None = None) -> None:
        """Save datasets configuration."""
        save_path = config_path or self.config_path
        if not save_path:
            raise ValueError("No configuration path specified")

        config = {"datasets": self.datasets_config}
        OmegaConf.save(config=OmegaConf.create(config), f=save_path)

    def get_dataset_info(self, dataset_name: str) -> dict[str, Any]:
        """Get information about a specific dataset."""
        if dataset_name not in self.datasets_config:
            raise KeyError(f"Dataset {dataset_name} not found in configuration")
        return self.datasets_config[dataset_name]

    def list_datasets(self) -> list[str]:
        """List all available datasets."""
        return list(self.datasets_config.keys())

    def add_dataset(self, name: str, info: dict[str, Any]) -> None:
        """Add a new dataset to the configuration."""
        if name in self.datasets_config:
            raise ValueError(f"Dataset {name} already exists")
        self.datasets_config[name] = info

    def remove_dataset(self, name: str) -> None:
        """Remove a dataset from the configuration."""
        if name not in self.datasets_config:
            raise KeyError(f"Dataset {name} not found")
        del self.datasets_config[name]

    def update_dataset(self, name: str, info: dict[str, Any]) -> None:
        """Update dataset information."""
        if name not in self.datasets_config:
            raise KeyError(f"Dataset {name} not found")
        self.datasets_config[name].update(info)

    def load_dataset(self, dataset_name: str, split: str | None = None, **kwargs) -> Any:
        """Load dataset with improved error handling and separation of concerns."""
        config = self.get_dataset_info(dataset_name)
        dataset_path = config.get("path")

        if not dataset_path:
            raise ValueError(f"Dataset path not specified for {dataset_name}")

        dataset = self.dataset_loader.load_dataset(dataset_path, split=split, **kwargs)
        self.loaded_datasets[dataset_name] = dataset
        return dataset

    def get_loaded_dataset(self, dataset_name: str) -> Any:
        """Get previously loaded dataset with improved error handling."""
        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_name} has not been loaded")
        return self.loaded_datasets[dataset_name]
