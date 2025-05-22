"""
Configuration module for CulicidaeLab.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf
from dotenv import load_dotenv

from .config_manager import ConfigManager, ConfigurableComponent

# Load environment variables from .env file
load_dotenv()


class Settings(ConfigurableComponent):
    _instance = None

    def __new__(cls, *args, **kwargs):
        # If no existing instance, create a new one
        if not cls._instance:
            cls._instance = super().__new__(cls)
            return cls._instance

        # If config_dir is provided, check if it's different from the existing instance
        if args or kwargs:
            config_dir = args[0] if args else kwargs.get("config_dir")
            if config_dir is not None:
                # Normalize paths
                new_config_dir = Path(config_dir).resolve()
                existing_config_dir = Path(cls._instance.config_manager.config_path).resolve()

                # If config directories are different, create a new instance
                if new_config_dir != existing_config_dir:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, config_dir: str | Path | None = None):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.root_dir = Path(__file__).parent
        self.environment = os.getenv("APP_ENV", "development")

        # Initialize configuration
        config_manager = ConfigManager(config_dir or "../conf")
        super().__init__(config_manager)

        # Load configuration with environment override
        self.config_manager.load_config(app_settings=self.environment)
        self._config = self.config_manager.get_config()  # Initialize config as a regular attribute
        self.load_config()

        # Initialize components
        self._initialize_components()

        self.initialized = True

    def _initialize_components(self) -> None:
        """Initialize individual components from configuration."""
        # Set up directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Set up required directories using ConfigManager's resource directories."""
        # Get resource directories from config manager
        resource_dirs = self.config_manager.get_resource_dirs()

        # Create directories if they don't exist
        for dir_name, dir_path in resource_dirs.items():
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

        # Optional: Map config-specific directories to resource directories
        resource_mapping = self._config.get("resource_mapping", {})
        for key, relative_path in resource_mapping.items():
            # Convert relative path to absolute
            abs_path = Path(relative_path).resolve()

            # Ensure the mapped directory exists
            if not abs_path.exists():
                abs_path.mkdir(parents=True, exist_ok=True)

            # Optionally set as an attribute if needed
            setattr(self, f"_{key}", abs_path)

    def get_resource_dir(self, resource_type: str) -> Path:
        """
        Get a specific resource directory.

        Args:
            resource_type (str): Type of resource directory
            (e.g., 'user_data_dir', 'cache_dir', 'model_dir', etc.)

        Returns:
            Path: Path to the requested resource directory
        """
        resource_dirs = self.config_manager.get_resource_dirs()

        if resource_type not in resource_dirs:
            raise ValueError(f"Unknown resource type: {resource_type}")

        return resource_dirs[resource_type]

    def get_dataset_path(self, dataset_type: str) -> Path:
        """
        Get dataset path, using resource directory management.

        Args:
            dataset_type (str): Type of dataset
            ('detection', 'segmentation', or 'classification')

        Returns:
            Path: Path to the dataset directory
        """
        # Get dataset directory from resource directories
        dataset_dir = self.get_resource_dir("dataset_dir")

        # Check if dataset type exists in config
        if dataset_type not in self._config.datasets.paths:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Combine dataset directory with specific dataset path
        dataset_path = dataset_dir / self._config.datasets.paths[dataset_type]

        # Create directory if it doesn't exist
        dataset_path.mkdir(parents=True, exist_ok=True)

        return dataset_path

    def get_processing_params(self) -> dict[str, Any]:
        """Get processing parameters for model inference."""
        return OmegaConf.to_container(self._config.processing, resolve=True)

    def _setup_config_dir(self, config_dir: str | Path | None) -> str:
        """
        Set up configuration directory, handling both external and default configs.

        Args:
            config_dir: External config directory path or None

        Returns:
            str: Path to the active config directory
        """
        if config_dir is None:
            return self.default_config_dir

        # Convert to absolute path if relative
        if not os.path.isabs(str(config_dir)):
            config_dir = os.path.join(str(self.root_dir), str(config_dir))

        # Check if external config directory exists and has required files
        config_dir = str(config_dir)
        if not os.path.exists(config_dir):
            print(f"Warning: Config directory {config_dir} not found. Using default configs.")
            return self.default_config_dir

        required_files = [
            f"app_settings_{self.environment}.yaml",
            "models_config.yaml",
            "species_classes.yaml",
            "species_metadata.yaml",
            "datasets_config.yaml",
        ]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(config_dir, f))]

        if missing_files:
            print(f"Warning: Missing required config files in {config_dir}: {missing_files}")
            print("Copying default configs to external directory...")

            # Create config directory if it doesn't exist
            os.makedirs(config_dir, exist_ok=True)

            # Copy missing default configs
            for file in missing_files:
                src = os.path.join(self.default_config_dir, file)
                dst = os.path.join(config_dir, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"Copied {file} to {config_dir}")
                else:
                    print(f"Warning: Default config {file} not found")

        return config_dir

    @property
    def weights_dir(self) -> Path:
        """Get the weights directory path."""
        return self.get_resource_dir("weights_dir")

    @property
    def datasets_dir(self) -> Path:
        """Get the datasets directory path."""
        return self.get_resource_dir("datasets_dir")

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self.get_resource_dir("cache_dir")

    @property
    def config_dir(self) -> Path:
        """Get the configuration directory path."""
        return Path(self._config.paths.config_dir).resolve()


# Module-level settings instance
_settings_instance = None


def get_settings(config_dir: str | Path | None = None) -> Settings:
    """
    Get settings instance with optional config directory.
    This ensures settings are properly initialized with the desired configuration.
    If no config_dir is provided and an instance already exists, returns the existing instance.

    Args:
        config_dir: Optional path to external config directory

    Returns:
        Settings instance
    """
    global _settings_instance

    # Create or get the settings instance
    settings = Settings(config_dir)

    # If no config_dir is provided, set as the module-level instance
    if config_dir is None and _settings_instance is None:
        _settings_instance = settings

    return settings
