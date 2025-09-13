"""Facade for managing configurations and resources in CulicidaeLab.

This module provides the `Settings` class and `get_settings` function, which
serve as the main entry point for accessing application settings, resource
paths, and other global parameters.
"""

import os
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager
import threading

from culicidaelab.core.config_manager import ConfigManager
from culicidaelab.core.resource_manager import ResourceManager
from culicidaelab.core.species_config import SpeciesConfig
from culicidaelab.core.config_models import CulicidaeLabConfig
from culicidaelab.core.utils import construct_weights_path


class Settings:
    """
    User-friendly facade for CulicidaeLab configuration management.

    This class provides a simple, stable interface to access configuration values,
    resource directories, and application settings. All actual operations
    are delegated to a validated configuration object managed by ConfigManager
    and a ResourceManager.
    """

    _instance: Optional["Settings"] = None
    _lock = threading.Lock()
    _initialized = False

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """Initializes the Settings facade.

        This loads the configuration using a ConfigManager and sets up a
        ResourceManager for file paths.

        Args:
            config_dir: Optional path to a user-provided configuration directory.
        """
        if self._initialized:
            return

        self._config_manager = ConfigManager(user_config_dir=config_dir)
        self.config: CulicidaeLabConfig = self._config_manager.get_config()
        self._resource_manager = ResourceManager()

        # Cache for species config (lazy loaded)
        self._species_config: SpeciesConfig | None = None

        # Store for singleton check
        self._current_config_dir = self._config_manager.user_config_dir

        self._initialized = True

    # Configuration Access
    def get_config(self, path: str | None = None, default: Any = None) -> Any:
        """Gets a configuration value using a dot-separated path.

        Example:
            >>> settings.get_config("predictors.classifier.confidence")

        Args:
            path: A dot-separated string path to the configuration value.
                If None, returns the entire configuration object.
            default: A default value to return if the path is not found.

        Returns:
            The configuration value, or the default value if not found.
        """
        if not path:
            return self.config

        obj = self.config
        try:
            for key in path.split("."):
                if isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    obj = getattr(obj, key)
            return obj if obj is not None else default
        except (AttributeError, KeyError):
            return default

    def set_config(self, path: str, value: Any) -> None:
        """
        Sets a configuration value at a specified dot-separated path.
        This method can traverse both objects (Pydantic models) and dictionaries.

        Note: This modifies the configuration in memory. To make it persistent,
        call `save_config()`.

        Args:
            path: A dot-separated string path to the configuration value.
            value: The new value to set.
        """
        keys = path.split(".")
        obj = self.config

        for key in keys[:-1]:
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                obj = getattr(obj, key)

            if obj is None:
                raise KeyError(f"The path part '{key}' in '{path}' was not found.")

        last_key = keys[-1]
        if isinstance(obj, dict):
            obj[last_key] = value
        else:
            setattr(obj, last_key, value)

    def save_config(self, file_path: str | Path | None = None) -> None:
        """Save current configuration to a user config file.
        Args:
            file_path: Optional path to save the configuration file.
                If None, defaults to "culicidaelab_saved.yaml" in the user config directory.
        """
        if file_path is None:
            if not self._config_manager.user_config_dir:
                raise ValueError("Cannot save config without a specified user config directory.")
            file_path = self._config_manager.user_config_dir / "culicidaelab_saved.yaml"
        self._config_manager.save_config(file_path)

    # Resource Directory Access
    @property
    def model_dir(self) -> Path:
        """Model weights directory."""
        return self._resource_manager.model_dir

    @property
    def weights_dir(self) -> Path:
        """Alias for model_dir."""
        return self.model_dir

    @property
    def dataset_dir(self) -> Path:
        """Datasets directory."""
        return self._resource_manager.dataset_dir

    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        return self._resource_manager.user_cache_dir

    @property
    def config_dir(self) -> Path:
        """The active user configuration directory."""
        return self._config_manager.user_config_dir or self._config_manager.default_config_path

    @property
    def species_config(self) -> SpeciesConfig:
        """Species configuration (lazily loaded)."""
        if self._species_config is None:
            self._species_config = SpeciesConfig(self.config.species)
        return self._species_config

    # Dataset Management
    def get_dataset_path(self, dataset_type: str) -> Path:
        """Gets the standardized path for a specific dataset directory.

        Args:
            dataset_type: The name of the dataset type (e.g., 'classification').

        Returns:
            An absolute path to the dataset directory.
        """
        if dataset_type not in self.config.datasets:
            raise ValueError(f"Dataset type '{dataset_type}' not configured.")

        dataset_path_str = self.config.datasets[dataset_type].path
        path = Path(dataset_path_str)
        if not path.is_absolute():
            path = self.dataset_dir / path

        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_datasets(self) -> list[str]:
        """Get list of configured dataset types."""
        return list(self.config.datasets.keys())

    # Model Management
    def get_model_weights_path(
        self,
        model_type: str,
        backend: str | None = None,
    ) -> Path:
        """Gets the configured path to a model's weights file.

        Args:
            model_type: The name of the model type (e.g., 'classifier').

        Returns:
            The path to the model weights file.
        """
        if model_type not in self.config.predictors:
            raise ValueError(f"Model type '{model_type}' not configured in 'predictors'.")

        return construct_weights_path(
            model_dir=self.model_dir,
            predictor_config=self.get_config(f"predictors.{model_type}"),
            backend=backend,
        )

    def list_model_types(self) -> list[str]:
        """Get list of available model types."""
        return list(self.config.predictors.keys())

    # API Key Management
    def get_api_key(self, provider: str) -> str | None:
        """Get API key for external provider from environment variables.
        Args:
            provider: The name of the provider (e.g., 'kaggle', 'huggingface', 'roboflow').
        """
        api_keys = {
            "kaggle": "KAGGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "roboflow": "ROBOFLOW_API_KEY",
        }
        if provider in api_keys:
            return os.getenv(api_keys[provider])
        return None

    # Utility Methods (delegated to ResourceManager)
    @contextmanager
    def temp_workspace(self, prefix: str = "workspace"):
        with self._resource_manager.temp_workspace(prefix) as workspace:
            yield workspace

    # Instantiation
    def instantiate_from_config(self, config_path: str, **kwargs: Any) -> Any:
        """Instantiates an object from a configuration path.

        This is a convenience method that finds a config object by its path
        and uses the underlying ConfigManager to instantiate it.

        Args:
            config_path: A dot-separated path to the configuration object
                (e.g., "predictors.classifier").
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            The instantiated object.
        """

        config_obj = self.get_config(config_path)
        if not config_obj:
            raise ValueError(f"No configuration object found at path: {config_path}")

        extra_deps = {"settings": self}

        return self._config_manager.instantiate_from_config(
            config_obj,
            extra_params=extra_deps,
            **kwargs,
        )


# Global access function
_SETTINGS_INSTANCE: Settings | None = None
_SETTINGS_LOCK = threading.Lock()


def get_settings(config_dir: str | Path | None = None) -> Settings:
    """
    Get the Settings singleton instance.

    This is the primary way to access Settings throughout the application.
    If a `config_dir` is provided that differs from the existing instance,
    a new instance will be created and returned.

    Args:
        config_dir: Optional path to a user-provided configuration directory.

    Returns:
        The Settings instance.
    """
    global _SETTINGS_INSTANCE
    with _SETTINGS_LOCK:
        resolved_path = Path(config_dir).resolve() if config_dir else None

        # Create a new instance if one doesn't exist, or if the config path has changed.
        if _SETTINGS_INSTANCE is None or _SETTINGS_INSTANCE._current_config_dir != resolved_path:
            _SETTINGS_INSTANCE = Settings(config_dir=config_dir)

        return _SETTINGS_INSTANCE


def list_models() -> list[str]:
    # Returns ["classifier", "detector", "segmenter"]
    return get_settings().list_model_types()


def list_datasets() -> list[str]:
    return get_settings().list_datasets()
