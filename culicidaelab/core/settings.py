"""Facade for managing configurations and resources in CulicidaeLab.

This module provides the `Settings` class and `get_settings` function, which
serve as the main entry point for accessing application settings, resource
paths, and other global parameters.

The Settings class implements the Singleton pattern to ensure consistent
configuration state across the application. It provides methods for:
- Accessing and modifying configuration values
- Managing model weights and datasets
- Handling API keys for external services
- Managing resource directories and workspace paths

Example:
    >>> from culicidaelab.core.settings import get_settings
    >>> settings = get_settings()
    >>> model_path = settings.get_model_weights_path('classifier')
    >>> dataset_path = settings.get_dataset_path('classification')
"""

import os
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager
import threading
import hashlib
import json

from culicidaelab.core.config_manager import ConfigManager
from culicidaelab.core.resource_manager import ResourceManager
from culicidaelab.core.species_config import SpeciesConfig
from culicidaelab.core.config_models import CulicidaeLabConfig
from culicidaelab.core.utils import create_safe_path


class Settings:
    """
    User-friendly facade for CulicidaeLab configuration management.

    This class provides a simple, stable interface to access configuration values,
    resource directories, and application settings. All actual operations
    are delegated to a validated configuration object managed by ConfigManager
    and a ResourceManager.

    The Settings class is implemented as a singleton to ensure consistent
    configuration state across the application. It manages:
    - Configuration values through get_config() and set_config()
    - Resource directories for models, datasets, and cache
    - Dataset paths and splits
    - Model weights paths and types
    - API keys for external services
    - Temporary workspaces for processing

    Attributes:
        config (CulicidaeLabConfig): The current configuration object
        model_dir (Path): Directory for model weights
        dataset_dir (Path): Directory for datasets
        cache_dir (Path): Directory for cached data
        config_dir (Path): Active user configuration directory
        species_config (SpeciesConfig): Configuration for species detection
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
    def get_cache_key_for_split(self, split: str | list[str] | None) -> str:
        """
        Generates a unique, deterministic hash for any valid split configuration.
        This hash is used to create unique directory names for dataset splits.

        Args:
            split (str | list[str] | None): The split configuration to hash.
                Can be a single split name (e.g., 'train'), a list of splits
                (e.g., ['train', 'val']), or None.

        Returns:
            str: A 16-character hexadecimal hash that uniquely identifies the
                split configuration. This hash is deterministic for the same
                input.

        Example:
            >>> settings.get_cache_key_for_split('train')
            'a1b2c3d4e5f6g7h8'
            >>> settings.get_cache_key_for_split(['train', 'val'])
            'h8g7f6e5d4c3b2a1'
        """
        if isinstance(split, list):
            split.sort()

        # json.dumps correctly handles None, converting it to the string "null"
        split_str = json.dumps(split, sort_keys=True)

        hasher = hashlib.sha256(split_str.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def construct_split_path(
        self,
        dataset_base_path: Path,
        split: str | list[str] | None = None,
    ) -> Path:
        """
        Gets the standardized, absolute path for a dataset's directory.

        This is the single source of truth for dataset path construction.

        Args:
            name (str): The name of the dataset (e.g., 'classification').
            split (str | list[str] | None, optional): If provided, returns the specific
                cache path for this split configuration. Otherwise, returns the base
                directory for the dataset.
            ensure_exists (bool): If True, ensures the directory is created on disk.

        Returns:
            Path: The absolute path to the dataset directory.
        """
        # Determine the final path (either base or split-specific)
        final_path = dataset_base_path
        if split is not None:
            split_key = self.get_cache_key_for_split(split)
            final_path = dataset_base_path / split_key

        return final_path

    def get_dataset_path(self, dataset_type: str, split: str | list[str] | None = None) -> Path:
        """Gets the standardized path for a specific dataset directory.

        Args:
            dataset_type: The name of the dataset type (e.g., 'classification').

        Returns:
            An absolute path to the dataset directory.
        """
        if dataset_type not in self.config.datasets:
            raise ValueError(f"Dataset type '{dataset_type}' not configured.")

        dataset_path_str = self.config.datasets[dataset_type].path
        filal_path = Path(dataset_path_str)
        if not filal_path.is_absolute():
            filal_path = self.dataset_dir / filal_path

        filal_path.mkdir(parents=True, exist_ok=True)

        if split is not None:
            filal_path = self.construct_split_path(
                dataset_base_path=filal_path,
                split=split,
            )
        return filal_path

    def list_datasets(self) -> list[str]:
        """Get list of configured dataset types in the application.

        Returns:
            list[str]: A list of dataset type identifiers that are configured
                in the application settings. These correspond to the different
                dataset categories available for training and inference.

        Example:
            >>> settings.list_datasets()
            ['classification', 'detection', 'segmentation']
        """
        return list(self.config.datasets.keys())

    # Model Management
    def construct_weights_path(
        self,
        predictor_type: str,
        backend: str | None = None,
    ) -> Path:
        """
        A pure, static function to construct a fully qualified model weights path.

        This is the single source of truth for model path construction, creating a
        structured path like: .../models/<predictor_type>/<backend>/<filename>.

        Args:
            model_dir (Path): The base directory for all models (e.g., '.../culicidaelab/models').
            predictor_type (str): The type of the predictor (e.g., 'classifier'). Used as a subdirectory.
            predictor_config (PredictorConfig): The Pydantic model for the predictor's configuration.
            backend (str | None, optional): The target backend (e.g., 'torch', 'onnx').
                                            If None, uses the default from the config.

        Returns:
            Path: The absolute, structured path to the model weights file.

        Raises:
            ValueError: If a valid backend or weights filename cannot be determined.
        """
        predictor_config = self.get_config(f"predictors.{predictor_type}")
        final_backend = backend if backend is not None else predictor_config.backend
        if not final_backend:
            raise ValueError(f"No backend specified for model '{predictor_type}'.")

        if not predictor_config.weights or final_backend not in predictor_config.weights:
            raise ValueError(f"Backend '{final_backend}' not defined in weights config for '{predictor_type}'.")

        filename = predictor_config.weights[final_backend].filename
        if not filename:
            raise ValueError(f"Filename for backend '{final_backend}' is missing in config for '{predictor_type}'.")

        # Sanitize the components that will become directories
        predictor_dir = create_safe_path(predictor_type)
        backend_dir = create_safe_path(final_backend)

        # Assemble the final, structured path
        return self.model_dir / predictor_dir / backend_dir / filename

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

        local_path = self.construct_weights_path(
            predictor_type=model_type,
            backend=backend,
        )
        return local_path

    def list_model_types(self) -> list[str]:
        """Get list of available model types configured in the application.

        Returns:
            list[str]: A list of model type identifiers (e.g., ['classifier',
                'detector', 'segmenter']) that are configured in the application.
                These types correspond to the different predictors available
                in the CulicidaeLab system.

        Example:
            >>> settings.list_model_types()
            ['classifier', 'detector', 'segmenter']
        """
        return list(self.config.predictors.keys())

    # API Key Management
    def get_api_key(self, provider: str) -> str | None:
        """Get API key for external provider from environment variables.

        The method looks for environment variables in the following format:
        - KAGGLE_API_KEY for 'kaggle' provider
        - HUGGINGFACE_API_KEY for 'huggingface' provider
        - ROBOFLOW_API_KEY for 'roboflow' provider

        Args:
            provider (str): The name of the provider. Must be one of:
                'kaggle', 'huggingface', or 'roboflow'.

        Returns:
            str | None: The API key if found in environment variables,
                None if the provider is not supported or the key is not set.

        Example:
            >>> api_key = settings.get_api_key('huggingface')
            >>> if api_key:
            ...     # Use the API key
            ... else:
            ...     # Handle missing key
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
        """Creates a temporary workspace directory that is automatically cleaned up.

        This context manager creates a temporary directory for processing operations
        and automatically cleans it up when the context is exited.

        Args:
            prefix (str, optional): Prefix for the temporary directory name.
                Defaults to "workspace".

        Yields:
            Path: Path to the temporary workspace directory.

        Example:
            >>> with settings.temp_workspace(prefix='processing') as workspace:
            ...     # Do some work in the temporary directory
            ...     (workspace / 'output.txt').write_text('results')
            # Directory is automatically cleaned up after the with block
        """
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
