"""
Centralized configuration management module using OmegaConf.

This module provides a singleton ConfigManager class for handling application
configurations, environment variables, and provider settings without Hydra dependency.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, TypeVar, Generic
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv

from culicidaelab.core.resource_manager import ResourceManager

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management using OmegaConf.

    This class implements the singleton pattern to ensure consistent configuration
    access across the application. It handles loading configurations from YAML files,
    environment variables, and provides utilities for instantiating configured objects.

    Attributes:
        resource_manager (ResourceManager): Manages application resource directories.
    """

    _instance: ConfigManager | None = None
    _env_keys: dict[str, str] = {
        "kaggle": "KAGGLE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "roboflow": "ROBOFLOW_API_KEY",
    }
    _default_config_path: str = "../conf"

    def __new__(
        cls,
        library_config_path: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> ConfigManager:
        """
        Create or return existing ConfigManager instance (Singleton pattern).

        Args:
            library_config_path: Path to library configuration directory.
            config_path: Path to user configuration directory.
            **kwargs: Additional keyword arguments.

        Returns:
            ConfigManager instance.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        library_config_path: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ConfigManager instance.

        Args:
            library_config_path: Path to library configuration directory.
            config_path: Path to user configuration directory.
            **kwargs: Additional keyword arguments.
        """
        if hasattr(self, "initialized") and self.initialized:
            return

        self.config_path = config_path
        self.library_config_path = library_config_path or self._default_config_path
        self._config: DictConfig | None = None
        self.resource_manager = ResourceManager()

        self.initialized = True
        self._load_env_vars()

    def _get_project_root(self) -> Path:
        """
        Get the project root directory by searching for configuration files.

        Returns:
            Path to the project root directory.

        Raises:
            FileNotFoundError: If project root cannot be determined.
        """
        current_file = Path(__file__).resolve()
        project_root = current_file

        while project_root.parent != project_root:
            if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
                return project_root
            project_root = project_root.parent

        raise FileNotFoundError("Could not determine project root directory")

    def _load_env_vars(self) -> None:
        """Load environment variables from .env file if it exists."""
        try:
            env_path = self._get_project_root() / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment variables from {env_path}")
        except Exception as e:
            logger.warning(f"Could not load environment variables: {e}")

    def get_api_key(self, provider: str) -> str | None:
        """
        Get API key for a specific provider from environment variables.

        Args:
            provider: Name of the provider (kaggle, huggingface, roboflow).

        Returns:
            API key if found, None otherwise.

        Raises:
            ValueError: If provider is not recognized.
        """
        if provider not in self._env_keys:
            raise ValueError(
                f"Unknown provider: {provider}. Available providers: {list(self._env_keys.keys())}",
            )

        return os.getenv(self._env_keys[provider])

    def _merge_configs(self, *configs: DictConfig) -> DictConfig:
        """
        Merge multiple configurations with the latter taking precedence.

        Args:
            *configs: Variable number of DictConfig objects to merge.

        Returns:
            Merged configuration.
        """
        if not configs:
            return OmegaConf.create({})

        merged = OmegaConf.create({})
        for config in configs:
            if config is not None:
                merged = OmegaConf.merge(merged, config)

        return merged

    def _load_config_file(
        self,
        config_path: Path,
        config_name: str,
    ) -> DictConfig | None:
        """
        Load a configuration file from the specified path.

        Args:
            config_path: Directory containing the configuration file.
            config_name: Name of the configuration file (without extension).

        Returns:
            Loaded configuration or None if file doesn't exist.
        """
        config_file = config_path / f"{config_name}.yaml"
        if not config_file.exists():
            config_file = config_path / f"{config_name}.yml"

        if config_file.exists():
            try:
                return OmegaConf.load(config_file)
            except Exception as e:
                logger.error(f"Error loading config file {config_file}: {e}")
                return None

        return None

    def load_config(
        self,
        config_name: str = "config",
        overrides: dict[str, Any] | None = None,
    ) -> DictConfig:
        """
        Load configuration with optional overrides.

        Args:
            config_name: Name of the configuration file to load.
            overrides: Dictionary of configuration overrides.

        Returns:
            Loaded and merged configuration.

        Raises:
            FileNotFoundError: If no configuration files are found.
        """
        configs_to_merge = []

        # Load library configuration
        library_config_path = self._resolve_config_path(self.library_config_path)
        if library_config_path.exists():
            library_config = self._load_config_file(library_config_path, config_name)
            if library_config:
                configs_to_merge.append(library_config)
                logger.info(f"Loaded library config from {library_config_path}")

        # Load user configuration if specified
        if self.config_path:
            user_config_path = self._resolve_config_path(self.config_path)
            if user_config_path.exists():
                user_config = self._load_config_file(user_config_path, config_name)
                if user_config:
                    configs_to_merge.append(user_config)
                    logger.info(f"Loaded user config from {user_config_path}")

        if not configs_to_merge:
            raise FileNotFoundError(f"No configuration files found for '{config_name}'")

        # Merge configurations
        self._config = self._merge_configs(*configs_to_merge)

        # Apply overrides if provided
        if overrides:
            override_config = OmegaConf.create(overrides)
            self._config = OmegaConf.merge(self._config, override_config)
            logger.info(f"Applied {len(overrides)} configuration overrides")

        return self._config

    def initialize_config(
        self,
        config_name: str = "config",
        overrides: dict[str, Any] | None = None,
    ) -> DictConfig:
        """
        Initialize configuration with support for both project and library configurations.

        Args:
            config_name: Name of the configuration file to load.
            overrides: Dictionary of configuration overrides.

        Returns:
            Initialized configuration.
        """
        return self.load_config(config_name, overrides)

    def get_config(self, config_path: str | None = None) -> Any:
        """
        Retrieve a specific configuration value or the entire configuration.

        Args:
            config_path: Dot-separated path to the configuration value.
                        If None, returns the entire configuration.

        Returns:
            Configuration value or entire configuration.

        Raises:
            ValueError: If configuration is not loaded.
        """
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config first.")

        if config_path is None:
            return self._config

        return OmegaConf.select(self._config, config_path)

    def instantiate_from_config(
        self,
        config_path: str,
        _target_key: str = "_target_",
        **kwargs,
    ) -> Any:
        """
        Instantiate an object from configuration.

        This method provides similar functionality to Hydra's instantiate
        but using pure Python imports and OmegaConf.

        Args:
            config_path: Dot-separated path to the configuration.
            _target_key: Key in config that contains the target class path.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            Instantiated object.

        Raises:
            ValueError: If configuration is not found or target is not specified.
            ImportError: If the target class cannot be imported.
        """
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config first.")

        # Get the config at the specified path
        cfg = OmegaConf.select(self._config, config_path)
        if cfg is None:
            raise ValueError(f"No configuration found at path: {config_path}")

        # Convert to container for easier manipulation
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        if not isinstance(cfg_dict, dict):
            raise ValueError(f"Configuration at {config_path} must be a dictionary")

        # Get the target class
        target = cfg_dict.get(_target_key)
        if not target:
            raise ValueError(
                f"No '{_target_key}' specified in configuration at {config_path}",
            )

        # Remove the target from the config dict
        cfg_dict.pop(_target_key, None)

        # Merge with additional kwargs
        cfg_dict.update(kwargs)

        # Import and instantiate the target class
        return self._import_and_instantiate(target, **cfg_dict)

    def _import_and_instantiate(self, target: str, **kwargs) -> Any:
        """
        Import a class from a string path and instantiate it.

        Args:
            target: String path to the class (e.g., 'package.module.ClassName').
            **kwargs: Keyword arguments to pass to the constructor.

        Returns:
            Instantiated object.

        Raises:
            ImportError: If the module or class cannot be imported.
        """
        try:
            module_path, class_name = target.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls(**kwargs)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import and instantiate {target}: {e}") from e

    def _resolve_config_path(self, base_path: str | None = None) -> Path:
        """
        Resolve the configuration path with multiple fallback strategies.

        Args:
            base_path: Base path to resolve from. Defaults to self.config_path.

        Returns:
            Resolved absolute path to the configuration directory.

        Raises:
            ValueError: If no valid configuration path can be resolved.
        """
        if base_path is None:
            base_path = self.config_path

        if not base_path:
            raise ValueError("No configuration path provided")

        # Try multiple resolution strategies
        resolution_strategies = [
            # 1. Absolute path
            lambda: (Path(base_path).resolve() if Path(base_path).is_absolute() else None),
            # 2. Relative to the current file's directory
            lambda: Path(__file__).parent.joinpath(base_path).resolve(),
            # 3. Relative to project root
            lambda: self._get_project_root().joinpath(base_path).resolve(),
            # 4. Relative to tests directory
            lambda: Path(__file__).parent.parent.parent.joinpath("tests", "conf").resolve(),
            # 5. Fallback to absolute path resolution
            lambda: Path(os.path.abspath(base_path)).resolve(),
        ]

        # Try each strategy until a valid path is found
        for strategy in resolution_strategies:
            try:
                resolved_path = strategy()
                if resolved_path and resolved_path.exists() and resolved_path.is_dir():
                    return resolved_path
            except Exception:
                continue

        # If no valid path is found, raise an error
        raise ValueError(f"Could not resolve configuration path: {base_path}")

    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider: Name of the provider.

        Returns:
            Provider configuration with API key included.

        Raises:
            ValueError: If provider is unknown or configuration cannot be loaded.
        """
        if not self._config:
            self.load_config()

        if provider not in self._env_keys:
            raise ValueError(
                f"Unknown provider: {provider}. Available providers: {list(self._env_keys.keys())}",
            )

        try:
            # First, try to get provider config from main configuration
            provider_config = self.get_config(f"providers.{provider}")

            if provider_config is None:
                # If not found in main config, try loading from provider-specific file
                config_path = self._resolve_config_path()
                provider_path = config_path / "providers" / f"{provider}.yaml"

                if provider_path.exists():
                    logger.info(f"Loading provider config from file: {provider_path}")
                    provider_config = OmegaConf.load(provider_path)
                else:
                    logger.warning(f"No configuration found for provider: {provider}")
                    provider_config = OmegaConf.create({})

            # Convert to container and add API key
            provider_config_dict = OmegaConf.to_container(provider_config, resolve=True)
            if not isinstance(provider_config_dict, dict):
                provider_config_dict = {}

            provider_config_dict["api_key"] = self.get_api_key(provider)

            return provider_config_dict

        except Exception as e:
            logger.error(f"Error resolving provider configuration for {provider}: {e}")
            raise ValueError(
                f"Could not load configuration for provider {provider}: {e}",
            ) from e

    def get_resource_dirs(self) -> dict[str, Path]:
        """
        Get standard resource directories for the application using ResourceManager.

        Returns:
            Dictionary of standard resource directories.
        """
        return {
            "user_data_dir": self.resource_manager.user_data_dir,
            "cache_dir": self.resource_manager.user_cache_dir,
            "model_dir": self.resource_manager.model_dir,
            "dataset_dir": self.resource_manager.dataset_dir,
            "downloads_dir": self.resource_manager.downloads_dir,
            "archived_files_dir": self.resource_manager.downloads_dir / "archives",
        }

    def set_config_value(self, config_path: str, value: Any) -> None:
        """Set a configuration value at the specified path."""
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config first.")

        *parts, last = config_path.split(".")
        current = self._config
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[last] = value

    def save_config(self, file_path: str | Path) -> None:
        """
        Save the current configuration to a file.

        Args:
            file_path: Path where to save the configuration.

        Raises:
            ValueError: If configuration is not loaded.
        """
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config first.")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(self._config, file_path)
        logger.info(f"Configuration saved to {file_path}")


class ConfigurableComponent(Generic[T]):
    """
    Base class for components that require configuration.

    This class provides a standard interface for components that need access
    to configuration management functionality.

    Attributes:
        config_manager (ConfigManager): The configuration manager instance.
    """

    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize the configurable component.

        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self._config: DictConfig | None = None

    def load_config(self, config_path: str | None = None) -> None:
        """
        Load component-specific configuration.

        Args:
            config_path: Path to a specific configuration file.
                        If None, uses the config manager's configuration.
        """
        if config_path:
            self._config = OmegaConf.load(config_path)
            logger.info(f"Loaded component config from {config_path}")
        else:
            self._config = self.config_manager.get_config()

    @property
    def config(self) -> DictConfig:
        """
        Get the component's configuration.

        Returns:
            The component's configuration.

        Raises:
            ValueError: If configuration is not loaded.
        """
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_config first.")
        return self._config
