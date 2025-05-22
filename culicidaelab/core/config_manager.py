"""
Centralized configuration management module using Hydra and OmegaConf.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar, Generic
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from hydra.utils import instantiate
from dotenv import load_dotenv


from culicidaelab.core.resource_manager import ResourceManager

T = TypeVar("T")


class ConfigManager:
    """Centralized configuration management using Hydra and OmegaConf."""

    _instance = None
    _env_keys = {
        "kaggle": "KAGGLE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "roboflow": "ROBOFLOW_API_KEY",
    }
    _default_config_path = "../conf"
    resource_manager = ResourceManager()

    def __new__(cls, library_config_path: _default_config_path, config_path: str | None = None, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, library_config_path: _default_config_path, config_path: str | None = None, **kwargs):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.config_path = config_path
        self._config: DictConfig | None = None
        self.library_config_path = library_config_path

        self.initialized = True
        self._load_env_vars()

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        current_file = Path(__file__).resolve()
        # Navigate up multiple levels to find the project root
        project_root = current_file
        while project_root.parent != project_root:
            if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
                return project_root
            project_root = project_root.parent
        raise FileNotFoundError("Could not determine project root directory")

    def _load_env_vars(self) -> None:
        """Load environment variables from .env file."""
        env_path = self._get_project_root() / ".env"
        load_dotenv(env_path)

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider from environment variables.

        Args:
            provider (str): Name of the provider (kaggle, huggingface, roboflow)

        Returns:
            Optional[str]: API key if found, None otherwise
        """
        if provider not in self._env_keys:
            raise ValueError(f"Unknown provider: {provider}")

        return os.getenv(self._env_keys[provider])

    def load_config(self, config_name: str = "config", overrides: list[str] = None, **kwargs) -> DictConfig:
        """
        Load configuration with optional overrides.

        Args:
            config_name (str): Name of the configuration file
            overrides (List[str]): List of override strings in Hydra format
            **kwargs: Additional overrides as keyword arguments

        Returns:
            DictConfig: Loaded configuration
        """
        try:
            # Clear any existing Hydra initialization
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            with initialize(version_base=None, config_path=self.library_config_path):
                hydra_overrides = overrides or []

                # Convert kwargs to Hydra override format
                for key, value in kwargs.items():
                    hydra_overrides.append(f"+{key}={value}")

                self._config = compose(
                    config_name=config_name,
                    overrides=hydra_overrides,
                    return_hydra_config=True,
                )

            return self._config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise

    def initialize_config(self, config_name: str = "config", overrides: list[str] = None):
        """Initialize configuration with support for both project and library configurations."""
        overrides = overrides or []

        if self.config_path:
            try:
                with initialize(version_base=None, config_path=self.config_path):
                    cfg = compose(config_name=config_name, overrides=overrides)
                    return cfg
            except Exception as e:
                print(f"Failed to initialize customconfig: {e}")

        with initialize(version_base=None, config_path=self.library_config_path):
            cfg = compose(config_name=config_name, overrides=overrides)
            return cfg

    def get_config(self, config_path: str | None = None) -> Any:
        """
        Retrieve a specific configuration value.

        Args:
            config_path (Optional[str]): Dot-separated path to the configuration value

        Returns:
            Any: Configuration value
        """
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config first.")

        if config_path is None:
            return self._config

        return OmegaConf.select(self._config, config_path)

    def instantiate(self, config_path: str, **kwargs) -> Any:
        """Instantiate an object from config using Hydra's instantiate."""
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config first.")

        # Get the config at the specified path
        cfg = OmegaConf.select(self._config, config_path)
        if cfg is None:
            raise ValueError(f"No configuration found at path: {config_path}")

        return instantiate(cfg, **kwargs)

    def _resolve_config_path(self, base_path: str | None = None) -> Path:
        """
        Resolve the configuration path with multiple fallback strategies.

        Args:
            base_path (Optional[str]): Base path to resolve from. Defaults to self.config_path.

        Returns:
            Path: Resolved absolute path to the configuration directory
        """
        # If no base path is provided, use the instance's config_path
        if base_path is None:
            base_path = self.config_path

        # Try multiple resolution strategies
        resolution_strategies = [
            # 1. Absolute path
            lambda: Path(base_path).resolve() if Path(base_path).is_absolute() else None,
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
            provider (str): Name of the provider

        Returns:
            Dict[str, Any]: Provider configuration with API key
        """
        if not self._config:
            self.load_config()

        if provider not in self._env_keys:
            raise ValueError(f"Unknown provider: {provider}")

        # Resolve the configuration path
        try:
            config_path = self._resolve_config_path()

            # Construct provider path with cross-platform path joining
            provider_path = config_path / "providers" / f"{provider}.yaml"

            print(f"Resolved Config Path: {config_path}")
            print(f"Resolved Provider Path: {provider_path}")

            # Check if the provider config exists in the loaded configuration
            provider_config = self._config.get(f"providers.{provider}", None)
            if provider_config is not None:
                print(f"Provider config found in loaded config: {provider}")
                provider_config = OmegaConf.to_container(provider_config, resolve=True)
            else:
                # If not found in main config, try loading from provider-specific file
                if not provider_path.exists():
                    raise ValueError(f"Provider config file does not exist: {provider_path}")

                print(f"Loading provider config from file: {provider_path}")
                provider_config = OmegaConf.load(str(provider_path))
                provider_config = OmegaConf.to_container(provider_config, resolve=True)

            # Add the API key from environment variables
            provider_config["api_key"] = self.get_api_key(provider)

            return provider_config

        except Exception as e:
            print(f"Error resolving provider configuration: {e}")
            raise

    def get_resource_dirs(self) -> dict[str, Path]:
        """
        Get standard resource directories for the application using ResourceManager.



        Returns:
            Dict[str, Path]: Dictionary of standard resource directories
        """

        return {
            "user_data_dir": self.resource_manager.user_data_dir,
            "cache_dir": self.resource_manager.user_cache_dir,
            "model_dir": self.resource_manager.model_dir,
            "dataset_dir": self.resource_manager.dataset_dir,
            "downloads_dir": self.resource_manager.downloads_dir,
            "archived_files_dir": self.resource_manager.downloads_dir
            / "archives",  # Specific directory for archived files
        }


class ConfigurableComponent(Generic[T]):
    """Base class for components that require configuration."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._config: DictConfig | None = None

    def load_config(self, config_path: str | None = None) -> None:
        """Load component-specific configuration."""
        if config_path:
            self._config = OmegaConf.load(config_path)
        else:
            self._config = self.config_manager.get_config()

    @property
    def config(self) -> DictConfig:
        """Get the component's configuration."""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_config first.")
        return self._config
