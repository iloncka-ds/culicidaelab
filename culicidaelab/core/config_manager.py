# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, Optional, Union, Generic, TypeVar
import shutil
import os
import importlib
import threading
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv


from .resource_manager import ResourceManager

T = TypeVar("T")


class ConfigManager:
    """
    Centralized configuration management using OmegaConf.

    This class implements the singleton pattern to ensure consistent configuration
    access across the application. It handles loading configurations from YAML files,
    environment variables, and provides utilities for instantiating configured objects.

    Attributes:
        resource_manager (ResourceManager): Manages application resource directories.
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    _env_keys: Dict[str, str] = {
        "kaggle": "KAGGLE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "roboflow": "ROBOFLOW_API_KEY",
    }

    _default_config_path: str = "../conf"

    def __new__(
        cls,
        library_config_path: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> "ConfigManager":
        """
        Create or return existing ConfigManager instance (Singleton pattern).

        Args:
            library_config_path: Path to library configuration directory.
            config_path: Path to user configuration directory.
            **kwargs: Additional keyword arguments.

        Returns:
            ConfigManager instance.
        """
        with cls._lock:
            config_path_str = str(Path(config_path).resolve()) if config_path else None

            # Create new instance if none exists or config path changed
            if cls._instance is None or (
                config_path_str and cls._instance.config_path != config_path_str
            ):
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False

            return cls._instance

    def __init__(
        self,
        library_config_path: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ConfigManager instance.

        Args:
            library_config_path: Path to library configuration directory.
            config_path: Path to user configuration directory.
            **kwargs: Additional keyword arguments.
        """
        if self._initialized:
            return

        self.library_config_path = self._get_library_config_path(library_config_path)
        self.config_path = self._resolve_config_path(config_path)

        # Initialize resource manager
        self.resource_manager = ResourceManager()

        # Load environment variables
        self._load_env_vars()

        # Configuration will be loaded when initialize_config is called
        self.config: Optional[DictConfig] = None

        self._initialized = True

    def _get_library_config_path(self, library_config_path: Optional[str]) -> Path:
        """Get the library configuration path."""
        if library_config_path:
            return Path(library_config_path).resolve()

        # Try to find library config directory
        possible_paths = [
            Path(__file__).parent / "conf",
            Path(__file__).parent.parent / "conf",
            Path(__file__).parent.parent / "culicidaelab" / "conf",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Fallback to default
        return Path(__file__).parent / "conf"

    def _get_project_root(self) -> Path:
        """
        Get the project root directory by searching for configuration files.

        Returns:
            Path to the project root directory.

        Raises:
            FileNotFoundError: If project root cannot be determined.
        """
        current_path = Path.cwd()

        # Look for configuration indicators
        config_indicators = ["pyproject.toml", "setup.py", "conf", "config", ".git"]

        for parent in [current_path] + list(current_path.parents):
            if any((parent / indicator).exists() for indicator in config_indicators):
                return parent

        raise FileNotFoundError("Could not determine project root directory")

    def _load_env_vars(self) -> None:
        """
        Load environment variables from .env file if it exists.
        """
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    def get_api_key(self, provider: str) -> Optional[str]:
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
                f"Unknown provider: {provider}. Available: {list(self._env_keys.keys())}"
            )

        env_key = self._env_keys[provider]
        return os.getenv(env_key)

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

        merged = configs[0]
        for config in configs[1:]:
            merged = OmegaConf.merge(merged, config)

        return merged

    def _load_config_file(
        self,
        config_path: Path,
        config_name: str,
    ) -> Optional[DictConfig]:
        """
        Load a configuration file from the specified path.

        Args:
            config_path: Directory containing the configuration file.
            config_name: Name of the configuration file (without extension).

        Returns:
            Loaded configuration or None if file doesn't exist.
        """
        yaml_file = config_path / f"{config_name}.yaml"
        yml_file = config_path / f"{config_name}.yml"

        for config_file in [yaml_file, yml_file]:
            if config_file.exists():
                try:
                    return OmegaConf.load(config_file)
                except Exception as e:
                    print(f"Warning: Could not load {config_file}: {e}")

        return None

    def ensure_config_structure(
        self, config_dir: Path, is_external: bool = False
    ) -> None:
        """
        Ensure configuration directory has required structure.

        Args:
            config_dir: Configuration directory path
            is_external: Whether this is an external (user-provided) config directory
        """
        if not is_external:
            # For default config, create if missing
            config_dir.mkdir(parents=True, exist_ok=True)

        # Check for required files and directories based on library structure
        required_structure = self._get_required_config_structure()
        missing_files = []
        missing_dirs = []

        # Check files
        for file_path in required_structure["files"]:
            full_path = config_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        # Check directories
        for dir_path in required_structure["directories"]:
            full_path = config_dir / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)

        if missing_files or missing_dirs:
            if is_external:
                # Copy from library config
                self._repair_config_structure(config_dir, missing_dirs, missing_files)
            else:
                # Create minimal default configs
                self._create_default_configs(config_dir, missing_files, missing_dirs)

    def _get_required_config_structure(self) -> Dict[str, list[str]]:
        """Get the required configuration structure."""
        return {
            "files": [
                "config.yaml",
                "datasets/detection.yaml",
                "datasets/segmentation.yaml",
                "datasets/classification.yaml",
                "predictors/detector.yaml",
                "predictors/segmenter.yaml",
                "predictors/classifier.yaml",
                "species/species_classes.yaml",
                "species/species_metadata.yaml",
                "providers/kaggle.yaml",
                "providers/huggingface.yaml",
                "providers/roboflow.yaml",
            ],
            "directories": [
                "datasets",
                "predictors",
                "species",
                "providers",
                "app_settings",
            ],
        }

    def _repair_config_structure(
        self, config_dir: Path, missing_dirs: list[str], missing_files: list[str]
    ) -> None:
        """
        Repair missing configuration files by copying from defaults.

        Args:
            config_dir: Target configuration directory
            missing_dirs: List of missing directories
            missing_files: List of missing files
        """
        # Create missing directories
        for dir_path in missing_dirs:
            (config_dir / dir_path).mkdir(parents=True, exist_ok=True)

        # Copy missing files from library config
        for file_path in missing_files:
            source_file = self.library_config_path / file_path
            target_file = config_dir / file_path

            # Ensure target directory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)

            if source_file.exists():
                shutil.copy2(source_file, target_file)
            else:
                # Create minimal default if library file doesn't exist
                self._create_minimal_config_file(target_file)

    def _create_default_configs(
        self, config_dir: Path, missing_files: list[str], missing_dirs: list[str]
    ) -> None:
        """
        Create minimal default configuration files.

        Args:
            config_dir: Configuration directory
            missing_files: List of missing files to create
            missing_dirs: List of missing directories to create
        """
        # Create missing directories
        for dir_path in missing_dirs:
            (config_dir / dir_path).mkdir(parents=True, exist_ok=True)

        # Default configurations
        defaults = {
            "config.yaml": {
                "processing": {
                    "batch_size": 32,
                    "confidence_threshold": 0.5,
                    "device": "auto",
                },
                "directories": {},
                "api_keys": {},
                "app_settings": {"environment": "development", "logging_level": "INFO"},
            },
            "datasets/detection.yaml": {
                "name": "mosquito_detection",
                "path": "datasets/detection",
                "format": "yolo",
                "classes": ["mosquito"],
            },
            "datasets/segmentation.yaml": {
                "name": "mosquito_segmentation",
                "path": "datasets/segmentation",
                "format": "coco",
                "classes": ["mosquito"],
            },
            "datasets/classification.yaml": {
                "name": "species_classification",
                "path": "datasets/classification",
                "format": "imagefolder",
                "classes": [],
            },
            "predictors/detector.yaml": {
                "_target_": "culicidaelab.predictors.detector.Detector",
                "model_path": "yolov8n.pt",
                "confidence": 0.5,
                "device": "auto",
            },
            "predictors/segmenter.yaml": {
                "_target_": "culicidaelab.predictors.segmenter.Segmenter",
                "model_path": "sam_b.pt",
                "device": "auto",
            },
            "predictors/classifier.yaml": {
                "_target_": "culicidaelab.predictors.classifier.Classifier",
                "model_path": "resnet50.pt",
                "device": "auto",
            },
            "species/species_classes.yaml": {"classes": [], "taxonomy": {}},
            "species/species_metadata.yaml": {"metadata": {}, "descriptions": {}},
            "providers/kaggle.yaml": {
                "_target_": "culicidaelab.providers.KaggleProvider",
                "api_key": "${oc.env:KAGGLE_API_KEY}",
            },
            "providers/huggingface.yaml": {
                "_target_": "culicidaelab.providers.HuggingFaceProvider",
                "api_key": "${oc.env:HUGGINGFACE_API_KEY}",
            },
            "providers/roboflow.yaml": {
                "_target_": "culicidaelab.providers.RoboflowProvider",
                "api_key": "${oc.env:ROBOFLOW_API_KEY}",
            },
        }

        for file_path in missing_files:
            if file_path in defaults:
                config_file = config_dir / file_path
                config_file.parent.mkdir(parents=True, exist_ok=True)
                OmegaConf.save(defaults[file_path], config_file)

    def _create_minimal_config_file(self, file_path: Path) -> None:
        """Create a minimal configuration file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save({}, file_path)

    def load_configurations(self, config_dir: Path) -> DictConfig:
        """
        Load and merge all configuration files from directory.

        Args:
            config_dir: Configuration directory path

        Returns:
            Merged configuration
        """
        configs = []

        # Load main config
        main_config = self._load_config_file(config_dir, "config")
        if main_config:
            configs.append(main_config)

        # Load configurations from subdirectories
        subdirs = ["datasets", "predictors", "species", "providers", "app_settings"]

        for subdir in subdirs:
            subdir_path = config_dir / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                subdir_configs = {}

                for config_file in subdir_path.glob("*.yaml"):
                    config_name = config_file.stem
                    config_data = self._load_config_file(subdir_path, config_name)
                    if config_data:
                        subdir_configs[config_name] = config_data

                if subdir_configs:
                    configs.append(OmegaConf.create({subdir: subdir_configs}))

        return self._merge_configs(*configs) if configs else OmegaConf.create({})

    def load_config(
        self,
        config_name: str = "config",
        overrides: Optional[Dict[str, Any]] = None,
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
        configs = []

        # Load from library config path first (default values)
        library_config = self._load_config_file(self.library_config_path, config_name)
        if library_config:
            configs.append(library_config)

        # Load from user config path (overrides)
        user_config = self._load_config_file(Path(self.config_path), config_name)
        if user_config:
            configs.append(user_config)

        if not configs:
            raise FileNotFoundError(f"No configuration files found for '{config_name}'")

        # Merge configurations
        merged_config = self._merge_configs(*configs)

        # Apply overrides if provided
        if overrides:
            override_config = OmegaConf.create(overrides)
            merged_config = OmegaConf.merge(merged_config, override_config)

        return merged_config

    def initialize_config(
        self,
        config_name: str = "config",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> DictConfig:
        """
        Initialize configuration with support for both project and library configurations.

        Args:
            config_name: Name of the configuration file to load.
            overrides: Dictionary of configuration overrides.

        Returns:
            Initialized configuration.
        """
        # Ensure config directory structure exists
        config_dir = Path(self.config_path)
        self.ensure_config_structure(config_dir, is_external=True)

        # Load all configurations from directory
        self.config = self.load_configurations(config_dir)

        # Apply overrides if provided
        if overrides:
            override_config = OmegaConf.create(overrides)
            self.config = OmegaConf.merge(self.config, override_config)

        return self.config

    def get_config(self, config_path: Optional[str] = None) -> Any:
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
        if self.config is None:
            raise ValueError(
                "Configuration not loaded. Call initialize_config() first."
            )

        if config_path is None:
            return self.config

        try:
            return OmegaConf.select(self.config, config_path)
        except Exception:
            return None

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
        config = self.get_config(config_path)
        if config is None:
            raise ValueError(f"Configuration not found at path: {config_path}")

        if not isinstance(config, DictConfig):
            raise ValueError(
                f"Configuration at {config_path} is not a valid config object"
            )

        if _target_key not in config:
            raise ValueError(f"Target key '{_target_key}' not found in configuration")

        target = config[_target_key]

        # Merge config parameters with kwargs (kwargs take precedence)
        config_params = {k: v for k, v in config.items() if k != _target_key}
        config_params.update(kwargs)

        return self._import_and_instantiate(target, **config_params)

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
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls(**kwargs)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import and instantiate '{target}': {e}")

    def _resolve_config_path(self, base_path: Optional[str] = None) -> str:
        """
        Resolve the configuration path with multiple fallback strategies.

        Args:
            base_path: Base path to resolve from. Defaults to self.config_path.

        Returns:
            Resolved absolute path to the configuration directory.

        Raises:
            ValueError: If no valid configuration path can be resolved.
        """
        if base_path:
            path = Path(base_path)
            if path.is_absolute():
                return str(path.resolve())
            else:
                # Make relative to current working directory
                return str(Path.cwd() / path)

        # Try to find configuration directory automatically
        search_paths = [
            Path.cwd() / "conf",
            Path.cwd() / "config",
            Path.cwd() / "configs",
            Path.cwd() / self._default_config_path.lstrip("../"),
        ]

        # Also try project root
        try:
            project_root = self._get_project_root()
            search_paths.extend(
                [
                    project_root / "conf",
                    project_root / "config",
                    project_root / "configs",
                ]
            )
        except FileNotFoundError:
            pass

        for path in search_paths:
            if path.exists():
                return str(path.resolve())

        # Default fallback - create conf directory in current working directory
        default_path = Path.cwd() / "conf"
        return str(default_path.resolve())

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider: Name of the provider.

        Returns:
            Provider configuration with API key included.

        Raises:
            ValueError: If provider is unknown or configuration cannot be loaded.
        """
        provider_config = self.get_config(f"providers.{provider}")
        if provider_config is None:
            raise ValueError(f"No configuration found for provider: {provider}")

        # Convert to dictionary and resolve any OmegaConf references
        config_dict = OmegaConf.to_container(provider_config, resolve=True)

        # Add API key from environment if not already present
        api_key = self.get_api_key(provider)
        if api_key and "api_key" not in config_dict:
            config_dict["api_key"] = api_key

        return config_dict

    def get_resource_dirs(self) -> Dict[str, Path]:
        """
        Get standard resource directories for the application using ResourceManager.

        Returns:
            Dictionary of standard resource directories.
        """
        return self.resource_manager.get_resource_directories()

    def set_config_value(self, config_path: str, value: Any) -> None:
        """
        Set a configuration value at the specified path.

        Args:
            config_path: Dot-separated path to the configuration value.
            value: Value to set.

        Raises:
            ValueError: If configuration is not loaded.
        """
        if self.config is None:
            raise ValueError(
                "Configuration not loaded. Call initialize_config() first."
            )

        OmegaConf.set(self.config, config_path, value)

    def save_config(self, file_path: Union[str, Path]) -> None:
        """
        Save the current configuration to a file.

        Args:
            file_path: Path where to save the configuration.

        Raises:
            ValueError: If configuration is not loaded.
        """
        if self.config is None:
            raise ValueError(
                "Configuration not loaded. Call initialize_config() first."
            )

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, path)


class ConfigurableComponent(Generic[T]):
    """
    Base class for components that require configuration.

    This class provides a standard interface for components that need access
    to configuration management functionality with automatic configuration updates.

    Attributes:
        config_manager (ConfigManager): The configuration manager instance.
    """

    def __init__(
        self, config_manager: ConfigManager, config_path: Optional[str] = None
    ) -> None:
        """
        Initialize the configurable component.

        Args:
            config_manager: Configuration manager instance.
            config_path: Optional path to component-specific configuration.
        """
        self.config_manager = config_manager
        self._component_config: Optional[DictConfig] = None
        self._config_path = config_path
        self._config_hash: Optional[str] = None
        self._auto_reload = True

        # Load initial configuration
        self.load_config(config_path)

    def load_config(self, config_path: Optional[str] = None) -> None:
        """
        Load component-specific configuration.

        Args:
            config_path: Path to a specific configuration file.
                        If None, uses the stored config path or manager's configuration.
        """
        # Update config path if provided
        if config_path is not None:
            self._config_path = config_path

        # Load configuration
        if self._config_path:
            self._component_config = self.config_manager.get_config(self._config_path)
        else:
            self._component_config = self.config_manager.get_config()

        # Update configuration hash for change detection
        if self._component_config is not None:
            self._config_hash = self._compute_config_hash(self._component_config)

        # Call hook for subclasses to handle configuration changes
        self._on_config_loaded()

    def reload_config(self) -> bool:
        """
        Reload configuration and check for changes.

        Returns:
            True if configuration changed, False otherwise.
        """
        old_hash = self._config_hash
        self.load_config()

        # Check if configuration actually changed
        changed = old_hash != self._config_hash

        if changed:
            self._on_config_changed(old_hash, self._config_hash)

        return changed

    def _compute_config_hash(self, config: DictConfig) -> str:
        """
        Compute a hash of the configuration for change detection.

        Args:
            config: Configuration to hash.

        Returns:
            Configuration hash string.
        """
        import hashlib

        # Convert config to string representation and hash it
        config_str = OmegaConf.to_yaml(config)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _on_config_loaded(self) -> None:
        """
        Hook called when configuration is loaded.
        Subclasses can override this to perform initialization.
        """
        pass

    def _on_config_changed(
        self, old_hash: Optional[str], new_hash: Optional[str]
    ) -> None:
        """
        Hook called when configuration changes are detected.

        Args:
            old_hash: Previous configuration hash.
            new_hash: New configuration hash.
        """
        pass

    def update_config(self, updates: Dict[str, Any], save: bool = False) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates.
            save: Whether to save changes to the configuration manager.
        """
        if self._component_config is None:
            raise ValueError(
                "Component configuration not loaded. Call load_config() first."
            )

        # Apply updates to component config
        for key, value in updates.items():
            OmegaConf.set(self._component_config, key, value)

        # Update hash
        self._config_hash = self._compute_config_hash(self._component_config)

        # Optionally save to config manager
        if save and self._config_path:
            for key, value in updates.items():
                full_path = f"{self._config_path}.{key}" if self._config_path else key
                self.config_manager.set_config_value(full_path, value)

        # Notify about configuration change
        self._on_config_changed(None, self._config_hash)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.

        Args:
            key: Dot-separated key path.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        if self._component_config is None:
            return default

        try:
            return OmegaConf.select(self._component_config, key, default=default)
        except Exception:
            return default

    def set_config_value(self, key: str, value: Any, save: bool = False) -> None:
        """
        Set a specific configuration value.

        Args:
            key: Dot-separated key path.
            value: Value to set.
            save: Whether to save to configuration manager.
        """
        self.update_config({key: value}, save=save)

    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        Subclasses should override this to implement validation logic.

        Returns:
            True if configuration is valid, False otherwise.
        """
        return self._component_config is not None

    def get_required_config_keys(self) -> list[str]:
        """
        Get list of required configuration keys.
        Subclasses should override this to specify required keys.

        Returns:
            List of required configuration keys.
        """
        return []

    def check_required_config(self) -> Dict[str, bool]:
        """
        Check if all required configuration keys are present.

        Returns:
            Dictionary mapping required keys to their presence status.
        """
        required_keys = self.get_required_config_keys()
        return {key: self.get_config_value(key) is not None for key in required_keys}

    def enable_auto_reload(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic configuration reloading.

        Args:
            enabled: Whether to enable auto-reload.
        """
        self._auto_reload = enabled

    @property
    def config(self) -> DictConfig:
        """
        Get the component's configuration with automatic reload check.

        Returns:
            The component's configuration.

        Raises:
            ValueError: If configuration is not loaded.
        """
        if self._component_config is None:
            raise ValueError(
                "Component configuration not loaded. Call load_config() first."
            )

        # Auto-reload if enabled (check for changes)
        if self._auto_reload:
            # Only check for changes, don't force reload every time
            try:
                current_config = (
                    self.config_manager.get_config(self._config_path)
                    if self._config_path
                    else self.config_manager.get_config()
                )
                if current_config is not None:
                    current_hash = self._compute_config_hash(current_config)
                    if current_hash != self._config_hash:
                        self.reload_config()
            except Exception:
                # If reload fails, continue with current config
                pass

        return self._component_config

    @property
    def config_path(self) -> Optional[str]:
        """Get the configuration path for this component."""
        return self._config_path

    @property
    def is_config_loaded(self) -> bool:
        """Check if configuration is loaded."""
        return self._component_config is not None
