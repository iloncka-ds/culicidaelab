import yaml
from importlib import resources
from pathlib import Path
from typing import Any, Dict
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError

from .config_models import CulicidaeLabConfig
from typing import Generic, TypeVar

T = TypeVar("T")


class ConfigManager:
    """
    Handles loading, merging, and validating configurations for the library.

    This manager implements a robust loading strategy:
    1. Loads default YAML configurations bundled with the library.
    2. Loads user-provided YAML configurations from a specified directory.
    3. Merges the user's configuration on top of the defaults.
    4. Validates the final merged configuration against Pydantic models.
    """

    def __init__(self, user_config_dir: str | Path | None = None):
        self.user_config_dir = Path(user_config_dir) if user_config_dir else None
        self.default_config_path = self._get_default_config_path()
        self.config: CulicidaeLabConfig = self._load()

    def _get_default_config_path(self) -> Path:
        """Reliably find the path to the bundled 'conf' directory."""
        try:
            # The modern and correct way to access package data
            return resources.files("culicidaelab") / "conf"
        except (ModuleNotFoundError, FileNotFoundError):
            # Fallback for development environments where the package might not be installed
            dev_path = Path(__file__).parent.parent / "conf"
            if dev_path.exists():
                return dev_path
            raise FileNotFoundError(
                "Could not find the default 'conf' directory. "
                "Ensure the 'culicidaelab' package is installed correctly or you are in the project root."
            )

    def _load_config_from_dir(self, config_dir: Path) -> Dict[str, Any]:
        """Loads all YAML files from a directory into a nested dictionary."""
        config_dict = {}
        if not config_dir or not config_dir.is_dir():
            return config_dict

        for yaml_file in config_dir.glob("**/*.yaml"):
            try:
                with yaml_file.open("r") as f:
                    data = yaml.safe_load(f)
                    if data is None:  # Handle empty YAML files
                        continue

                # Create nested structure based on file path, e.g., 'datasets/classification.yaml' -> {'datasets': {'classification': ...}}
                relative_path = yaml_file.relative_to(config_dir)
                keys = list(relative_path.parts[:-1]) + [relative_path.stem]

                d = config_dict
                for key in keys[:-1]:
                    d = d.setdefault(key, {})
                d[keys[-1]] = data
            except Exception as e:
                print(f"Warning: Could not load or parse {yaml_file}: {e}")

        return config_dict

    def _load(self) -> CulicidaeLabConfig:
        """Executes the full load, merge, and validation process."""
        # 1. Load default configs
        default_config_dict = self._load_config_from_dir(self.default_config_path)

        # 2. Load user configs
        user_config_dict = self._load_config_from_dir(self.user_config_dir)

        # 3. Merge user configs on top of defaults
        merged_config = _deep_merge(user_config_dict, default_config_dict)

        # 4. Validate with Pydantic
        try:
            validated_config = CulicidaeLabConfig(**merged_config)
            return validated_config
        except ValidationError as e:
            print(f"FATAL: Configuration validation failed. Please check your YAML files or environment variables.")
            print(e)
            raise

    def get_config(self) -> CulicidaeLabConfig:
        """Returns the fully validated Pydantic configuration object."""
        return self.config

    def save_config(self, file_path: str | Path) -> None:
        """
        Save the current configuration state to a YAML file.

        Args:
            file_path: Path where to save the configuration.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Dump the Pydantic model to a dictionary, then to YAML
        config_dict = self.config.model_dump(mode="json")

        # Use OmegaConf to save for compatibility if needed, or just yaml
        OmegaConf.save(config=config_dict, f=path)

    def instantiate_from_config(self, config_obj: Any, **kwargs) -> Any:
        """
        Instantiate an object from a Pydantic config model.

        Args:
            config_obj: A Pydantic model instance (e.g., a predictor config).
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            Instantiated object.
        """
        if not hasattr(config_obj, "_target_"):
            raise ValueError(f"Target key '_target_' not found in configuration object")

        target_path = config_obj._target_

        # Get parameters from the model, excluding the target key
        config_params = config_obj.model_dump()
        config_params.pop("_target_", None)

        # Merge with any runtime kwargs
        config_params.update(kwargs)

        try:
            module_path, class_name = target_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls(**config_params)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import and instantiate '{target_path}': {e}")


class ConfigurableComponent(Generic[T]):
    """
    Base class for components that require configuration.

    This class provides a standard interface for components that need access
    to configuration management functionality with automatic configuration updates.

    Attributes:
        config_manager (ConfigManager): The configuration manager instance.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        config_path: str | None = None,
    ) -> None:
        """
        Initialize the configurable component.

        Args:
            config_manager: Configuration manager instance.
            config_path: Optional path to component-specific configuration.
        """
        self.config_manager = config_manager
        self._component_config: DictConfig | None = None
        self._config_path = config_path
        self._config_hash: str | None = None
        self._auto_reload = True

        # Load initial configuration
        self.load_config(config_path)

    def load_config(self, config_path: str | None = None) -> None:
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
        self,
        old_hash: str | None,
        new_hash: str | None,
    ) -> None:
        """
        Hook called when configuration changes are detected.

        Args:
            old_hash: Previous configuration hash.
            new_hash: New configuration hash.
        """
        pass

    def update_config(self, updates: dict[str, Any], save: bool = False) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates.
            save: Whether to save changes to the configuration manager.
        """
        if self._component_config is None:
            raise ValueError(
                "Component configuration not loaded. Call load_config() first.",
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

    def check_required_config(self) -> dict[str, bool]:
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
                "Component configuration not loaded. Call load_config() first.",
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
    def config_path(self) -> str | None:
        """Get the configuration path for this component."""
        return self._config_path

    @property
    def is_config_loaded(self) -> bool:
        """Check if configuration is loaded."""
        return self._component_config is not None
