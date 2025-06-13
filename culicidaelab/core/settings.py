from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager
import threading

from culicidaelab.core.config_manager import ConfigManager
from culicidaelab.core.species_config import SpeciesConfig


class Settings:
    """
    User-friendly facade for CulicidaeLab configuration management.

    This class provides a simple interface to access configuration values,
    resource directories, and application settings. All actual operations
    are delegated to ConfigManager and ResourceManager.

    Usage Examples:
        settings = get_settings()
        settings.get_model_weights('detection')
        print(f"Config directory: {settings.config_dir}")
        print(f"Weights directory: {settings.weights_dir}")

        # Custom config directory
        custom_settings = get_settings(config_dir=custom_config_dir)
    """

    _instance: Optional["Settings"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, config_dir: str | Path | None = None) -> "Settings":
        """Create or return singleton instance, handling config directory changes."""
        with cls._lock:
            config_dir_str = str(Path(config_dir).resolve()) if config_dir else None

            # Create new instance if none exists or config directory changed
            if cls._instance is None or (config_dir_str and cls._instance._current_config_dir != config_dir_str):
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False

            return cls._instance

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """Initialize Settings facade with ConfigManager and ResourceManager."""
        if self._initialized:
            return

        # Store current config directory for singleton comparison
        self._current_config_dir = str(Path(config_dir).resolve()) if config_dir else None

        # Initialize managers
        self._config_manager = ConfigManager(config_path=config_dir)
        self._resource_manager = self._config_manager.resource_manager

        # Load initial configuration
        self._config_manager.initialize_config()

        # Cache for species config (lazy loaded)
        self._species_config: SpeciesConfig | None = None

        self._initialized = True

    # Configuration Access (delegated to ConfigManager)
    def get_config(self, path: str | None = None, default: Any = None) -> Any:
        """Get configuration value at specified path."""
        return self._config_manager.get_config(path) if path else self._config_manager.get_config()

    def set_config(self, path: str, value: Any) -> None:
        """Set configuration value at specified path."""
        self._config_manager.set_config_value(path, value)

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update multiple configuration values."""
        for path, value in updates.items():
            self.set_config(path, value)

    def save_config(self, file_path: str | Path | None = None) -> None:
        """Save current configuration to file."""
        if file_path is None:
            file_path = self._config_manager.config_path / "config.yaml"
        self._config_manager.save_config(file_path)

    def reload_config(self) -> None:
        """Reload configuration from files."""
        self._config_manager.initialize_config()
        # Clear species config cache
        self._species_config = None

    # Resource Directory Access (delegated to ResourceManager)
    def get_resource_dir(self, resource_type: str) -> Path:
        """Get path to a specific resource directory."""
        resource_dirs = self._resource_manager.get_all_directories()
        if resource_type not in resource_dirs:
            raise ValueError(f"Unknown resource type: {resource_type}. Available: {list(resource_dirs.keys())}")
        return resource_dirs[resource_type]

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
        """Configuration directory."""
        return Path(self._config_manager.config_path)

    # Species Configuration (lazy loaded)
    @property
    def species_config(self) -> SpeciesConfig:
        """Species configuration (lazily loaded)."""
        if self._species_config is None:
            # Try to load from species directory
            species_files = [
                self.config_dir / "species" / "species_classes.yaml",
                self.config_dir / "species" / "species_metadata.yaml",
                self.config_dir / "species.yaml",
            ]

            for species_file in species_files:
                if species_file.exists():
                    self._species_config = SpeciesConfig.from_yaml(species_file)
                    break

            if self._species_config is None:
                # Create default species config
                self._species_config = SpeciesConfig()

        return self._species_config

    # Dataset Management
    def get_dataset_path(self, dataset_type: str) -> Path:
        """Get path to dataset directory for specified type."""
        dataset_config = self.get_config(f"datasets.{dataset_type}")
        if not dataset_config:
            raise ValueError(f"Dataset type '{dataset_type}' not configured")

        if isinstance(dataset_config, dict) and "path" in dataset_config:
            dataset_path = dataset_config["path"]
        else:
            dataset_path = str(dataset_config)

        # Resolve path relative to dataset directory
        path = Path(dataset_path)
        if not path.is_absolute():
            path = self.dataset_dir / path

        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_datasets(self) -> list[str]:
        """Get list of configured dataset types."""
        datasets_config = self.get_config("datasets", {})
        return list(datasets_config.keys()) if datasets_config else []

    def add_dataset(self, dataset_type: str, dataset_path: str | Path) -> Path:
        """Add new dataset configuration."""
        self.set_config(f"datasets.{dataset_type}.path", str(dataset_path))
        return self.get_dataset_path(dataset_type)

    # Model Management
    def get_model_weights(self, model_type: str) -> Path:
        """Get path to model weights file."""
        # First try to get from models configuration
        model_config = self.get_config(f"models.{model_type}")
        if not model_config:
            # Try predictors configuration
            model_config = self.get_config(f"predictors.{model_type}")

        if not model_config:
            raise ValueError(f"Model type '{model_type}' not configured in models or predictors")

        if isinstance(model_config, dict) and "weights" in model_config:
            weights_file = model_config["weights"]
        elif isinstance(model_config, dict) and "model_path" in model_config:
            weights_file = model_config["model_path"]
        else:
            weights_file = str(model_config)

        # Resolve path relative to model directory
        weights_path = Path(weights_file)
        if not weights_path.is_absolute():
            weights_path = self.model_dir / weights_path

        return weights_path

    def list_model_types(self) -> list[str]:
        """Get list of available model types."""
        models_config = self.get_config("models", {})
        predictors_config = self.get_config("predictors", {})

        model_types = set()
        if models_config:
            model_types.update(models_config.keys())
        if predictors_config:
            model_types.update(predictors_config.keys())

        return list(model_types)

    def set_model_weights(self, model_type: str, weights_path: str | Path) -> None:
        """Set custom weights path for model type."""
        # Check if it's in models or predictors config
        if self.get_config(f"models.{model_type}"):
            self.set_config(f"models.{model_type}.weights", str(weights_path))
        elif self.get_config(f"predictors.{model_type}"):
            self.set_config(f"predictors.{model_type}.model_path", str(weights_path))
        else:
            # Default to models
            self.set_config(f"models.{model_type}.weights", str(weights_path))

    # Processing Parameters
    def get_processing_params(self, model_type: str | None = None) -> dict[str, Any]:
        """Get processing parameters for model inference."""
        if model_type:
            # Get model-specific params if available
            model_params = self.get_config(f"models.{model_type}.params", {})
            predictor_params = self.get_config(f"predictors.{model_type}.params", {})
            general_params = self.get_config("processing", {})

            # Merge params with model-specific taking precedence
            merged_params = {**general_params}
            merged_params.update(model_params)
            merged_params.update(predictor_params)
            return merged_params
        else:
            return self.get_config("processing", {})

    def set_processing_param(self, param_name: str, value: Any, model_type: str | None = None) -> None:
        """Set processing parameter."""
        if model_type:
            # Determine if it's in models or predictors
            if self.get_config(f"models.{model_type}"):
                self.set_config(f"models.{model_type}.params.{param_name}", value)
            elif self.get_config(f"predictors.{model_type}"):
                self.set_config(f"predictors.{model_type}.params.{param_name}", value)
            else:
                # Default to models
                self.set_config(f"models.{model_type}.params.{param_name}", value)
        else:
            self.set_config(f"processing.{param_name}", value)

    # API Key Management (delegated to ConfigManager)
    def get_api_key(self, provider: str) -> str | None:
        """Get API key for external provider."""
        return self._config_manager.get_api_key(provider)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for provider."""
        import os

        env_keys = {
            "kaggle": "KAGGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "roboflow": "ROBOFLOW_API_KEY",
        }

        if provider in env_keys:
            # Set environment variable
            os.environ[env_keys[provider]] = api_key
            # Also save to config
            self.set_config(f"api_keys.{provider}", api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Utility Methods (delegated to ResourceManager)
    @contextmanager
    def temp_workspace(self, prefix: str = "workspace"):
        """Context manager for temporary workspaces."""
        with self._resource_manager.temp_workspace(prefix) as workspace:
            yield workspace

    def clean_old_files(self, days: int = 5, include_cache: bool = True) -> dict[str, int]:
        """Clean up old files."""
        return self._resource_manager.clean_old_files(days, include_cache)

    def get_disk_usage(self) -> dict[str, dict[str, int | str]]:
        """Get disk usage statistics."""
        return self._resource_manager.get_disk_usage()

    # Summary and Validation
    def get_summary(self) -> dict[str, Any]:
        """Get summary of current settings configuration."""
        return {
            "config_dir": str(self.config_dir),
            "is_external_config": self._current_config_dir is not None,
            "model_dir": str(self.model_dir),
            "dataset_dir": str(self.dataset_dir),
            "cache_dir": str(self.cache_dir),
            "available_models": self.list_model_types(),
            "available_datasets": self.list_datasets(),
            "api_keys_configured": [
                provider for provider in ["kaggle", "huggingface", "roboflow"] if self.get_api_key(provider) is not None
            ],
            "disk_usage": self.get_disk_usage(),
        }

    def print_summary(self) -> None:
        """Print formatted configuration summary."""
        summary = self.get_summary()
        print("=== CulicidaeLab Settings Summary ===")
        for key, value in summary.items():
            if key == "disk_usage":
                print(f"{key.replace('_', ' ').title()}:")
                for dir_name, usage in value.items():
                    print(f"  {dir_name}: {usage.get('human_readable', 'N/A')}")
            elif isinstance(value, list):
                print(f"{key.replace('_', ' ').title()}: {', '.join(value) if value else 'None'}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

    def validate_setup(self) -> dict[str, bool]:
        """Validate current setup and return status."""
        results = {}

        # Check directories
        results["config_dir_exists"] = self.config_dir.exists()
        results["model_dir_exists"] = self.model_dir.exists()
        results["dataset_dir_exists"] = self.dataset_dir.exists()

        # Check configuration files
        required_files = ["config.yaml"]
        for file_name in required_files:
            file_path = self.config_dir / file_name
            results[f"{file_name.replace('.', '_')}_exists"] = file_path.exists()

        # Check if any models are configured
        results["models_configured"] = len(self.list_model_types()) > 0
        results["datasets_configured"] = len(self.list_datasets()) > 0

        # Check API keys
        for provider in ["kaggle", "huggingface", "roboflow"]:
            results[f"{provider}_api_key_configured"] = self.get_api_key(provider) is not None

        return results

    # Provider configurations
    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get provider configuration."""
        return self._config_manager.get_provider_config(provider)

    # Object instantiation from config
    def instantiate_from_config(self, config_path: str, **kwargs) -> Any:
        """Instantiate object from configuration."""
        return self._config_manager.instantiate_from_config(config_path, **kwargs)


# Global access function
def get_settings(config_dir: str | Path | None = None) -> Settings:
    """
    Get Settings singleton instance.

    This is the primary way to access Settings throughout the application.

    Args:
        config_dir: Optional external configuration directory

    Returns:
        Settings instance

    Examples:
        >>> settings = get_settings()
        >>> settings.get_model_weights('detection')
        >>> print(f"Config directory: {settings.config_dir}")
        >>>
        >>> # Custom config directory
        >>> custom_settings = get_settings(config_dir="/path/to/custom/config")
    """
    return Settings(config_dir)
