import yaml
from importlib import resources
from pathlib import Path
from typing import Any, Dict
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError

from .config_models import CulicidaeLabConfig
from typing import Generic, TypeVar

T = TypeVar("T")

def _deep_merge(source: Dict, destination: Dict) -> Dict:
    """Recursively merge two dictionaries. Source values overwrite destination."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

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
        if not hasattr(config_obj, "target_"):
            raise ValueError(f"Target key '_target_' not found in configuration object")

        target_path = config_obj.target_

        # Get parameters from the model, excluding the target key
        config_params = config_obj.model_dump()
        config_params.pop("target_", None)

        # Merge with any runtime kwargs
        config_params.update(kwargs)

        try:
            module_path, class_name = target_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls(**config_params)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import and instantiate '{target_path}': {e}")

