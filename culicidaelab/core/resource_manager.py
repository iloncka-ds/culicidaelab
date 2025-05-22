from pathlib import Path
import os
import shutil
import platform
import appdirs
import tempfile
import toml


class ResourceManager:
    """
    Centralized resource management for models, datasets, and temporary files
    that works across different operating systems.
    """

    def __init__(self):
        """
        Initialize resource paths with cross-platform compatibility.

        Args:
            app_name (str): Name of the application for directory naming
        """
        # Determine the base directories using appdirs
        config = toml.load("pyproject.toml")
        self.app_name = config.get("tool", {}).get("poetry", {}).get("name", "default_app_name")

        # User-specific data directory (for persistent storage)
        self.user_data_dir = Path(appdirs.user_data_dir(app_name))

        # User-specific cache directory
        self.user_cache_dir = Path(appdirs.user_cache_dir(app_name))

        # Temporary directory for runtime operations
        self.temp_dir = Path(tempfile.gettempdir()) / app_name

        # Subdirectories for different resource types
        self.model_dir = self.user_data_dir / "models"
        self.dataset_dir = self.user_data_dir / "datasets"
        self.downloads_dir = self.user_data_dir / "downloads"

        # Create necessary directories
        self._initialize_directories()

    def _initialize_directories(self):
        """
        Create necessary directories with proper permissions.
        """
        # Ensure user data directory exists
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Set appropriate permissions (Unix-like systems)
        if platform.system() != "Windows":
            try:
                os.chmod(self.user_data_dir, 0o755)
                os.chmod(self.model_dir, 0o755)
                os.chmod(self.dataset_dir, 0o755)
            except Exception as e:
                print(f"Warning: Could not set directory permissions: {e}")

    def get_model_path(self, model_name: str) -> Path:
        """
        Get a standardized path for a specific model.

        Args:
            model_name (str): Name of the model

        Returns:
            Path: Absolute path to the model directory
        """
        model_path = self.model_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    def get_dataset_path(self, dataset_name: str) -> Path:
        """
        Get a standardized path for a specific dataset.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            Path: Absolute path to the dataset directory
        """
        dataset_path = self.dataset_dir / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def create_temp_workspace(self, prefix: str = "workspace") -> Path:
        """
        Create a temporary workspace for runtime operations.

        Args:
            prefix (str): Prefix for the temporary directory

        Returns:
            Path: Path to the created temporary workspace
        """
        temp_workspace = self.temp_dir / f"{prefix}_{os.getpid()}"
        temp_workspace.mkdir(parents=True, exist_ok=True)
        return temp_workspace

    def clean_temp_workspace(self, workspace_path: Path, force: bool = False):
        """
        Clean up a temporary workspace.

        Args:
            workspace_path (Path): Path to the workspace to clean
            force (bool): If True, force remove even if not in temp directory
        """
        # Safety check to prevent accidental deletion
        if not force and not str(workspace_path).startswith(str(self.temp_dir)):
            raise ValueError("Cannot clean workspace outside of temp directory")

        try:
            shutil.rmtree(workspace_path)
        except FileNotFoundError:
            pass  # Directory already removed

    def clean_old_downloads(self, days: int = 5):
        """
        Clean up old download and temporary files.

        Args:
            days (int): Number of days after which files are considered old
        """
        import time

        current_time = time.time()
        for item in self.downloads_dir.iterdir():
            if current_time - item.stat().st_mtime > (days * 86400):
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"Could not remove {item}: {e}")

        # Clean temporary directory
        for item in self.temp_dir.iterdir():
            if current_time - item.stat().st_mtime > (days * 86400):
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"Could not remove {item}: {e}")
