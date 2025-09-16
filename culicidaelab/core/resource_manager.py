"""Centralized resource management for models, datasets, and files.

This module provides cross-platform resource management with error handling,
logging, and comprehensive path management capabilities.
"""

import hashlib
import logging
import os
import platform
import shutil
import tempfile
import time
import toml
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import appdirs

logger = logging.getLogger(__name__)


class ResourceManager:
    """Centralized resource management for models, datasets, and temporary files.

    This class provides thread-safe operations for managing application resources,
    including models, datasets, cache files, and temporary workspaces.

    Args:
        app_name (str, optional): Application name used for directory naming.
            If None, the name is loaded from `pyproject.toml`.
        custom_base_dir (str | Path, optional): Custom base directory for all
            resources. If None, system-default paths are used.

    Attributes:
        app_name (str): The application name.
        user_data_dir (Path): User-specific data directory for persistent storage.
        user_cache_dir (Path): User-specific cache directory for temporary files.
        temp_dir (Path): Temporary directory for runtime operations.
        model_dir (Path): Directory for storing model files.
        dataset_dir (Path): Directory for storing dataset files.
        downloads_dir (Path): Directory for downloaded files.

    Raises:
        OSError: If initialization fails.
    """

    def __init__(
        self,
        app_name: str | None = None,
        custom_base_dir: str | Path | None = None,
    ):
        """Initializes resource paths with cross-platform compatibility."""
        self._lock = Lock()
        self.app_name = self._determine_app_name(app_name)
        self._initialize_paths(custom_base_dir)
        self._initialize_directories()
        logger.info(f"ResourceManager initialized for app: {self.app_name}")
        logger.debug(f"Resource directories: {self.get_all_directories()}")

    def __repr__(self) -> str:
        """String representation of ResourceManager."""
        return f"ResourceManager(app_name='{self.app_name}', " f"user_data_dir='{self.user_data_dir}')"

    @contextmanager
    def temp_workspace(self, prefix: str = "workspace", suffix: str = ""):
        """A context manager for temporary workspaces that auto-cleans on exit.

        Args:
            prefix (str): A prefix for the temporary directory name.
            suffix (str): A suffix for the temporary directory name.

        Yields:
            Path: The path to the temporary workspace.

        Example:
            >>> with resource_manager.temp_workspace("processing") as ws:
            ...     # Use ws for temporary operations
            ...     (ws / "temp.txt").write_text("data")
            # The workspace is automatically removed here.
        """
        try:
            # Create the temp directory inside our app's main temp_dir
            workspace_path_str = tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=self.temp_dir)
            workspace_path = Path(workspace_path_str)
            logger.info(f"Created temporary workspace: {workspace_path}")
            yield workspace_path
        finally:
            if "workspace_path" in locals() and workspace_path.exists():
                try:
                    shutil.rmtree(workspace_path)
                    logger.info(f"Cleaned up temporary workspace: {workspace_path}")
                except Exception as e:
                    # Log error but don't prevent context exit
                    logger.error(f"Failed to clean up workspace {workspace_path}: {e}")

    def clean_old_files(
        self,
        days: int = 5,
        include_cache: bool = True,
    ) -> dict[str, int]:
        """Cleans up old download and temporary files.

        Args:
            days (int): The number of days after which files are considered old.
            include_cache (bool): Whether to include the cache directory in cleanup.

        Returns:
            dict[str, int]: A dictionary with cleanup statistics.

        Raises:
            ValueError: If `days` is negative.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")

        cleanup_stats = {"downloads_cleaned": 0, "temp_cleaned": 0, "cache_cleaned": 0}
        cutoff_time = time.time() - (days * 86400)

        cleanup_stats["downloads_cleaned"] = self._clean_directory(
            self.downloads_dir,
            cutoff_time,
        )
        cleanup_stats["temp_cleaned"] = self._clean_directory(
            self.temp_dir,
            cutoff_time,
        )
        if include_cache:
            cleanup_stats["cache_cleaned"] = self._clean_directory(
                self.user_cache_dir,
                cutoff_time,
            )

        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def create_checksum(self, file_path: str | Path, algorithm: str = "md5") -> str:
        """Creates a checksum for a file.

        Args:
            file_path (str | Path): The path to the file.
            algorithm (str): The hashing algorithm to use ('md5', 'sha1', 'sha256').

        Returns:
            str: The hexadecimal checksum string.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If checksum creation fails.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            msg = f"File does not exist: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            msg = f"Failed to create checksum for {file_path}: {e}"
            logger.error(msg)
            raise OSError(msg) from e

    def get_all_directories(self) -> dict[str, Path]:
        """Gets all managed directories.

        Returns:
            dict[str, Path]: A dictionary mapping directory names to their paths.
        """
        return {
            "user_data_dir": self.user_data_dir,
            "user_cache_dir": self.user_cache_dir,
            "temp_dir": self.temp_dir,
            "model_dir": self.model_dir,
            "dataset_dir": self.dataset_dir,
            "downloads_dir": self.downloads_dir,
            "logs_dir": self.logs_dir,
            "config_dir": self.config_dir,
        }

    def get_cache_path(self, cache_name: str, create_if_missing: bool = True) -> Path:
        """Gets a path for cache files.

        Args:
            cache_name (str): The name of the cache.
            create_if_missing (bool): Whether to create the directory if it
                doesn't exist.

        Returns:
            Path: The path to the cache directory.
        """
        if not cache_name or not cache_name.strip():
            raise ValueError("Cache name cannot be empty")

        cache_path = self.user_cache_dir / self._sanitize_name(cache_name)
        if create_if_missing:
            self._create_directory(cache_path, "cache")
        return cache_path

    def get_dataset_path(
        self,
        dataset_name: str,
        create_if_missing: bool = True,
    ) -> Path:
        """Gets a standardized path for a specific dataset.

        Args:
            dataset_name (str): The name of the dataset.
            create_if_missing (bool): Whether to create the directory if it
                doesn't exist.

        Returns:
            Path: The absolute path to the dataset directory.
        """
        if not dataset_name or not dataset_name.strip():
            raise ValueError("Dataset name cannot be empty")

        dataset_path = self.dataset_dir / self._sanitize_name(dataset_name)
        if create_if_missing:
            self._create_directory(dataset_path, "dataset")
        return dataset_path

    def get_disk_usage(self) -> dict[str, dict[str, int | str]]:
        """Gets disk usage statistics for all managed directories.

        Returns:
            dict: A dictionary with disk usage information for each directory.
        """
        directories = {
            "user_data": self.user_data_dir,
            "cache": self.user_cache_dir,
            "models": self.model_dir,
            "datasets": self.dataset_dir,
            "downloads": self.downloads_dir,
            "temp": self.temp_dir,
        }
        return {name: self._get_directory_size(path) for name, path in directories.items()}

    def get_model_path(self, model_name: str, create_if_missing: bool = True) -> Path:
        """Gets a standardized path for a specific model.

        Args:
            model_name (str): The name of the model.
            create_if_missing (bool): Whether to create the directory if it
                doesn't exist.

        Returns:
            Path: The absolute path to the model directory.
        """
        if not model_name or not model_name.strip():
            raise ValueError("Model name cannot be empty")

        model_path = self.model_dir / self._sanitize_name(model_name)
        if create_if_missing:
            self._create_directory(model_path, "model")
        return model_path

    def verify_checksum(
        self,
        file_path: str | Path,
        expected_checksum: str,
        algorithm: str = "md5",
    ) -> bool:
        """Verifies a file's checksum.

        Args:
            file_path (str | Path): The path to the file.
            expected_checksum (str): The expected checksum value.
            algorithm (str): The hashing algorithm used.

        Returns:
            bool: True if the checksum matches, False otherwise.
        """
        try:
            actual_checksum = self.create_checksum(file_path, algorithm)
            return actual_checksum.lower() == expected_checksum.lower()
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Checksum verification failed: {e}")
            return False

    def _clean_directory(self, directory: Path, cutoff_time: float) -> int:
        """Cleans files older than `cutoff_time` in a directory."""
        cleaned_count = 0
        if not directory.exists():
            return cleaned_count

        try:
            for item in directory.iterdir():
                try:
                    if item.stat().st_mtime < cutoff_time:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned old file/directory: {item}")
                except Exception as e:
                    logger.warning(f"Could not remove {item}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
        return cleaned_count

    def _create_directory(self, path: Path, dir_type: str) -> None:
        """Helper method to create a directory."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            msg = f"Failed to create {dir_type} directory {path}: {e}"
            logger.error(msg)
            raise OSError(msg) from e

    def _determine_app_name(self, app_name: str | None = None) -> str:
        """Determines the application name from various sources."""
        if app_name:
            return app_name
        try:
            # Try to get the project name from pyproject.toml
            app_name = self._get_project_name_from_pyproject()
            if app_name:
                return app_name
        except Exception as e:
            logger.warning(
                f"Could not find installed package metadata. {e}" "Falling back to default app name 'culicidaelab'.",
            )
        return "culicidaelab"

    def _get_project_name_from_pyproject(self) -> str | None:
        """Gets the project name from the pyproject.toml file."""
        try:
            # Find the root of the project
            root_dir = os.path.dirname(os.path.abspath(__file__))
            while not os.path.exists(os.path.join(root_dir, "pyproject.toml")):
                parent_dir = os.path.dirname(root_dir)
                if parent_dir == root_dir:
                    # We've reached the top, no pyproject.toml found
                    return None
                root_dir = parent_dir

            # Read the pyproject.toml file
            pyproject_path = os.path.join(root_dir, "pyproject.toml")
            with open(pyproject_path, encoding="utf-8") as f:
                pyproject_data = toml.load(f)

            # Get the project name from the pyproject.toml data
            project_name = pyproject_data.get("project", {}).get("name")
            return project_name
        except Exception as e:
            logger.error(f"Failed to read project name from pyproject.toml: {e}")
            return None

    def _format_bytes(self, bytes_count: int | float) -> str:
        """Formats bytes into a human-readable string."""
        if bytes_count is None:
            raise ValueError("bytes_count must not be None")
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        for unit in units:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} {units[-1]}"

    def _get_directory_size(self, path: Path) -> dict[str, int | str]:
        """Gets size information for a directory."""
        if not path.exists():
            return {"size_bytes": 0, "size_human": "0 B", "file_count": 0}

        total_size = 0
        file_count = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
        except Exception as e:
            logger.warning(f"Error calculating size for {path}: {e}")

        return {
            "size_bytes": total_size,
            "size_human": self._format_bytes(total_size),
            "file_count": file_count,
        }

    def _initialize_directories(self) -> None:
        """Creates necessary directories with proper permissions."""
        directories = self.get_all_directories().values()
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {directory}")
            except Exception as e:
                msg = f"Failed to create directory {directory}: {e}"
                logger.error(msg)
                raise OSError(msg) from e

        if platform.system() != "Windows":
            self._set_directory_permissions(list(directories))

    def _initialize_paths(self, custom_base_dir: str | Path | None = None) -> None:
        """Initializes all resource paths."""
        if custom_base_dir:
            base_dir = Path(custom_base_dir).resolve()
            self.user_data_dir = base_dir / "data"
            self.user_cache_dir = base_dir / "cache"
        else:
            self.user_data_dir = Path(appdirs.user_data_dir(self.app_name))
            self.user_cache_dir = Path(appdirs.user_cache_dir(self.app_name))

        self.temp_dir = Path(tempfile.gettempdir()) / self.app_name
        self.model_dir = self.user_data_dir / "models"
        self.dataset_dir = self.user_data_dir / "datasets"
        self.downloads_dir = self.user_data_dir / "downloads"
        self.logs_dir = self.user_data_dir / "logs"
        self.config_dir = self.user_data_dir / "config"

    def _is_safe_to_delete(self, path: Path) -> bool:
        """Checks if a path is safe to delete (i.e., within managed dirs)."""
        safe_parents = [self.temp_dir, self.user_cache_dir]
        try:
            resolved_path = path.resolve()
            return any(str(resolved_path).startswith(str(p.resolve())) for p in safe_parents)
        except Exception:
            return False

    def _sanitize_name(self, name: str) -> str:
        """Sanitizes a name for use as a directory/file name."""
        import re

        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name).strip(". ")
        return sanitized or "unnamed"

    def _set_directory_permissions(self, directories: list[Path]) -> None:
        """Sets directory permissions on Unix-like systems (0o700)."""
        try:
            for directory in directories:
                os.chmod(directory, 0o700)
        except Exception as e:
            logger.warning(f"Could not set directory permissions: {e}")
