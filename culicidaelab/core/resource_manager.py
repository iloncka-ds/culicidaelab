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

from culicidaelab.core.utils import create_safe_path

logger = logging.getLogger(__name__)


class ResourceManager:
    """Centralized resource management for models, datasets, and temporary files.

    This class provides thread-safe operations for managing application resources,
    including models, datasets, cache files, and temporary workspaces. It ensures
    that all file operations are handled in a consistent and safe manner.

    Args:
        app_name (str, optional): The name of the application, used for creating
            dedicated directories. If not provided, it is inferred from the
            `pyproject.toml` file. Defaults to None.
        custom_base_dir (str | Path, optional): A custom base directory for
            storing all resources. If None, system-appropriate default
            directories are used (e.g., AppData on Windows). Defaults to None.

    Attributes:
        app_name (str): The application name.
        user_data_dir (Path): The root directory for user-specific data.
        user_cache_dir (Path): The directory for user-specific cache files.
        temp_dir (Path): The directory for temporary runtime files.
        model_dir (Path): The directory where model files are stored.
        dataset_dir (Path): The directory where datasets are stored.
        downloads_dir (Path): The directory for downloaded files.
        logs_dir (Path): The directory for log files.
        config_dir (Path): The directory for configuration files.

    Raises:
        OSError: If the resource directories cannot be created.
        ValueError: If the application name cannot be determined.
    """

    def __init__(
        self,
        app_name: str | None = None,
        custom_base_dir: str | Path | None = None,
    ):
        """Initializes the ResourceManager with cross-platform compatibility.

        Sets up the necessary directory structure for the application's resources.
        """
        self._lock = Lock()
        self.app_name = self._determine_app_name(app_name)
        self._initialize_paths(custom_base_dir)
        self._initialize_directories()
        logger.info(f"ResourceManager initialized for app: {self.app_name}")
        logger.debug(f"Resource directories: {self.get_all_directories()}")

    def __repr__(self) -> str:
        """Returns a string representation of the ResourceManager instance.

        Returns:
            str: A string representation of the object.
        """
        return f"ResourceManager(app_name='{self.app_name}', " f"user_data_dir='{self.user_data_dir}')"

    @contextmanager
    def temp_workspace(self, prefix: str = "workspace", suffix: str = ""):
        """Provides a temporary workspace that is automatically cleaned up.

        This context manager creates a temporary directory and yields its path,
        ensuring the directory and its contents are removed upon exiting the
        context, even if errors occur.

        Args:
            prefix (str): A prefix for the temporary directory's name.
            suffix (str): A suffix for the temporary directory's name.

        Yields:
            Path: The path to the temporary workspace.

        Example:
            >>> resource_manager = ResourceManager()
            >>> with resource_manager.temp_workspace(prefix="job_") as ws:
            ...     # Perform temporary operations within this workspace
            ...     (ws / "temp_file.txt").write_text("some data")
            ...     print(f"Workspace created at: {ws}")
            # The workspace directory is automatically removed here.
        """
        workspace_path = None
        try:
            # Create the temp directory inside the app's main temp_dir
            workspace_path_str = tempfile.mkdtemp(
                prefix=prefix,
                suffix=suffix,
                dir=self.temp_dir,
            )
            workspace_path = Path(workspace_path_str)
            logger.info(f"Created temporary workspace: {workspace_path}")
            yield workspace_path
        finally:
            if workspace_path and workspace_path.exists():
                try:
                    shutil.rmtree(workspace_path)
                    logger.info(f"Cleaned up temporary workspace: {workspace_path}")
                except Exception as e:
                    # Log the error but do not raise it to avoid masking other exceptions
                    logger.error(
                        f"Failed to clean up workspace {workspace_path}: {e}",
                    )

    def clean_old_files(
        self,
        days: int = 5,
        include_cache: bool = True,
    ) -> dict[str, int]:
        """Cleans up old files from download and temporary directories.

        Args:
            days (int): The age in days for a file to be considered old.
            include_cache (bool): If True, the cache directory is also cleaned.

        Returns:
            dict[str, int]: A dictionary containing statistics of the cleanup.

        Raises:
            ValueError: If `days` is a negative number.
        """
        if days < 0:
            raise ValueError("Days must be a non-negative number.")

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
        """Creates a checksum for a given file.

        Args:
            file_path (str | Path): The path to the file.
            algorithm (str): The hashing algorithm to use (e.g., 'md5', 'sha256').

        Returns:
            str: The hexadecimal checksum string.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            OSError: If there is an error reading the file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                # Read the file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            msg = f"Failed to create checksum for {file_path}: {e}"
            logger.error(msg)
            raise OSError(msg) from e

    def get_all_directories(self) -> dict[str, Path]:
        """Retrieves all managed directory paths.

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

    def get_dataset_path(
        self,
        dataset_name: str,
        create_if_missing: bool = True,
    ) -> Path:
        """Constructs a standardized path for a dataset.

        Args:
            dataset_name (str): The name of the dataset.
            create_if_missing (bool): If True, creates the directory if it
                does not exist.

        Returns:
            Path: The absolute path to the dataset directory.

        Raises:
            ValueError: If `dataset_name` is empty or contains only whitespace.
        """
        if not dataset_name or not dataset_name.strip():
            raise ValueError("Dataset name cannot be empty.")

        safe_dataset_name = create_safe_path(dataset_name)
        dataset_path = self.dataset_dir / safe_dataset_name
        if create_if_missing:
            self._create_directory(dataset_path, "dataset")
        return dataset_path

    def get_disk_usage(self) -> dict[str, dict[str, int | str]]:
        """Calculates disk usage for all managed directories.

        Returns:
            dict: A dictionary with disk usage details for each directory,
                  including size in bytes, human-readable size, and file count.
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

    def verify_checksum(
        self,
        file_path: str | Path,
        expected_checksum: str,
        algorithm: str = "md5",
    ) -> bool:
        """Verifies the checksum of a file against an expected value.

        Args:
            file_path (str | Path): The path to the file.
            expected_checksum (str): The expected checksum.
            algorithm (str): The hashing algorithm used for the checksum.

        Returns:
            bool: True if the checksums match, False otherwise.
        """
        try:
            actual_checksum = self.create_checksum(file_path, algorithm)
            return actual_checksum.lower() == expected_checksum.lower()
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Checksum verification failed for {file_path}: {e}")
            return False

    def _clean_directory(self, directory: Path, cutoff_time: float) -> int:
        """Removes files in a directory older than a specified time."""
        cleaned_count = 0
        if not directory.exists():
            return cleaned_count

        try:
            for item in directory.iterdir():
                try:
                    # Check if the item's modification time is older than the cutoff
                    if item.stat().st_mtime < cutoff_time:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        cleaned_count += 1
                        logger.debug(f"Removed old item: {item}")
                except Exception as e:
                    logger.warning(f"Could not remove {item}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
        return cleaned_count

    def _create_directory(self, path: Path, dir_type: str) -> None:
        """Creates a directory if it doesn't exist."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            msg = f"Failed to create {dir_type} directory at {path}: {e}"
            logger.error(msg)
            raise OSError(msg) from e

    def _determine_app_name(self, app_name: str | None = None) -> str:
        """Determines the application name."""
        if app_name:
            return app_name
        try:
            # Attempt to get the project name from pyproject.toml
            pyproject_name = self._get_project_name_from_pyproject()
            if pyproject_name:
                return pyproject_name
        except Exception as e:
            logger.warning(
                f"Could not determine app name from pyproject.toml: {e}. " "Falling back to default 'culicidaelab'.",
            )
        return "culicidaelab"

    def _get_project_name_from_pyproject(self) -> str | None:
        """Reads the project name from the pyproject.toml file."""
        try:
            # Traverse up to find the project root containing pyproject.toml
            current_dir = Path(__file__).parent
            while not (current_dir / "pyproject.toml").exists():
                if current_dir.parent == current_dir:  # Reached the filesystem root
                    return None
                current_dir = current_dir.parent

            pyproject_path = current_dir / "pyproject.toml"
            with open(pyproject_path, encoding="utf-8") as f:
                pyproject_data = toml.load(f)

            return pyproject_data.get("project", {}).get("name")
        except Exception as e:
            logger.error(f"Failed to read project name from pyproject.toml: {e}")
            return None

    def _format_bytes(self, bytes_count: int | float) -> str:
        """Formats a byte count into a human-readable string."""
        import math

        if bytes_count is None:
            raise ValueError("bytes_count cannot be None.")
        if bytes_count == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        # Determine the appropriate unit using logarithm
        power = int(math.log(bytes_count, 1024)) if bytes_count > 0 else 0
        unit_index = min(power, len(units) - 1)
        value = bytes_count / (1024**unit_index)
        return f"{value:.1f} {units[unit_index]}"

    def _get_directory_size(self, path: Path) -> dict[str, int | str]:
        """Calculates the total size and file count of a directory."""
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
        """Creates all necessary application directories."""
        directories = self.get_all_directories().values()
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                msg = f"Failed to create directory {directory}: {e}"
                logger.error(msg)
                raise OSError(msg) from e

        # Set secure permissions on non-Windows systems
        if platform.system() != "Windows":
            self._set_directory_permissions(list(directories))

    def _initialize_paths(self, custom_base_dir: str | Path | None = None) -> None:
        """Initializes all resource paths based on the environment."""
        if custom_base_dir:
            base_dir = Path(custom_base_dir).resolve()
            self.user_data_dir = base_dir / "data"
            self.user_cache_dir = base_dir / "cache"
        else:
            # Use system-appropriate directories
            self.user_data_dir = Path(appdirs.user_data_dir(self.app_name))
            self.user_cache_dir = Path(appdirs.user_cache_dir(self.app_name))

        self.temp_dir = Path(tempfile.gettempdir()) / self.app_name
        self.model_dir = self.user_data_dir / "models"
        self.dataset_dir = self.user_data_dir / "datasets"
        self.downloads_dir = self.user_data_dir / "downloads"
        self.logs_dir = self.user_data_dir / "logs"
        self.config_dir = self.user_data_dir / "config"

    def _is_safe_to_delete(self, path: Path) -> bool:
        """Checks if a path is within a managed directory and safe to delete."""
        safe_parents = [self.temp_dir, self.user_cache_dir]
        try:
            resolved_path = path.resolve()
            # Ensure the path is a child of one of the safe parent directories
            return any(resolved_path.is_relative_to(p.resolve()) for p in safe_parents)
        except Exception:
            return False

    def _set_directory_permissions(self, directories: list[Path]) -> None:
        """Sets directory permissions to 0o700 on Unix-like systems."""
        try:
            for directory in directories:
                os.chmod(directory, 0o700)
        except Exception as e:
            logger.warning(f"Could not set directory permissions: {e}")
